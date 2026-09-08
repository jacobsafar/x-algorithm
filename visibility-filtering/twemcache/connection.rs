use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
    ReadHalf, WriteHalf,
};
use tokio::net::TcpStream;
use tokio::sync::{oneshot, Mutex as TokioMutex};
use tokio::task::JoinHandle;
use tokio::time::timeout;

use super::error::{Result, TwemcacheError};
use super::key::{Key, Server, Value};
use super::metrics::{ConnEvent, Metrics};

const TCP_KEEPALIVE: Duration = Duration::from_secs(30);
const IDLE_PING_INTERVAL: Duration = Duration::from_secs(30);
const STALL_TIMEOUT: Duration = Duration::from_secs(10);

pub trait AsyncRW: AsyncRead + AsyncWrite + Unpin + Send + 'static {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send + 'static> AsyncRW for T {}

type BoxedStream = Box<dyn AsyncRW>;

enum Pending {
    Get(oneshot::Sender<Result<Option<Value>>>),
    Version(oneshot::Sender<Result<()>>),
}

impl Pending {
    fn fail(self, err: TwemcacheError) {
        match self {
            Pending::Get(tx) => {
                let _ = tx.send(Err(err));
            }
            Pending::Version(tx) => {
                let _ = tx.send(Err(err));
            }
        }
    }
}

struct Shared {
    pending: StdMutex<VecDeque<Pending>>,
    in_flight: AtomicUsize,
    dead: AtomicBool,
    metrics: Metrics,
}

impl Shared {
    fn mark_dead(&self, err: TwemcacheError) {
        if self.dead.swap(true, Ordering::AcqRel) {
            return;
        }
        self.metrics.record_connection_event(ConnEvent::TornDown);
        #[expect(clippy::unwrap_used, reason = "propagate FIFO-lock poisoning")]
        let mut q = self.pending.lock().unwrap();
        while let Some(p) = q.pop_front() {
            p.fail(err.clone());
        }
        self.in_flight.store(0, Ordering::Release);
    }
}

enum Reply {
    Value { data: Vec<u8> },
    Miss,
    Version,
}

pub struct PipelinedConnection {
    shared: Arc<Shared>,
    writer: Arc<TokioMutex<WriteHalf<BoxedStream>>>,
    request_timeout: Duration,
    reader: JoinHandle<()>,
    idle_ping: JoinHandle<()>,
}

impl Drop for PipelinedConnection {
    fn drop(&mut self) {
        self.reader.abort();
        self.idle_ping.abort();
    }
}

impl PipelinedConnection {
    pub fn from_stream(stream: BoxedStream, request_timeout: Duration, metrics: Metrics) -> Self {
        Self::from_stream_with_stall(stream, request_timeout, STALL_TIMEOUT, metrics)
    }

    fn from_stream_with_stall(
        stream: BoxedStream,
        request_timeout: Duration,
        stall_timeout: Duration,
        metrics: Metrics,
    ) -> Self {
        let (read_half, write_half) = tokio::io::split(stream);
        let shared = Arc::new(Shared {
            pending: StdMutex::new(VecDeque::new()),
            in_flight: AtomicUsize::new(0),
            dead: AtomicBool::new(false),
            metrics,
        });
        let reader = tokio::spawn(reader_loop(
            BufReader::new(read_half),
            Arc::clone(&shared),
            stall_timeout,
        ));
        let writer = Arc::new(TokioMutex::new(write_half));
        let idle_ping = tokio::spawn(idle_ping_loop(
            Arc::clone(&shared),
            Arc::clone(&writer),
            request_timeout,
        ));
        Self {
            shared,
            writer,
            request_timeout,
            reader,
            idle_ping,
        }
    }

    pub async fn connect(
        server: &Server,
        tls: Option<&tokio_rustls::client::TlsConnector>,
        connect_timeout: Duration,
        request_timeout: Duration,
        metrics: Metrics,
    ) -> Result<Self> {
        let addr = server.address();
        let tcp = timeout(connect_timeout, TcpStream::connect(&addr))
            .await
            .map_err(|_| TwemcacheError::Timeout)?
            .map_err(|e| TwemcacheError::Io(format!("TCP connect failed: {e}")))?;

        tcp.set_nodelay(true).ok();
        let keepalive = socket2::TcpKeepalive::new().with_time(TCP_KEEPALIVE);
        socket2::SockRef::from(&tcp)
            .set_tcp_keepalive(&keepalive)
            .ok();

        let stream: BoxedStream = if let Some(connector) = tls {
            let domain: tokio_rustls::rustls::pki_types::ServerName<'static> = "localhost"
                .try_into()
                .map_err(|_| TwemcacheError::Io("invalid TLS domain".into()))?;
            let tls = timeout(connect_timeout, connector.connect(domain, tcp))
                .await
                .map_err(|_| TwemcacheError::Timeout)?
                .map_err(|e| TwemcacheError::Io(format!("TLS handshake failed: {e}")))?;
            Box::new(tls)
        } else {
            Box::new(tcp)
        };

        Ok(Self::from_stream(stream, request_timeout, metrics))
    }

    pub fn in_flight(&self) -> usize {
        self.shared.in_flight.load(Ordering::Acquire)
    }

    pub fn is_dead(&self) -> bool {
        self.shared.dead.load(Ordering::Acquire)
    }

    pub async fn get(&self, key: &Key) -> Result<Option<Value>> {
        if self.shared.dead.load(Ordering::Acquire) {
            return Err(TwemcacheError::Unavailable);
        }

        let (tx, rx) = oneshot::channel();
        let key_bytes = key.as_bytes();
        let len = 4 + key_bytes.len() + 2;
        let mut stack_buf = [0u8; 256 + 6];
        let fallback;
        #[expect(
            clippy::indexing_slicing,
            reason = "this arm requires len <= stack_buf.len()"
        )]
        let cmd: &[u8] = if len <= stack_buf.len() {
            stack_buf[..4].copy_from_slice(b"get ");
            stack_buf[4..4 + key_bytes.len()].copy_from_slice(key_bytes);
            stack_buf[4 + key_bytes.len()..len].copy_from_slice(b"\r\n");
            &stack_buf[..len]
        } else {
            fallback = format!("get {key}\r\n");
            fallback.as_bytes()
        };

        {
            let mut w = self.writer.lock().await;
            if self.shared.dead.load(Ordering::Acquire) {
                return Err(TwemcacheError::Unavailable);
            }
            {
                #[expect(clippy::unwrap_used, reason = "propagate FIFO-lock poisoning")]
                let mut q = self.shared.pending.lock().unwrap();
                q.push_back(Pending::Get(tx));
                self.shared.in_flight.fetch_add(1, Ordering::AcqRel);
            }

            if let Err(e) = w.write_all(cmd).await {
                drop(w);
                self.shared
                    .mark_dead(TwemcacheError::Io(format!("write failed: {e}")));
                return Err(TwemcacheError::Io(format!("write failed: {e}")));
            }
            if let Err(e) = w.flush().await {
                drop(w);
                self.shared
                    .mark_dead(TwemcacheError::Io(format!("flush failed: {e}")));
                return Err(TwemcacheError::Io(format!("flush failed: {e}")));
            }
        }

        match timeout(self.request_timeout, rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err(TwemcacheError::Unavailable),
            Err(_) => Err(TwemcacheError::Timeout),
        }
    }
}

async fn dispatch_version(
    shared: &Arc<Shared>,
    writer: &TokioMutex<WriteHalf<BoxedStream>>,
    request_timeout: Duration,
) -> Result<()> {
    if shared.dead.load(Ordering::Acquire) {
        return Err(TwemcacheError::Unavailable);
    }
    let (tx, rx) = oneshot::channel();
    {
        let mut w = writer.lock().await;
        if shared.dead.load(Ordering::Acquire) {
            return Err(TwemcacheError::Unavailable);
        }
        {
            #[expect(clippy::unwrap_used, reason = "propagate FIFO-lock poisoning")]
            let mut q = shared.pending.lock().unwrap();
            q.push_back(Pending::Version(tx));
            shared.in_flight.fetch_add(1, Ordering::AcqRel);
        }
        if let Err(e) = w.write_all(b"version\r\n").await {
            drop(w);
            shared.mark_dead(TwemcacheError::Io(format!("write failed: {e}")));
            return Err(TwemcacheError::Io(format!("write failed: {e}")));
        }
        if let Err(e) = w.flush().await {
            drop(w);
            shared.mark_dead(TwemcacheError::Io(format!("flush failed: {e}")));
            return Err(TwemcacheError::Io(format!("flush failed: {e}")));
        }
    }
    match timeout(request_timeout, rx).await {
        Ok(Ok(res)) => res,
        Ok(Err(_)) => Err(TwemcacheError::Unavailable),
        Err(_) => Err(TwemcacheError::Timeout),
    }
}

async fn idle_ping_loop(
    shared: Arc<Shared>,
    writer: Arc<TokioMutex<WriteHalf<BoxedStream>>>,
    request_timeout: Duration,
) {
    let mut interval = tokio::time::interval(IDLE_PING_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        if shared.dead.load(Ordering::Acquire) {
            return;
        }
        if shared.in_flight.load(Ordering::Acquire) > 0 {
            continue;
        }
        let _ = dispatch_version(&shared, &writer, request_timeout).await;
    }
}

async fn reader_loop(
    mut reader: BufReader<ReadHalf<BoxedStream>>,
    shared: Arc<Shared>,
    stall_timeout: Duration,
) {
    loop {
        match timeout(stall_timeout, read_one_reply(&mut reader)).await {
            Ok(Ok(Some(reply))) => {
                if deliver(&shared, reply).is_err() {
                    shared.mark_dead(TwemcacheError::Io("protocol desync".into()));
                    return;
                }
            }
            Ok(Ok(None)) => {
                shared.mark_dead(TwemcacheError::Io("connection closed".into()));
                return;
            }
            Ok(Err(e)) => {
                shared.mark_dead(e);
                return;
            }
            Err(_) => {
                if shared.in_flight.load(Ordering::Acquire) > 0 {
                    shared.mark_dead(TwemcacheError::Unavailable);
                    return;
                }
            }
        }
    }
}

fn deliver(shared: &Shared, reply: Reply) -> std::result::Result<(), ()> {
    let pending = {
        #[expect(clippy::unwrap_used, reason = "propagate FIFO-lock poisoning")]
        let mut q = shared.pending.lock().unwrap();
        match q.pop_front() {
            Some(p) => {
                shared.in_flight.fetch_sub(1, Ordering::AcqRel);
                p
            }
            None => return Err(()),
        }
    };

    match (pending, reply) {
        (Pending::Get(tx), Reply::Miss) => {
            let _ = tx.send(Ok(None));
        }
        (Pending::Get(tx), Reply::Value { data }) => {
            let _ = tx.send(Ok(Some(data)));
        }
        (Pending::Get(tx), Reply::Version) => {
            let _ = tx.send(Err(TwemcacheError::Io(
                "unexpected VERSION reply for get".into(),
            )));
        }
        (Pending::Version(tx), Reply::Version) => {
            let _ = tx.send(Ok(()));
        }
        (Pending::Version(tx), _) => {
            let _ = tx.send(Err(TwemcacheError::Io(
                "unexpected non-VERSION reply for version".into(),
            )));
        }
    }
    Ok(())
}

async fn read_one_reply<R: AsyncBufRead + Unpin>(reader: &mut R) -> Result<Option<Reply>> {
    let mut header = String::new();
    let n = reader
        .read_line(&mut header)
        .await
        .map_err(|e| TwemcacheError::Io(format!("read header: {e}")))?;
    if n == 0 {
        return Ok(None);
    }
    let header = header.trim_end_matches('\n').trim_end_matches('\r');

    if header == "END" {
        return Ok(Some(Reply::Miss));
    }
    if header.starts_with("VERSION") {
        return Ok(Some(Reply::Version));
    }
    if header.starts_with("ERROR")
        || header.starts_with("CLIENT_ERROR")
        || header.starts_with("SERVER_ERROR")
    {
        return Err(TwemcacheError::Io(format!("memcached: {header}")));
    }

    let parts: Vec<&str> = header.split_whitespace().collect();
    let (flags_part, byte_count_part) = match parts.as_slice() {
        ["VALUE", _key, flags, byte_count, ..] => (*flags, *byte_count),
        _ => {
            return Err(TwemcacheError::Io(format!(
                "unexpected response: {header:?}"
            )));
        }
    };
    let _flags: u32 = flags_part
        .parse()
        .map_err(|_| TwemcacheError::Io(format!("invalid flags: {header:?}")))?;
    let byte_count: usize = byte_count_part
        .parse()
        .map_err(|_| TwemcacheError::Io(format!("invalid byte count: {header:?}")))?;

    let mut data = vec![0u8; byte_count];
    reader
        .read_exact(&mut data)
        .await
        .map_err(|e| TwemcacheError::Io(format!("read value: {e}")))?;

    let mut crlf = [0u8; 2];
    reader
        .read_exact(&mut crlf)
        .await
        .map_err(|e| TwemcacheError::Io(format!("read value crlf: {e}")))?;

    let mut end = String::new();
    reader
        .read_line(&mut end)
        .await
        .map_err(|e| TwemcacheError::Io(format!("read END: {e}")))?;
    let end = end.trim_end_matches('\n').trim_end_matches('\r');
    if end != "END" {
        return Err(TwemcacheError::Io(format!("expected END, got {end:?}")));
    }

    Ok(Some(Reply::Value { data }))
}

pub fn build_mtls_connector(
    ca_cert_path: &str,
    client_cert_path: &str,
    client_key_path: &str,
) -> Result<tokio_rustls::client::TlsConnector> {
    use tokio_rustls::client::TlsConnector;
    use tokio_rustls::rustls::{client::ClientConfig, RootCertStore};

    let _ = rustls::crypto::CryptoProvider::install_default(
        rustls::crypto::aws_lc_rs::default_provider(),
    );

    let mut roots = RootCertStore::empty();
    for cert in load_certs(ca_cert_path)? {
        roots
            .add(cert)
            .map_err(|e| TwemcacheError::Io(format!("add CA cert: {e}")))?;
    }
    let verifier = ChainVerifyingSkipNameVerifier::new(Arc::new(roots))?;

    let client_certs = load_certs(client_cert_path)?;
    let client_key = load_key(client_key_path)?;

    let config = ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(verifier))
        .with_client_auth_cert(client_certs, client_key)
        .map_err(|e| TwemcacheError::Io(format!("client auth cert: {e}")))?;

    Ok(TlsConnector::from(Arc::new(config)))
}

fn load_certs(path: &str) -> Result<Vec<rustls::pki_types::CertificateDer<'static>>> {
    use std::io::BufReader as IoBufReader;
    let file = std::fs::File::open(path).map_err(|e| TwemcacheError::Io(e.to_string()))?;
    let mut reader = IoBufReader::new(file);
    let mut certs = Vec::new();
    while let Some(rustls_pemfile::Item::X509Certificate(cert)) =
        rustls_pemfile::read_one(&mut reader).map_err(|e| TwemcacheError::Io(e.to_string()))?
    {
        certs.push(cert);
    }
    if certs.is_empty() {
        return Err(TwemcacheError::Io(format!("no certs in {path}")));
    }
    Ok(certs)
}

fn load_key(path: &str) -> Result<rustls::pki_types::PrivateKeyDer<'static>> {
    use std::io::BufReader as IoBufReader;
    let file = std::fs::File::open(path).map_err(|e| TwemcacheError::Io(e.to_string()))?;
    let mut reader = IoBufReader::new(file);
    while let Some(item) =
        rustls_pemfile::read_one(&mut reader).map_err(|e| TwemcacheError::Io(e.to_string()))?
    {
        match item {
            rustls_pemfile::Item::Pkcs8Key(k) => {
                return Ok(rustls::pki_types::PrivateKeyDer::Pkcs8(k));
            }
            rustls_pemfile::Item::Sec1Key(k) => {
                return Ok(rustls::pki_types::PrivateKeyDer::Sec1(k));
            }
            rustls_pemfile::Item::Pkcs1Key(k) => {
                return Ok(rustls::pki_types::PrivateKeyDer::Pkcs1(k));
            }
            _ => continue,
        }
    }
    Err(TwemcacheError::Io(format!("no key in {path}")))
}

#[derive(Debug)]
struct ChainVerifyingSkipNameVerifier {
    inner: Arc<rustls::client::WebPkiServerVerifier>,
}

impl ChainVerifyingSkipNameVerifier {
    fn new(roots: Arc<rustls::RootCertStore>) -> Result<Self> {
        let inner = rustls::client::WebPkiServerVerifier::builder_with_provider(
            roots,
            Arc::new(rustls::crypto::aws_lc_rs::default_provider()),
        )
        .build()
        .map_err(|e| TwemcacheError::Io(format!("build server cert verifier: {e}")))?;
        Ok(Self { inner })
    }
}

impl rustls::client::danger::ServerCertVerifier for ChainVerifyingSkipNameVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer,
        intermediates: &[rustls::pki_types::CertificateDer],
        server_name: &rustls::pki_types::ServerName,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        use rustls::CertificateError;
        match self.inner.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        ) {
            Err(rustls::Error::InvalidCertificate(
                CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. },
            )) => Ok(rustls::client::danger::ServerCertVerified::assertion()),
            other => other,
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer,
        dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer,
        dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    fn key(s: &str) -> Key {
        Key::new(s.as_bytes().to_vec()).unwrap()
    }

    async fn read_get_key<R: AsyncBufRead + Unpin>(reader: &mut R) -> String {
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let line = line.trim_end_matches('\n').trim_end_matches('\r');
        line.strip_prefix("get ").unwrap().to_string()
    }

    fn conn(stream: tokio::io::DuplexStream, rt_ms: u64) -> Arc<PipelinedConnection> {
        Arc::new(PipelinedConnection::from_stream(
            Box::new(stream),
            Duration::from_millis(rt_ms),
            Metrics::disabled(),
        ))
    }

    #[tokio::test]
    async fn single_hit_with_zero_flags() {
        let (client, server) = tokio::io::duplex(1024);
        let conn = conn(client, 500);
        let (mut sr, mut sw) = tokio::io::split(server);
        let mut sr = BufReader::new(&mut sr);

        let task = {
            let conn = Arc::clone(&conn);
            tokio::spawn(async move { conn.get(&key("k")).await })
        };

        let k = read_get_key(&mut sr).await;
        assert_eq!(k, "k");
        sw.write_all(b"VALUE k 0 3\r\nabc\r\nEND\r\n")
            .await
            .unwrap();

        let got = task.await.unwrap().unwrap();
        assert_eq!(got, Some(b"abc".to_vec()));
        assert_eq!(conn.in_flight(), 0);
        assert!(!conn.is_dead());
    }

    #[tokio::test]
    async fn single_miss() {
        let (client, server) = tokio::io::duplex(1024);
        let conn = conn(client, 500);
        let (mut sr, mut sw) = tokio::io::split(server);
        let mut sr = BufReader::new(&mut sr);

        let task = {
            let conn = Arc::clone(&conn);
            tokio::spawn(async move { conn.get(&key("missing")).await })
        };

        read_get_key(&mut sr).await;
        sw.write_all(b"END\r\n").await.unwrap();

        assert_eq!(task.await.unwrap().unwrap(), None);
    }

    #[tokio::test]
    async fn nonzero_flags_are_accepted() {
        let (client, server) = tokio::io::duplex(1024);
        let conn = conn(client, 500);
        let (mut sr, mut sw) = tokio::io::split(server);
        let mut sr = BufReader::new(&mut sr);

        let task = {
            let conn = Arc::clone(&conn);
            tokio::spawn(async move { conn.get(&key("k")).await })
        };

        read_get_key(&mut sr).await;
        sw.write_all(b"VALUE k 5 3\r\nabc\r\nEND\r\n")
            .await
            .unwrap();

        assert_eq!(task.await.unwrap().unwrap(), Some(b"abc".to_vec()));
        assert!(!conn.is_dead());
    }

    #[tokio::test]
    async fn pipelined_responses_match_requests_in_order() {
        let (client, server) = tokio::io::duplex(4096);
        let conn = conn(client, 1000);
        let (mut sr, mut sw) = tokio::io::split(server);
        let mut sr = BufReader::new(&mut sr);

        let mut tasks = Vec::new();
        for k in ["k1", "k2", "k3"] {
            let conn = Arc::clone(&conn);
            tasks.push(tokio::spawn(async move {
                let v = conn.get(&key(k)).await.unwrap();
                (k, v)
            }));
        }

        for _ in 0..3 {
            let k = read_get_key(&mut sr).await;
            let resp = format!("VALUE {k} 0 {}\r\n{k}\r\nEND\r\n", k.len());
            sw.write_all(resp.as_bytes()).await.unwrap();
        }

        for t in tasks {
            let (k, v) = t.await.unwrap();
            assert_eq!(v, Some(k.as_bytes().to_vec()));
        }
        assert_eq!(conn.in_flight(), 0);
    }

    #[tokio::test]
    async fn late_arrival_is_discarded_connection_stays_usable() {
        let (client, server) = tokio::io::duplex(1024);
        let conn = conn(client, 50);
        let (sr, mut sw) = tokio::io::split(server);
        let mut sr = BufReader::new(sr);

        let server_task = tokio::spawn(async move {
            let k1 = read_get_key(&mut sr).await;
            let k2 = read_get_key(&mut sr).await;
            let r1 = format!("VALUE {k1} 0 3\r\nabc\r\nEND\r\n");
            let r2 = format!("VALUE {k2} 0 3\r\ndef\r\nEND\r\n");
            sw.write_all(r1.as_bytes()).await.unwrap();
            sw.write_all(r2.as_bytes()).await.unwrap();
            tokio::time::sleep(Duration::from_secs(10)).await;
        });

        assert!(matches!(
            conn.get(&key("k1")).await,
            Err(TwemcacheError::Timeout)
        ));

        assert_eq!(conn.get(&key("k2")).await.unwrap(), Some(b"def".to_vec()));
        assert!(!conn.is_dead());

        server_task.abort();
    }

    #[tokio::test]
    async fn slow_response_within_stall_window_keeps_connection_alive() {
        let (client, server) = tokio::io::duplex(1024);
        let conn = conn(client, 30);
        let (sr, mut sw) = tokio::io::split(server);
        let mut sr = BufReader::new(sr);

        let server_task = tokio::spawn(async move {
            let k1 = read_get_key(&mut sr).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
            let r1 = format!("VALUE {k1} 0 3\r\nabc\r\nEND\r\n");
            sw.write_all(r1.as_bytes()).await.unwrap();
            loop {
                let k = read_get_key(&mut sr).await;
                let r = format!("VALUE {k} 0 3\r\ndef\r\nEND\r\n");
                sw.write_all(r.as_bytes()).await.unwrap();
            }
        });

        assert!(matches!(
            conn.get(&key("k1")).await,
            Err(TwemcacheError::Timeout)
        ));
        assert!(!conn.is_dead());

        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(!conn.is_dead());

        assert_eq!(conn.get(&key("k2")).await.unwrap(), Some(b"def".to_vec()));
        assert!(!conn.is_dead());

        server_task.abort();
    }

    #[tokio::test]
    async fn stall_tears_down_connection() {
        let (client, server) = tokio::io::duplex(1024);
        let conn = Arc::new(PipelinedConnection::from_stream_with_stall(
            Box::new(client),
            Duration::from_millis(30),
            Duration::from_millis(60),
            Metrics::disabled(),
        ));
        let (sr, _sw) = tokio::io::split(server);
        let mut sr = BufReader::new(sr);

        let server_task = tokio::spawn(async move {
            read_get_key(&mut sr).await;
            tokio::time::sleep(Duration::from_millis(500)).await;
        });

        assert!(matches!(
            conn.get(&key("k")).await,
            Err(TwemcacheError::Timeout)
        ));

        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(conn.is_dead());
        assert!(matches!(
            conn.get(&key("k2")).await,
            Err(TwemcacheError::Unavailable)
        ));

        server_task.abort();
    }
}
