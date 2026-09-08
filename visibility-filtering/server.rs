use crate::filter_tweets::FilterTweetsEndpoint;
use crate::get_safety_labels::GetSafetyLabelsEndpoint;
use std::sync::Arc;
use tonic::codec::CompressionEncoding;
use tonic::{Request, Response, Status};
use xai_visibility_filtering_proto as vf_pb;

pub struct VFServer {
    filter_tweets: FilterTweetsEndpoint,
    get_safety_labels: GetSafetyLabelsEndpoint,
}

#[tonic::async_trait]
impl xai_x_service_builder::XService for VFServer {
    type Config = ();

    async fn build(ctx: xai_x_service_builder::ServiceContext<()>) -> Self {
        VFServer::new(&ctx.datacenter, ctx.feature_switches).await
    }

    fn register(self: Arc<Self>, routes: &mut tonic::service::RoutesBuilder) {
        routes.add_service(
            vf_pb::VisibilityFilteringServiceServer::from_arc(self)
                .accept_compressed(CompressionEncoding::Zstd)
                .accept_compressed(CompressionEncoding::Gzip),
        );
    }
}

impl VFServer {
    pub(crate) async fn new(
        datacenter: &str,
        feature_switches: Arc<xai_feature_switches::FeatureSwitches>,
    ) -> Self {
        crate::server_deps::build_prod_server(datacenter, feature_switches).await
    }

    pub(crate) fn from_endpoints(
        filter_tweets: FilterTweetsEndpoint,
        get_safety_labels: GetSafetyLabelsEndpoint,
    ) -> Self {
        Self {
            filter_tweets,
            get_safety_labels,
        }
    }
}

#[tonic::async_trait]
impl vf_pb::VisibilityFilteringService for VFServer {
    async fn filter_tweets(
        &self,
        request: Request<vf_pb::VisibilityFilterRequest>,
    ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
        self.filter_tweets.handle(request).await
    }

    async fn get_safety_labels(
        &self,
        request: Request<vf_pb::GetSafetyLabelsRequest>,
    ) -> Result<Response<vf_pb::GetSafetyLabelsResponse>, Status> {
        self.get_safety_labels.handle(request).await
    }
}
