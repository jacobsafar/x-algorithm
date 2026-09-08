use super::lookup::LookupError;
use xai_visibility_filtering_proto as vf_pb;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum LabelSource {
    Twemcache,
    Manhattan,
}

impl LabelSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Twemcache => "twemcache",
            Self::Manhattan => "manhattan",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FallbackReason {
    Timeout,
    Backpressure,
    Decode,
    MissingResponse,
    Other,
}

impl FallbackReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Backpressure => "backpressure",
            Self::Decode => "decode",
            Self::MissingResponse => "missing_response",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, enum_map::Enum)]
pub(crate) enum FailureKind {
    ManhattanFetch,
    ManhattanDecode,
}

impl FailureKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ManhattanFetch => "manhattan_fetch",
            Self::ManhattanDecode => "manhattan_decode",
        }
    }
}

#[derive(Debug)]
pub(crate) enum TwemcacheOutcome {
    Hit(vf_pb::SafetyLabelMap),
    NotFound,
    Miss,
    FallThrough(FallbackReason),
}

#[derive(Debug)]
pub(crate) enum ManhattanOutcome {
    Resolved(vf_pb::SafetyLabelMap),
    Failure(LookupError),
}
