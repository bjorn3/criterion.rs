use std::collections::BTreeMap;

pub(crate) use self::implementation::*;

#[cfg(any(not(feature = "pmu"), not(target_os = "linux")))]
pub mod implementation {
    use super::*;

    #[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Serialize)]
    pub struct EventName;

    pub(crate) fn measure_fn<F, T>(mut function: F) -> (T, Option<BTreeMap<EventName, u64>>)
    where
        F: FnMut() -> T,
    {
        (function(), None)
    }
}

// TODO make this pub(crate)
#[cfg(all(feature = "pmu", target_os = "linux"))]
pub mod implementation {
    use super::*;

    #[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Serialize)]
    pub struct EventName(::perf_events::events::Event);

    pub(crate) fn measure_fn<F, T>(mut function: F) -> (T, Option<BTreeMap<EventName, u64>>)
    where
        F: FnMut() -> T,
    {
        let counter = ::perf_events::Counts::start_all_available();

        let returned = function();

        let metrics = counter.map(|mut c| {
            c.read()
                .into_iter()
                .map(|(e, v)| (EventName(e), v))
                .collect()
        });

        (returned, metrics.ok())
    }
}
