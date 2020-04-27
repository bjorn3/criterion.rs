use std::path::Path;
use std::collections::BTreeMap;

use stats::{Distribution, Tails};
use stats::bivariate::Data;
use stats::bivariate::regression::Slope;
use stats::univariate::Sample;
use stats::univariate::outliers::tukey::{self, LabeledSample};

use estimate::{Distributions, Estimates, Statistic};
use metrics::EventName;
use routine::Routine;
use benchmark::BenchmarkConfig;
use {ConfidenceInterval, Criterion, Estimate, Throughput};
use {format, fs};
use report::{BenchmarkId, ReportContext};

macro_rules! elapsed {
    ($msg:expr, $block:expr) => ({
        let start = ::std::time::Instant::now();
        let out = $block;
        let elapsed = &start.elapsed();

        info!("{} took {}", $msg, format::time(::DurationExt::to_nanos(elapsed) as f64));

        out
    })
}

mod compare;

// Common analysis procedure
pub(crate) fn common<T>(
    id: &BenchmarkId,
    routine: &mut Routine<T>,
    config: &BenchmarkConfig,
    criterion: &Criterion,
    report_context: &ReportContext,
    parameter: &T,
    throughput: Option<Throughput>,
) {
    let (iters, times, raw_metrics) =
        routine.sample(id, config, criterion, report_context, parameter);

    let data = Data::new(&iters, &times);
    let (distribution, slope) = regression(data, config);
}

// Performs a simple linear regression on the sample
pub(super) fn regression(data: Data<f64, f64>, config: &BenchmarkConfig) -> (Distribution<f64>, Estimate) {
    let _ = data.bootstrap(config.nresamples, |d| (Slope::fit(d).0,));

    std::process::exit(0);
}

// Classifies the outliers in the sample
fn outliers<'a, P: AsRef<Path>>(
    new_directory: P,
    avg_times: &'a Sample<f64>,
) -> LabeledSample<'a, f64> {
    let sample = tukey::classify(avg_times);
    log_if_err!(fs::save(
        &sample.fences(),
        &new_directory.as_ref().join("tukey.json")
    ));
    sample
}

// Estimates the statistics of the population from the sample
fn estimates(avg_times: &Sample<f64>, config: &BenchmarkConfig) -> (Distributions, Estimates) {
    fn stats(sample: &Sample<f64>) -> (f64, f64, f64, f64) {
        let mean = sample.mean();
        let std_dev = sample.std_dev(Some(mean));
        let median = sample.percentiles().median();
        let mad = sample.median_abs_dev(Some(median));

        (mean, std_dev, median, mad)
    }

    let cl = config.confidence_level;
    let nresamples = config.nresamples;

    let (mean, std_dev, median, mad) = stats(avg_times);
    let mut point_estimates = BTreeMap::new();
    point_estimates.insert(Statistic::Mean, mean);
    point_estimates.insert(Statistic::StdDev, std_dev);
    point_estimates.insert(Statistic::Median, median);
    point_estimates.insert(Statistic::MedianAbsDev, mad);

    let (dist_mean, dist_stddev, dist_median, dist_mad) = elapsed!(
        "Bootstrapping the absolute statistics.",
        avg_times.bootstrap(nresamples, stats)
    );

    let mut distributions = Distributions::new();
    distributions.insert(Statistic::Mean, dist_mean);
    distributions.insert(Statistic::StdDev, dist_stddev);
    distributions.insert(Statistic::Median, dist_median);
    distributions.insert(Statistic::MedianAbsDev, dist_mad);

    let estimates = Estimate::new(&distributions, &point_estimates, cl);

    (distributions, estimates)
}

fn rename_new_dir_to_base<P: AsRef<Path>>(base: P, new: P) {
    let (base, new) = (base.as_ref(), new.as_ref());
    if base.exists() {
        try_else_return!(fs::rmrf(base));
    }
    if new.exists() {
        try_else_return!(fs::mv(new, base));
    };
}
