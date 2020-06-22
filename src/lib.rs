#[cfg(feature = "html_reports")]
extern crate criterion_plot;
extern crate criterion_stats as stats;
extern crate failure;
#[cfg(feature = "html_reports")]
extern crate handlebars;
extern crate itertools;
extern crate serde;
#[macro_use]
extern crate log;
#[macro_use]
extern crate failure_derive;
#[macro_use]
extern crate serde_derive;
#[macro_use]
mod macros_private {
    macro_rules! try_else_return {
        ( $ x : expr ) => {
            try_else_return!($x, || {});
        };
        ( $ x : expr , $ el : expr ) => {
            match $x {
                Ok(x) => x,
                Err(e) => {
                    ::error::log_error(&e);
                    let closure = $el;
                    return closure();
                }
            }
        };
    }
}
#[macro_use]
mod analysis {
    macro_rules! elapsed {
        ( $ msg : expr , $ block : expr ) => {{
            let start = ::std::time::Instant::now();
            let out = $block;
            let elapsed = &start.elapsed();
            info!(
                "{} took {}",
                $msg,
                format::time(::DurationExt::to_nanos(elapsed) as f64)
            );
            out
        }};
    }
}
mod benchmark {
    use report::{BenchmarkId, ReportContext};
    use routine::{Function, Routine};
    use std::cell::RefCell;
    use std::fmt::Debug;
    use std::time::Duration;
    use {Bencher, Criterion, DurationExt, PlotConfiguration, Throughput};
    #[derive(Debug)]
    pub struct BenchmarkConfig {
        pub confidence_level: f64,
        pub measurement_time: Duration,
        pub noise_threshold: f64,
        pub nresamples: usize,
        pub sample_size: usize,
        pub significance_level: f64,
        pub warm_up_time: Duration,
    }
    struct PartialBenchmarkConfig {
        confidence_level: Option<f64>,
        measurement_time: Option<Duration>,
        noise_threshold: Option<f64>,
        nresamples: Option<usize>,
        sample_size: Option<usize>,
        significance_level: Option<f64>,
        warm_up_time: Option<Duration>,
        plot_config: PlotConfiguration,
    }
    impl Default for PartialBenchmarkConfig {
        fn default() -> Self {
            PartialBenchmarkConfig {
                confidence_level: None,
                measurement_time: None,
                noise_threshold: None,
                nresamples: None,
                sample_size: None,
                significance_level: None,
                warm_up_time: None,
                plot_config: PlotConfiguration::default(),
            }
        }
    }
    pub struct NamedRoutine<T> {
        pub id: String,
        pub f: Box<RefCell<Routine<T>>>,
    }
    pub struct ParameterizedBenchmark<T: Debug> {
        config: PartialBenchmarkConfig,
        values: Vec<T>,
        routines: Vec<NamedRoutine<T>>,
        throughput: Option<Box<Fn(&T) -> Throughput>>,
    }
    pub struct Benchmark {
        config: PartialBenchmarkConfig,
        routines: Vec<NamedRoutine<()>>,
        throughput: Option<Throughput>,
    }
    pub trait BenchmarkDefinition: Sized {
        fn run(self, group_id: &str, c: &Criterion);
    }
    impl Benchmark {
        pub fn new<S, F>(id: S, f: F) -> Benchmark
        where
            S: Into<String>,
            F: FnMut(&mut Bencher) + 'static,
        {
            Benchmark {
                config: Default::default(),
                routines: vec![],
                throughput: None,
            }
            .with_function(id, f)
        }
        pub fn with_function<S, F>(mut self, id: S, mut f: F) -> Benchmark
        where
            S: Into<String>,
            F: FnMut(&mut Bencher) + 'static,
        {
            let routine = NamedRoutine {
                id: id.into(),
                f: Box::new(RefCell::new(Function::new(move |b, _| f(b)))),
            };
            self.routines.push(routine);
            self
        }
    }
    impl BenchmarkDefinition for Benchmark {
        fn run(self, group_id: &str, c: &Criterion) {
            let report_context = ReportContext {
                output_directory: c.output_directory.clone(),
                plotting: c.plotting,
                plot_config: self.config.plot_config.clone(),
            };
            let config = BenchmarkConfig {
                confidence_level: 0.5,
                measurement_time: Duration::from_nanos(1),
                noise_threshold: 0.01,
                nresamples: 1,
                sample_size: 2,
                significance_level: 0.05,
                warm_up_time: Duration::from_millis(0),
            };
            let id = BenchmarkId::new(group_id.to_owned(), None, None, None);
            let (iters, times, _raw_metrics) =
                Function::new(move |b, _| ()).sample(&id, &config, c, &report_context, &None::<()>);
            let data = stats::bivariate::Data::new(&[1.0, 2.0], &[1.0, 2.0]);
            data.bootstrap(1, |d| (0,));
        }
    }
}
mod error {
    use failure::Error;
    use std::io;
    use std::path::PathBuf;
    #[derive(Debug, Fail)]
    #[fail(display = "Failed to access file {:?}: {}", path, inner)]
    pub struct AccessError {
        pub path: PathBuf,
        #[cause]
        pub inner: io::Error,
    }
    pub type Result<T> = ::std::result::Result<T, Error>;
    pub(crate) fn log_error(e: &Error) {
        unimplemented!()
    }
}
mod estimate {
    use stats::Distribution;
    use std::collections::BTreeMap;
    use std::fmt;
    use Estimate;
    #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Deserialize, Serialize, Debug)]
    pub enum Statistic {
        Mean,
        Median,
        MedianAbsDev,
        Slope,
        StdDev,
    }
    pub(crate) type Estimates = BTreeMap<Statistic, Estimate>;
    pub(crate) type Distributions = BTreeMap<Statistic, Distribution<f64>>;
}
mod format {
    use Throughput;
    pub fn time(ns: f64) -> String {
        unimplemented!()
    }
    pub fn throughput(throughput: &Throughput, ns: f64) -> String {
        unimplemented!()
    }
}
mod fs {
    use error::{AccessError, Result};
    use serde::de::DeserializeOwned;
    use std::path::Path;
    pub fn mkdirp<P>(path: &P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        unimplemented!()
    }
    pub fn save_string<P>(data: &str, path: &P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        unimplemented!()
    }
}
mod metrics {
    pub(crate) use self::implementation::*;
    use std::collections::BTreeMap;
    #[cfg(any(not(feature = "pmu"), not(target_os = "linux")))]
    pub mod implementation {
        use super::*;
        #[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Serialize)]
        pub struct EventName;
        pub(crate) fn measure_fn<F, T>(mut function: F) -> (T, Option<BTreeMap<EventName, u64>>)
        where
            F: FnMut() -> T,
        {
            unimplemented!()
        }
    }
}
mod program {
    use std::io::BufReader;
    use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
    pub struct Program {
        buffer: String,
        stdin: ChildStdin,
        _child: Child,
        stderr: ChildStderr,
        stdout: BufReader<ChildStdout>,
    }
}
mod report {
    use estimate::{Distributions, Estimates, Statistic};
    use metrics::EventName;
    use stats::univariate::outliers::tukey::LabeledSample;
    use stats::univariate::Sample;
    use stats::Distribution;
    use std::cell::Cell;
    use std::collections::BTreeMap;
    use std::fmt;
    use std::path::PathBuf;
    use {PlotConfiguration, Plotting, Throughput};
    pub(crate) struct ComparisonData {
        pub p_value: f64,
        pub t_distribution: Distribution<f64>,
        pub t_value: f64,
        pub relative_estimates: Estimates,
        pub relative_distributions: Distributions,
        pub significance_threshold: f64,
        pub noise_threshold: f64,
        pub base_iter_counts: Vec<f64>,
        pub base_sample_times: Vec<f64>,
        pub base_avg_times: Vec<f64>,
        pub base_estimates: Estimates,
    }
    pub(crate) struct MeasurementData<'a> {
        pub iter_counts: &'a Sample<f64>,
        pub sample_times: &'a Sample<f64>,
        pub avg_times: LabeledSample<'a, f64>,
        pub absolute_estimates: Estimates,
        pub distributions: Distributions,
        pub comparison: Option<ComparisonData>,
        pub throughput: Option<Throughput>,
        pub metrics: Option<BTreeMap<EventName, MetricMeasurementData<'a>>>,
    }
    pub(crate) struct MetricMeasurementData<'a> {
        pub sample: &'a Sample<f64>,
        pub avg: LabeledSample<'a, f64>,
        pub absolute_estimates: Estimates,
        pub distributions: Distributions,
    }
    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    pub enum ValueType {
        Bytes,
        Elements,
        Value,
    }
    #[derive(Clone)]
    pub struct BenchmarkId {
        pub group_id: String,
        pub function_id: Option<String>,
        pub value_str: Option<String>,
        pub throughput: Option<Throughput>,
        full_id: String,
    }
    impl BenchmarkId {
        pub fn new(
            group_id: String,
            function_id: Option<String>,
            value_str: Option<String>,
            throughput: Option<Throughput>,
        ) -> BenchmarkId {
            let full_id = match (&function_id, &value_str) {
                (&Some(ref func), &Some(ref val)) => format!("{}/{}/{}", group_id, func, val),
                (&Some(ref func), &None) => format!("{}/{}", group_id, func),
                (&None, &Some(ref val)) => format!("{}/{}", group_id, val),
                (&None, &None) => group_id.clone(),
            };
            BenchmarkId {
                group_id,
                function_id,
                value_str,
                throughput,
                full_id,
            }
        }
        pub fn id(&self) -> &str {
            unimplemented!()
        }
    }
    impl fmt::Display for BenchmarkId {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            unimplemented!()
        }
    }
    pub struct ReportContext {
        pub output_directory: PathBuf,
        pub plotting: Plotting,
        pub plot_config: PlotConfiguration,
    }
    pub(crate) trait Report {
        fn benchmark_start(&self, id: &BenchmarkId, context: &ReportContext);
        fn warmup(&self, id: &BenchmarkId, context: &ReportContext, warmup_ns: f64);
        fn analysis(&self, id: &BenchmarkId, context: &ReportContext);
        fn measurement_start(
            &self,
            id: &BenchmarkId,
            context: &ReportContext,
            sample_count: u64,
            estimate_ns: f64,
            iter_count: u64,
        );
        fn measurement_complete(
            &self,
            id: &BenchmarkId,
            context: &ReportContext,
            measurements: &MeasurementData,
        );
        fn summarize(&self, context: &ReportContext, all_ids: &[BenchmarkId]);
    }
    pub(crate) struct Reports {
        reports: Vec<Box<Report>>,
    }
    impl Reports {
        pub fn new(reports: Vec<Box<Report>>) -> Reports {
            Reports { reports }
        }
    }
    impl Report for Reports {
        fn benchmark_start(&self, id: &BenchmarkId, context: &ReportContext) {
            unimplemented!()
        }
        fn warmup(&self, id: &BenchmarkId, context: &ReportContext, warmup_ns: f64) {
            unimplemented!()
        }
        fn analysis(&self, id: &BenchmarkId, context: &ReportContext) {
            unimplemented!()
        }
        fn measurement_start(
            &self,
            id: &BenchmarkId,
            context: &ReportContext,
            sample_count: u64,
            estimate_ns: f64,
            iter_count: u64,
        ) {
            unimplemented!()
        }
        fn measurement_complete(
            &self,
            id: &BenchmarkId,
            context: &ReportContext,
            measurements: &MeasurementData,
        ) {
            unimplemented!()
        }
        fn summarize(&self, context: &ReportContext, all_ids: &[BenchmarkId]) {
            unimplemented!()
        }
    }
    pub(crate) struct CliReport {
        pub enable_text_overwrite: bool,
        pub enable_text_coloring: bool,
        pub verbose: bool,
        last_line_len: Cell<usize>,
    }
    impl CliReport {
        pub fn new(
            enable_text_overwrite: bool,
            enable_text_coloring: bool,
            verbose: bool,
        ) -> CliReport {
            CliReport {
                enable_text_overwrite: enable_text_overwrite,
                enable_text_coloring: enable_text_coloring,
                verbose: verbose,
                last_line_len: Cell::new(0),
            }
        }
    }
    impl Report for CliReport {
        fn benchmark_start(&self, id: &BenchmarkId, _: &ReportContext) {
            unimplemented!()
        }
        fn warmup(&self, id: &BenchmarkId, _: &ReportContext, warmup_ns: f64) {
            unimplemented!()
        }
        fn analysis(&self, id: &BenchmarkId, _: &ReportContext) {
            unimplemented!()
        }
        fn measurement_start(
            &self,
            id: &BenchmarkId,
            _: &ReportContext,
            sample_count: u64,
            estimate_ns: f64,
            iter_count: u64,
        ) {
            unimplemented!()
        }
        fn measurement_complete(
            &self,
            id: &BenchmarkId,
            _: &ReportContext,
            meas: &MeasurementData,
        ) {
            unimplemented!()
        }
        fn summarize(&self, _: &ReportContext, _: &[BenchmarkId]) {
            unimplemented!()
        }
    }
}
mod routine {
    use benchmark::BenchmarkConfig;
    use metrics::{measure_fn, EventName};
    use program::Program;
    use report::{BenchmarkId, ReportContext};
    use std::collections::BTreeMap;
    use std::marker::PhantomData;
    use std::time::{Duration, Instant};
    use {Bencher, Criterion, DurationExt};
    pub trait Routine<T> {
        fn start(&mut self, parameter: &T) -> Option<Program>;
        fn bench(
            &mut self,
            m: &mut Option<Program>,
            iters: &[u64],
            parameter: &T,
        ) -> (Vec<f64>, Option<BTreeMap<EventName, Vec<u64>>>);
        fn warm_up(
            &mut self,
            m: &mut Option<Program>,
            how_long: Duration,
            parameter: &T,
        ) -> (u64, u64);
        fn sample(
            &mut self,
            id: &BenchmarkId,
            config: &BenchmarkConfig,
            criterion: &Criterion,
            report_context: &ReportContext,
            parameter: &T,
        ) -> (
            Box<[f64]>,
            Box<[f64]>,
            Option<BTreeMap<EventName, Box<[u64]>>>,
        ) {
            let m_iters_f = vec![];
            let m_elapsed = vec![];
            (
                m_iters_f.into_boxed_slice(),
                m_elapsed.into_boxed_slice(),
                None,
            )
        }
    }
    pub struct Function<F, T>
    where
        F: FnMut(&mut Bencher, &T),
    {
        f: F,
        _phantom: PhantomData<T>,
    }
    impl<F, T> Function<F, T>
    where
        F: FnMut(&mut Bencher, &T),
    {
        pub fn new(f: F) -> Function<F, T> {
            Function {
                f: f,
                _phantom: PhantomData,
            }
        }
    }
    impl<F, T> Routine<T> for Function<F, T>
    where
        F: FnMut(&mut Bencher, &T),
    {
        fn start(&mut self, _: &T) -> Option<Program> {
            unimplemented!()
        }
        fn bench(
            &mut self,
            _: &mut Option<Program>,
            iters: &[u64],
            parameter: &T,
        ) -> (Vec<f64>, Option<BTreeMap<EventName, Vec<u64>>>) {
            unimplemented!()
        }
        fn warm_up(
            &mut self,
            _: &mut Option<Program>,
            how_long: Duration,
            parameter: &T,
        ) -> (u64, u64) {
            unimplemented!()
        }
    }
}
#[cfg(feature = "html_reports")]
mod kde {
    use stats::univariate::kde::kernel::Gaussian;
    use stats::univariate::kde::{Bandwidth, Kde};
    use stats::univariate::Sample;
    pub fn sweep_and_estimate(
        sample: &Sample<f64>,
        npoints: usize,
        range: Option<(f64, f64)>,
        point_to_estimate: f64,
    ) -> (Box<[f64]>, Box<[f64]>, f64) {
        let x_min = sample.min();
        let x_max = sample.max();
        let kde = Kde::new(sample, Gaussian, Bandwidth::Silverman);
        let h = kde.bandwidth();
        let (start, end) = match range {
            Some((start, end)) => (start, end),
            None => (x_min - 3. * h, x_max + 3. * h),
        };
        let xs: Vec<_> = itertools_num::linspace(start, end, npoints).collect();
        let ys = kde.map(&xs);
        let point_estimate = kde.estimate(point_to_estimate);
        (xs.into_boxed_slice(), ys, point_estimate)
    }
}
#[cfg(feature = "html_reports")]
mod plot {
    use criterion_plot::prelude::*;
    use estimate::{Distributions, Estimates};
    use report::BenchmarkId;
    use stats::bivariate::regression::Slope;
    use stats::bivariate::Data;
    use stats::univariate::outliers::tukey::LabeledSample;
    use stats::univariate::Sample;
    use stats::Distribution;
    use std::path::Path;
    use std::process::Child;
    pub mod both {
        use super::{debug_script, escape_underscores, scale_time};
        use super::{DARK_BLUE, DARK_RED, DEFAULT_FONT, KDE_POINTS, LINEWIDTH, SIZE};
        use criterion_plot::prelude::*;
        use estimate::Estimates;
        use kde;
        use report::BenchmarkId;
        use stats::bivariate::Data;
        use stats::univariate::Sample;
        use std::iter;
        use std::path::Path;
        use std::process::Child;
        #[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
        pub(crate) fn regression<P: AsRef<Path>>(
            base_data: Data<f64, f64>,
            base_estimates: &Estimates,
            data: Data<f64, f64>,
            estimates: &Estimates,
            id: &BenchmarkId,
            path: P,
            size: Option<Size>,
            thumbnail_mode: bool,
        ) -> Child {
            unimplemented!()
        }
        pub fn pdfs<P: AsRef<Path>>(
            base_avg_times: &Sample<f64>,
            avg_times: &Sample<f64>,
            id: &BenchmarkId,
            path: P,
            size: Option<Size>,
            thumbnail_mode: bool,
        ) -> Child {
            let base_mean = base_avg_times.mean();
            let new_mean = avg_times.mean();
            let (base_xs, base_ys, base_y_mean) =
                kde::sweep_and_estimate(base_avg_times, KDE_POINTS, None, base_mean);
            let (xs, ys, y_mean) = kde::sweep_and_estimate(avg_times, KDE_POINTS, None, new_mean);
            let base_xs_ = Sample::new(&base_xs);
            let xs_ = Sample::new(&xs);
            let (x_scale, prefix) = scale_time(base_xs_.max().max(xs_.max()));
            let zeros = iter::repeat(0);
            let mut figure = Figure::new();
            if !thumbnail_mode {
                unimplemented!()
            }
            figure
                .set(Font(DEFAULT_FONT))
                .set(size.unwrap_or(SIZE))
                .configure(Axis::BottomX, |a| unimplemented!())
                .configure(Axis::LeftY, |a| a.set(Label("Density (a.u.)")))
                .configure(Axis::RightY, |a| a.hide())
                .configure(Key, |k| unimplemented!())
                .plot(
                    FilledCurve {
                        x: &*base_xs,
                        y1: &*base_ys,
                        y2: zeros.clone(),
                    },
                    |c| c.set(DARK_RED).set(Label("Base PDF")).set(Opacity(0.5)),
                )
                .plot(
                    Lines {
                        x: &[base_mean, base_mean],
                        y: &[0., base_y_mean],
                    },
                    |c| c.set(DARK_RED).set(Label("Base Mean")).set(LINEWIDTH),
                )
                .plot(
                    FilledCurve {
                        x: &*xs,
                        y1: &*ys,
                        y2: zeros,
                    },
                    |c| c.set(DARK_BLUE).set(Label("New PDF")).set(Opacity(0.5)),
                )
                .plot(
                    Lines {
                        x: &[new_mean, new_mean],
                        y: &[0., y_mean],
                    },
                    |c| c.set(DARK_BLUE).set(Label("New Mean")).set(LINEWIDTH),
                );
            debug_script(&path, &figure);
            figure.set(Output(path.as_ref().to_owned())).draw().unwrap()
        }
    }
    fn escape_underscores(string: &str) -> String {
        unimplemented!()
    }
    fn scale_time(ns: f64) -> (f64, &'static str) {
        unimplemented!()
    }
    static DEFAULT_FONT: &'static str = "Helvetica";
    static KDE_POINTS: usize = 500;
    static SIZE: Size = Size(1280, 720);
    const LINEWIDTH: LineWidth = LineWidth(2.);
    const DARK_BLUE: Color = Color::Rgb(31, 120, 180);
    const DARK_RED: Color = Color::Rgb(227, 26, 28);
    fn debug_script<P: AsRef<Path>>(path: P, figure: &Figure) {
        unimplemented!()
    }
    pub fn pdf_small<P: AsRef<Path>>(sample: &Sample<f64>, path: P, size: Option<Size>) -> Child {
        unimplemented!()
    }
    pub fn pdf<P: AsRef<Path>>(
        data: Data<f64, f64>,
        labeled_sample: LabeledSample<f64>,
        id: &BenchmarkId,
        path: P,
        size: Option<Size>,
    ) -> Child {
        unimplemented!()
    }
    pub fn regression<P: AsRef<Path>>(
        data: Data<f64, f64>,
        point: &Slope<f64>,
        (lb, ub): (Slope<f64>, Slope<f64>),
        id: &BenchmarkId,
        path: P,
        size: Option<Size>,
        thumbnail_mode: bool,
    ) -> Child {
        unimplemented!()
    }
    pub(crate) fn abs_distributions<P: AsRef<Path>>(
        distributions: &Distributions,
        estimates: &Estimates,
        id: &BenchmarkId,
        output_directory: P,
    ) -> Vec<Child> {
        unimplemented!()
    }
    pub(crate) fn rel_distributions<P: AsRef<Path>>(
        distributions: &Distributions,
        estimates: &Estimates,
        id: &BenchmarkId,
        output_directory: P,
        nt: f64,
    ) -> Vec<Child> {
        unimplemented!()
    }
    pub fn t_test<P: AsRef<Path>>(
        t: f64,
        distribution: &Distribution<f64>,
        id: &BenchmarkId,
        output_directory: P,
    ) -> Child {
        unimplemented!()
    }
}
#[cfg(feature = "html_reports")]
mod html {
    use criterion_plot::Size;
    use estimate::Statistic;
    use format;
    use fs;
    use handlebars::Handlebars;
    use plot;
    use report::{BenchmarkId, MeasurementData, Report, ReportContext};
    use stats::bivariate::regression::Slope;
    use stats::bivariate::Data;
    use stats::univariate::Sample;
    use std::path::{Path, PathBuf};
    use std::process::Child;
    use Estimate;
    const THUMBNAIL_SIZE: Size = Size(450, 300);
    fn wait_on_gnuplot(children: Vec<Child>) {
        unimplemented!()
    }
    #[derive(Serialize)]
    struct Context {
        title: String,
        confidence: String,
        thumbnail_width: usize,
        thumbnail_height: usize,
        slope: ConfidenceInterval,
        r2: ConfidenceInterval,
        mean: ConfidenceInterval,
        std_dev: ConfidenceInterval,
        median: ConfidenceInterval,
        mad: ConfidenceInterval,
        throughput: Option<ConfidenceInterval>,
        additional_plots: Vec<Plot>,
        comparison: Option<Comparison>,
    }
    #[derive(Serialize)]
    struct IndividualBenchmark {
        name: String,
        path: String,
    }
    #[derive(Serialize)]
    struct ConfidenceInterval {
        lower: String,
        upper: String,
        point: String,
    }
    #[derive(Serialize)]
    struct Plot {
        name: String,
        url: String,
    }
    impl Plot {
        fn new(name: &str, url: &str) -> Plot {
            unimplemented!()
        }
    }
    #[derive(Serialize)]
    struct Comparison {
        p_value: String,
        inequality: String,
        significance_level: String,
        explanation: String,
        change: ConfidenceInterval,
        additional_plots: Vec<Plot>,
    }
    pub struct Html {
        handlebars: Handlebars,
    }
    impl Html {
        pub fn new() -> Html {
            let mut handlebars = Handlebars::new();
            handlebars
                .register_template_string(
                    "report",
                    include_str!("html/benchmark_report.html.handlebars"),
                )
                .expect("Unable to parse benchmark report template.");
            handlebars
                .register_template_string(
                    "summary_report",
                    include_str!("html/summary_report.html.handlebars"),
                )
                .expect("Unable to parse summary report template.");
            Html { handlebars }
        }
    }
    impl Report for Html {
        fn benchmark_start(&self, _: &BenchmarkId, _: &ReportContext) {
            unimplemented!()
        }
        fn warmup(&self, _: &BenchmarkId, _: &ReportContext, _: f64) {
            unimplemented!()
        }
        fn analysis(&self, _: &BenchmarkId, _: &ReportContext) {
            unimplemented!()
        }
        fn measurement_start(&self, _: &BenchmarkId, _: &ReportContext, _: u64, _: f64, _: u64) {
            unimplemented!()
        }
        fn measurement_complete(
            &self,
            id: &BenchmarkId,
            report_context: &ReportContext,
            measurements: &MeasurementData,
        ) {
            if !report_context.plotting.is_enabled() {
                unimplemented!()
            }
            try_else_return!(fs::mkdirp(
                &report_context
                    .output_directory
                    .join(id.to_string())
                    .join("report")
            ));
            let slope_estimate = &measurements.absolute_estimates[&Statistic::Slope];
            fn time_interval(est: &Estimate) -> ConfidenceInterval {
                unimplemented!()
            }
            let data = Data::new(
                measurements.iter_counts.as_slice(),
                measurements.sample_times.as_slice(),
            );
            elapsed! { "Generating plots" , self . generate_plots ( id , report_context , measurements ) }
            let throughput = measurements
                .throughput
                .as_ref()
                .map(|thr| ConfidenceInterval {
                    lower: format::throughput(thr, slope_estimate.confidence_interval.upper_bound),
                    upper: format::throughput(thr, slope_estimate.confidence_interval.lower_bound),
                    point: format::throughput(thr, slope_estimate.point_estimate),
                });
            let context = Context {
                title: id.id().to_owned(),
                confidence: format!("{:.2}", slope_estimate.confidence_interval.confidence_level),
                thumbnail_width: THUMBNAIL_SIZE.0,
                thumbnail_height: THUMBNAIL_SIZE.1,
                slope: time_interval(slope_estimate),
                mean: time_interval(&measurements.absolute_estimates[&Statistic::Mean]),
                median: time_interval(&measurements.absolute_estimates[&Statistic::Median]),
                mad: time_interval(&measurements.absolute_estimates[&Statistic::MedianAbsDev]),
                std_dev: time_interval(&measurements.absolute_estimates[&Statistic::StdDev]),
                throughput: throughput,
                r2: ConfidenceInterval {
                    lower: format!(
                        "{:0.7}",
                        Slope(slope_estimate.confidence_interval.lower_bound).r_squared(data)
                    ),
                    upper: format!(
                        "{:0.7}",
                        Slope(slope_estimate.confidence_interval.upper_bound).r_squared(data)
                    ),
                    point: format!(
                        "{:0.7}",
                        Slope(slope_estimate.point_estimate).r_squared(data)
                    ),
                },
                additional_plots: vec![
                    Plot::new("Slope", "slope.svg"),
                    Plot::new("Mean", "mean.svg"),
                    Plot::new("Std. Dev.", "SD.svg"),
                    Plot::new("Median", "median.svg"),
                    Plot::new("MAD", "MAD.svg"),
                ],
                comparison: self.comparison(measurements),
            };
            let text = self
                .handlebars
                .render("report", &context)
                .expect("Failed to render benchmark report template");
            try_else_return!(fs::save_string(
                &text,
                &report_context
                    .output_directory
                    .join(id.to_string())
                    .join("report")
                    .join("index.html")
            ));
        }
        fn summarize(&self, context: &ReportContext, all_ids: &[BenchmarkId]) {
            unimplemented!()
        }
    }
    impl Html {
        fn comparison(&self, measurements: &MeasurementData) -> Option<Comparison> {
            unimplemented!()
        }
        fn generate_plots(
            &self,
            id: &BenchmarkId,
            context: &ReportContext,
            measurements: &MeasurementData,
        ) {
            let data = Data::new(
                measurements.iter_counts.as_slice(),
                measurements.sample_times.as_slice(),
            );
            let slope_estimate = &measurements.absolute_estimates[&Statistic::Slope];
            let point = Slope::fit(data);
            let slope_dist = &measurements.distributions[&Statistic::Slope];
            let (lb, ub) =
                slope_dist.confidence_interval(slope_estimate.confidence_interval.confidence_level);
            let (lb_, ub_) = (Slope(lb), Slope(ub));
            let report_dir = context.output_directory.join(id.to_string()).join("report");
            let mut gnuplots = vec![];
            gnuplots.push(plot::pdf(
                data,
                measurements.avg_times,
                id,
                report_dir.join("pdf.svg"),
                None,
            ));
            gnuplots.extend(plot::abs_distributions(
                &measurements.distributions,
                &measurements.absolute_estimates,
                id,
                &context.output_directory,
            ));
            gnuplots.push(plot::regression(
                data,
                &point,
                (lb_, ub_),
                id,
                report_dir.join("regression.svg"),
                None,
                false,
            ));
            gnuplots.push(plot::pdf_small(
                &*measurements.avg_times,
                report_dir.join("pdf_small.svg"),
                Some(THUMBNAIL_SIZE),
            ));
            gnuplots.push(plot::regression(
                data,
                &point,
                (lb_, ub_),
                id,
                report_dir.join("regression_small.svg"),
                Some(THUMBNAIL_SIZE),
                true,
            ));
            if let Some(ref comp) = measurements.comparison {
                try_else_return!(fs::mkdirp(&report_dir.join("change")));
                let base_data = Data::new(&comp.base_iter_counts, &comp.base_sample_times);
                let both_dir = report_dir.join("both");
                try_else_return!(fs::mkdirp(&both_dir));
                gnuplots.push(plot::both::regression(
                    base_data,
                    &comp.base_estimates,
                    data,
                    &measurements.absolute_estimates,
                    id,
                    both_dir.join("regression.svg"),
                    None,
                    false,
                ));
                gnuplots.push(plot::both::pdfs(
                    Sample::new(&comp.base_avg_times),
                    &*measurements.avg_times,
                    id,
                    both_dir.join("pdf.svg"),
                    None,
                    false,
                ));
                gnuplots.push(plot::t_test(
                    comp.t_value,
                    &comp.t_distribution,
                    id,
                    &context.output_directory,
                ));
                gnuplots.extend(plot::rel_distributions(
                    &comp.relative_distributions,
                    &comp.relative_estimates,
                    id,
                    &context.output_directory,
                    comp.noise_threshold,
                ));
                gnuplots.push(plot::both::regression(
                    base_data,
                    &comp.base_estimates,
                    data,
                    &measurements.absolute_estimates,
                    id,
                    report_dir.join("relative_regression_small.svg"),
                    Some(THUMBNAIL_SIZE),
                    true,
                ));
                gnuplots.push(plot::both::pdfs(
                    Sample::new(&comp.base_avg_times),
                    &*measurements.avg_times,
                    id,
                    report_dir.join("relative_pdf_small.svg"),
                    Some(THUMBNAIL_SIZE),
                    true,
                ));
            }
            wait_on_gnuplot(gnuplots);
        }
    }
    enum ComparisonResult {
        Improved,
        Regressed,
        NonSignificant,
    }
}
use benchmark::BenchmarkConfig;
use benchmark::NamedRoutine;
pub use benchmark::{Benchmark, BenchmarkDefinition, ParameterizedBenchmark};
#[cfg(feature = "html_reports")]
use html::Html;
use plotting::Plotting;
use report::{CliReport, Report, Reports};
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{fmt, mem};
pub struct Fun<I: fmt::Debug> {
    f: NamedRoutine<I>,
}
#[derive(Clone, Copy)]
pub struct Bencher {
    iters: u64,
    elapsed: Duration,
}
impl Bencher {
    pub fn iter<O, R>(&mut self, mut routine: R)
    where
        R: FnMut() -> O,
    {
        unimplemented!()
    }
}
pub struct Criterion {
    config: BenchmarkConfig,
    plotting: Plotting,
    filter: Option<String>,
    report: Box<Report>,
    output_directory: PathBuf,
}
impl Default for Criterion {
    fn default() -> Criterion {
        #[allow(unused_mut, unused_assignments)]
        let mut plotting = Plotting::NotAvailable;
        let mut reports: Vec<Box<Report>> = vec![];
        reports.push(Box::new(CliReport::new(false, false, false)));
        #[cfg(feature = "html_reports")]
        {
            plotting = if criterion_plot::version().is_ok() {
                unimplemented!()
            } else {
                println!("Gnuplot not found, disabling plotting");
                Plotting::NotAvailable
            };
            reports.push(Box::new(Html::new()));
        }
        let output_directory: PathBuf = [
            env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| String::from("target")),
            String::from("criterion"),
        ]
        .iter()
        .collect();
        Criterion {
            config: BenchmarkConfig {
                confidence_level: 0.95,
                measurement_time: Duration::new(5, 0),
                noise_threshold: 0.01,
                nresamples: 100_000,
                sample_size: 100,
                significance_level: 0.05,
                warm_up_time: Duration::new(3, 0),
            },
            plotting: plotting,
            filter: None,
            report: Box::new(Reports::new(reports)),
            output_directory,
        }
    }
}
mod plotting {
    #[derive(Debug, Clone, Copy)]
    pub enum Plotting {
        Disabled,
        Enabled,
        NotAvailable,
    }
    impl Plotting {
        pub fn is_enabled(&self) -> bool {
            unimplemented!()
        }
    }
}
trait DurationExt {
    fn to_nanos(&self) -> u64;
}
impl DurationExt for Duration {
    fn to_nanos(&self) -> u64 {
        unimplemented!()
    }
}
#[derive(Clone, Copy, PartialEq, Deserialize, Serialize, Debug)]
struct ConfidenceInterval {
    confidence_level: f64,
    lower_bound: f64,
    upper_bound: f64,
}
#[derive(Clone, Copy, PartialEq, Deserialize, Serialize, Debug)]
struct Estimate {
    confidence_interval: ConfidenceInterval,
    point_estimate: f64,
    standard_error: f64,
}
#[derive(Debug, Clone, Serialize)]
pub enum Throughput {
    Bytes(u32),
    Elements(u32),
}
#[derive(Debug, Clone, Copy)]
pub enum AxisScale {
    Linear,
    Logarithmic,
}
#[derive(Debug, Clone)]
pub struct PlotConfiguration {
    summary_scale: AxisScale,
}
impl Default for PlotConfiguration {
    fn default() -> PlotConfiguration {
        PlotConfiguration {
            summary_scale: AxisScale::Linear,
        }
    }
}
