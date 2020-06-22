extern crate clap;
#[cfg(feature = "html_reports")]
extern crate criterion_plot;
extern crate criterion_stats as stats;
extern crate failure;
#[cfg(feature = "html_reports")]
extern crate handlebars;
extern crate itertools;
extern crate itertools_num;
extern crate serde;
extern crate serde_json;
extern crate simplelog;
#[macro_use]
extern crate log;
#[macro_use]
extern crate failure_derive;
#[macro_use]
extern crate serde_derive;
#[macro_use]
mod macros_private {
    macro_rules! log_if_err {
        ( $ x : expr ) => {
            let closure = || {
                try_else_return!($x);
            };
            closure();
        };
    }
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
    use benchmark::BenchmarkConfig;
    use estimate::{Distributions, Estimates, Statistic};
    use report::{BenchmarkId, ReportContext};
    use routine::Routine;
    use stats::bivariate::regression::Slope;
    use stats::bivariate::Data;
    use stats::univariate::outliers::tukey::{self, LabeledSample};
    use stats::univariate::Sample;
    use stats::{Distribution, Tails};
    use std::collections::BTreeMap;
    use std::path::Path;
    use {format, fs};
    use {ConfidenceInterval, Criterion, Estimate, Throughput};
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
    pub(crate) fn common<T>(
        id: &BenchmarkId,
        routine: &mut Routine<T>,
        config: &BenchmarkConfig,
        criterion: &Criterion,
        report_context: &ReportContext,
        parameter: &T,
        throughput: Option<Throughput>,
    ) {
        unimplemented!()
    }
    pub(super) fn regression(
        data: Data<f64, f64>,
        config: &BenchmarkConfig,
    ) -> (Distribution<f64>, Estimate) {
        unimplemented!()
    }
}
mod benchmark {
    use analysis;
    use program::CommandFactory;
    use report::{BenchmarkId, ReportContext};
    use routine::{Function, Routine};
    use std::cell::RefCell;
    use std::fmt::Debug;
    use std::process::Command;
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
    impl PartialBenchmarkConfig {
        fn to_complete(&self, defaults: &BenchmarkConfig) -> BenchmarkConfig {
            BenchmarkConfig {
                confidence_level: self.confidence_level.unwrap_or(defaults.confidence_level),
                measurement_time: self.measurement_time.unwrap_or(defaults.measurement_time),
                noise_threshold: self.noise_threshold.unwrap_or(defaults.noise_threshold),
                nresamples: self.nresamples.unwrap_or(defaults.nresamples),
                sample_size: self.sample_size.unwrap_or(defaults.sample_size),
                significance_level: self
                    .significance_level
                    .unwrap_or(defaults.significance_level),
                warm_up_time: self.warm_up_time.unwrap_or(defaults.warm_up_time),
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
    macro_rules ! benchmark_config { ( $ type : tt ) => { # [ doc = " Changes the size of the sample for this benchmark" ] # [ doc = "" ] # [ doc = " A bigger sample should yield more accurate results, if paired with a \"sufficiently\" large" ] # [ doc = " measurement time, on the other hand, it also increases the analysis time" ] # [ doc = "" ] # [ doc = " # Panics" ] # [ doc = "" ] # [ doc = " Panics if set to zero" ] pub fn sample_size ( mut self , n : usize ) -> Self { assert ! ( n > 0 ) ; self . config . sample_size = Some ( n ) ; self } # [ doc = " Changes the warm up time for this benchmark" ] # [ doc = "" ] # [ doc = " # Panics" ] # [ doc = "" ] # [ doc = " Panics if the input duration is zero" ] pub fn warm_up_time ( mut self , dur : Duration ) -> Self { assert ! ( dur . to_nanos ( ) > 0 ) ; self . config . warm_up_time = Some ( dur ) ; self } # [ doc = " Changes the measurement time for this benchmark" ] # [ doc = "" ] # [ doc = " With a longer time, the measurement will become more resilient to transitory peak loads" ] # [ doc = " caused by external programs" ] # [ doc = "" ] # [ doc = " **Note**: If the measurement time is too \"low\", Criterion will automatically increase it" ] # [ doc = "" ] # [ doc = " # Panics" ] # [ doc = "" ] # [ doc = " Panics if the input duration in zero" ] pub fn measurement_time ( mut self , dur : Duration ) -> Self { assert ! ( dur . to_nanos ( ) > 0 ) ; self . config . measurement_time = Some ( dur ) ; self } # [ doc = " Changes the number of resamples for this benchmark" ] # [ doc = "" ] # [ doc = " Number of resamples to use for the" ] # [ doc = " [bootstrap](http://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Case_resampling)" ] # [ doc = "" ] # [ doc = " A larger number of resamples reduces the random sampling errors, which are inherent to the" ] # [ doc = " bootstrap method, but also increases the analysis time" ] # [ doc = "" ] # [ doc = " # Panics" ] # [ doc = "" ] # [ doc = " Panics if the number of resamples is set to zero" ] pub fn nresamples ( mut self , n : usize ) -> Self { assert ! ( n > 0 ) ; self . config . nresamples = Some ( n ) ; self } # [ doc = " Changes the noise threshold for this benchmark" ] # [ doc = "" ] # [ doc = " This threshold is used to decide if an increase of `X%` in the execution time is considered" ] # [ doc = " significant or should be flagged as noise" ] # [ doc = "" ] # [ doc = " *Note:* A value of `0.02` is equivalent to `2%`" ] # [ doc = "" ] # [ doc = " # Panics" ] # [ doc = "" ] # [ doc = " Panics is the threshold is set to a negative value" ] pub fn noise_threshold ( mut self , threshold : f64 ) -> Self { assert ! ( threshold >= 0.0 ) ; self . config . noise_threshold = Some ( threshold ) ; self } # [ doc = " Changes the confidence level for this benchmark" ] # [ doc = "" ] # [ doc = " The confidence level is used to calculate the" ] # [ doc = " [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval) of the estimated" ] # [ doc = " statistics" ] # [ doc = "" ] # [ doc = " # Panics" ] # [ doc = "" ] # [ doc = " Panics if the confidence level is set to a value outside the `(0, 1)` range" ] pub fn confidence_level ( mut self , cl : f64 ) -> Self { assert ! ( cl > 0.0 && cl < 1.0 ) ; self . config . confidence_level = Some ( cl ) ; self } # [ doc = " Changes the [significance level](https://en.wikipedia.org/wiki/Statistical_significance)" ] # [ doc = " for this benchmark" ] # [ doc = "" ] # [ doc = " The significance level is used for" ] # [ doc = " [hypothesis testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)" ] # [ doc = "" ] # [ doc = " # Panics" ] # [ doc = "" ] # [ doc = " Panics if the significance level is set to a value outside the `(0, 1)` range" ] pub fn significance_level ( mut self , sl : f64 ) -> Self { assert ! ( sl > 0.0 && sl < 1.0 ) ; self . config . significance_level = Some ( sl ) ; self } # [ doc = " Changes the plot configuration for this benchmark." ] pub fn plot_config ( mut self , new_config : PlotConfiguration ) -> Self { self . config . plot_config = new_config ; self } } }
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
    impl<T> ParameterizedBenchmark<T> where T: Debug + 'static {}
    impl<T> BenchmarkDefinition for ParameterizedBenchmark<T>
    where
        T: Debug + 'static,
    {
        fn run(self, group_id: &str, c: &Criterion) {
            let report_context = ReportContext {
                output_directory: c.output_directory.clone(),
                plotting: c.plotting,
                plot_config: self.config.plot_config.clone(),
            };
            let config = self.config.to_complete(&c.config);
            let num_parameters = self.values.len();
            let num_routines = self.routines.len();
            let mut all_ids = vec![];
            let mut any_matched = false;
            for routine in self.routines {
                for value in &self.values {
                    let function_id = if num_routines == 1 && group_id == routine.id {
                        None
                    } else {
                        Some(routine.id.clone())
                    };
                    let value_str = if num_parameters == 1 {
                        None
                    } else {
                        Some(format!("{:?}", value))
                    };
                    let throughput = self.throughput.as_ref().map(|func| func(value));
                    let id = BenchmarkId::new(
                        group_id.to_owned(),
                        function_id,
                        value_str,
                        throughput.clone(),
                    );
                    if c.filter_matches(id.id()) {
                        any_matched = true;
                        analysis::common(
                            &id,
                            &mut *routine.f.borrow_mut(),
                            &config,
                            c,
                            &report_context,
                            value,
                            throughput,
                        );
                    }
                    all_ids.push(id);
                }
            }
            if all_ids.len() > 1 && any_matched {
                c.report.summarize(&report_context, &all_ids);
            }
            if any_matched {
                println!();
            }
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
        error!("error: {}", e.cause());
        for cause in e.causes() {
            error!("caused by: {}", cause);
        }
        debug!("backtrace: {}", e.backtrace());
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
    impl fmt::Display for Statistic {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match *self {
                Statistic::Mean => f.pad("mean"),
                Statistic::Median => f.pad("median"),
                Statistic::MedianAbsDev => f.pad("MAD"),
                Statistic::Slope => f.pad("slope"),
                Statistic::StdDev => f.pad("SD"),
            }
        }
    }
    pub(crate) type Estimates = BTreeMap<Statistic, Estimate>;
    pub(crate) type Distributions = BTreeMap<Statistic, Distribution<f64>>;
}
mod format {
    use Throughput;
    pub fn change(pct: f64, signed: bool) -> String {
        if signed {
            format!("{:>+6}%", signed_short(pct * 1e2))
        } else {
            format!("{:>6}%", short(pct * 1e2))
        }
    }
    fn short(n: f64) -> String {
        if n < 10.0 {
            format!("{:.4}", n)
        } else if n < 100.0 {
            format!("{:.3}", n)
        } else if n < 1000.0 {
            format!("{:.2}", n)
        } else if n < 10000.0 {
            format!("{:.1}", n)
        } else {
            format!("{}", n)
        }
    }
    fn signed_short(n: f64) -> String {
        let n_abs = n.abs();
        if n_abs < 10.0 {
            format!("{:+.4}", n)
        } else if n_abs < 100.0 {
            format!("{:+.3}", n)
        } else if n_abs < 1000.0 {
            format!("{:+.2}", n)
        } else {
            format!("{:+}", n)
        }
    }
    pub fn time(ns: f64) -> String {
        if ns < 1.0 {
            format!("{:>6} ps", short(ns * 1e3))
        } else if ns < 10f64.powi(3) {
            format!("{:>6} ns", short(ns))
        } else if ns < 10f64.powi(6) {
            format!("{:>6} us", short(ns / 1e3))
        } else if ns < 10f64.powi(9) {
            format!("{:>6} ms", short(ns / 1e6))
        } else {
            format!("{:>6} s", short(ns / 1e9))
        }
    }
    pub fn throughput(throughput: &Throughput, ns: f64) -> String {
        match *throughput {
            Throughput::Bytes(bytes) => bytes_per_second(f64::from(bytes) * (1e9 / ns)),
            Throughput::Elements(elems) => elements_per_second(f64::from(elems) * (1e9 / ns)),
        }
    }
    pub fn bytes_per_second(bytes_per_second: f64) -> String {
        if bytes_per_second < 1024.0 {
            format!("{:>6}   B/s", short(bytes_per_second))
        } else if bytes_per_second < 1024.0 * 1024.0 {
            format!("{:>6} KiB/s", short(bytes_per_second / 1024.0))
        } else if bytes_per_second < 1024.0 * 1024.0 * 1024.0 {
            format!("{:>6} MiB/s", short(bytes_per_second / (1024.0 * 1024.0)))
        } else {
            format!(
                "{:>6} GiB/s",
                short(bytes_per_second / (1024.0 * 1024.0 * 1024.0))
            )
        }
    }
    pub fn elements_per_second(elements_per_second: f64) -> String {
        if elements_per_second < 1000.0 {
            format!("{:>6}  elem/s", short(elements_per_second))
        } else if elements_per_second < 1000.0 * 1000.0 {
            format!("{:>6} Kelem/s", short(elements_per_second / 1000.0))
        } else if elements_per_second < 1000.0 * 1000.0 * 1000.0 {
            format!(
                "{:>6} Melem/s",
                short(elements_per_second / (1000.0 * 1000.0))
            )
        } else {
            format!(
                "{:>6} Gelem/s",
                short(elements_per_second / (1000.0 * 1000.0 * 1000.0))
            )
        }
    }
    pub fn iter_count(iterations: u64) -> String {
        if iterations < 10_000 {
            format!("{} iterations", iterations)
        } else if iterations < 1_000_000 {
            format!("{:.0}k iterations", (iterations as f64) / 1000.0)
        } else if iterations < 10_000_000 {
            format!("{:.1}M iterations", (iterations as f64) / (1000.0 * 1000.0))
        } else if iterations < 1_000_000_000 {
            format!("{:.0}M iterations", (iterations as f64) / (1000.0 * 1000.0))
        } else if iterations < 10_000_000_000 {
            format!(
                "{:.1}B iterations",
                (iterations as f64) / (1000.0 * 1000.0 * 1000.0)
            )
        } else {
            format!(
                "{:.0}B iterations",
                (iterations as f64) / (1000.0 * 1000.0 * 1000.0)
            )
        }
    }
}
mod fs {
    use error::{AccessError, Result};
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use std::fs::{self, File};
    use std::io::Read;
    use std::path::Path;
    pub fn load<A, P: ?Sized>(path: &P) -> Result<A>
    where
        A: DeserializeOwned,
        P: AsRef<Path>,
    {
        let mut f = File::open(path).map_err(|inner| AccessError {
            inner,
            path: path.as_ref().to_owned(),
        })?;
        let mut string = String::new();
        let _ = f.read_to_string(&mut string);
        let result: A = serde_json::from_str(string.as_str())?;
        Ok(result)
    }
    pub fn mkdirp<P>(path: &P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        fs::create_dir_all(path.as_ref())?;
        Ok(())
    }
    pub fn save_string<P>(data: &str, path: &P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        use std::io::Write;
        File::create(path)
            .and_then(|mut f| f.write_all(data.as_bytes()))
            .map_err(|inner| AccessError {
                inner,
                path: path.as_ref().to_owned(),
            })?;
        Ok(())
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
            (function(), None)
        }
    }
}
mod program {
    use metrics::EventName;
    use routine::Routine;
    use std::collections::BTreeMap;
    use std::fmt;
    use std::io::BufReader;
    use std::marker::PhantomData;
    use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::{Duration, Instant};
    use DurationExt;
    pub struct Program {
        buffer: String,
        stdin: ChildStdin,
        _child: Child,
        stderr: ChildStderr,
        stdout: BufReader<ChildStdout>,
    }
    impl Program {
        pub fn spawn(cmd: &mut Command) -> Program {
            cmd.stderr(Stdio::piped());
            cmd.stdin(Stdio::piped());
            cmd.stdout(Stdio::piped());
            let mut child = match cmd.spawn() {
                Err(e) => panic!("`{:?}`: {}", cmd, e),
                Ok(child) => child,
            };
            Program {
                buffer: String::new(),
                stderr: child.stderr.take().unwrap(),
                stdin: child.stdin.take().unwrap(),
                stdout: BufReader::new(child.stdout.take().unwrap()),
                _child: child,
            }
        }
        pub fn send<T>(&mut self, line: T) -> &mut Program
        where
            T: fmt::Display,
        {
            use std::io::Write;
            match writeln!(&mut self.stdin, "{}", line) {
                Err(e) => panic!("`write into child stdin`: {}", e),
                Ok(_) => self,
            }
        }
        pub fn recv(&mut self) -> &str {
            use std::io::{BufRead, Read};
            self.buffer.clear();
            match self.stdout.read_line(&mut self.buffer) {
                Err(e) => {
                    self.buffer.clear();
                    match self.stderr.read_to_string(&mut self.buffer) {
                        Err(e) => {
                            panic!("`read from child stderr`: {}", e);
                        }
                        Ok(_) => {
                            println!("stderr:\n{}", self.buffer);
                        }
                    }
                    panic!("`read from child stdout`: {}", e);
                }
                Ok(_) => &self.buffer,
            }
        }
        fn bench(&mut self, iters: &[u64]) -> Vec<f64> {
            let mut n = 0;
            for iters in iters {
                self.send(iters);
                n += 1;
            }
            (0..n)
                .map(|_| {
                    let msg = self.recv();
                    let msg = msg.trim();
                    let elapsed: u64 = msg.parse().expect("Couldn't parse program output");
                    elapsed as f64
                })
                .collect()
        }
        fn warm_up(&mut self, how_long_ns: Duration) -> (u64, u64) {
            let mut iters = 1;
            let mut total_iters = 0;
            let start = Instant::now();
            loop {
                self.send(iters).recv();
                total_iters += iters;
                let elapsed = start.elapsed();
                if elapsed > how_long_ns {
                    return (elapsed.to_nanos(), total_iters);
                }
                iters *= 2;
            }
        }
    }
    impl Routine<()> for Command {
        fn start(&mut self, _: &()) -> Option<Program> {
            Some(Program::spawn(self))
        }
        fn bench(
            &mut self,
            program: &mut Option<Program>,
            iters: &[u64],
            _: &(),
        ) -> (Vec<f64>, Option<BTreeMap<EventName, Vec<u64>>>) {
            let program = program.as_mut().unwrap();
            (program.bench(iters), None)
        }
        fn warm_up(
            &mut self,
            program: &mut Option<Program>,
            how_long_ns: Duration,
            _: &(),
        ) -> (u64, u64) {
            let program = program.as_mut().unwrap();
            program.warm_up(how_long_ns)
        }
    }
    pub struct CommandFactory<F, T>
    where
        F: FnMut(&T) -> Command + 'static,
    {
        f: F,
        _phantom: PhantomData<T>,
    }
    impl<F, T> CommandFactory<F, T> where F: FnMut(&T) -> Command + 'static {}
    impl<F, T> Routine<T> for CommandFactory<F, T>
    where
        F: FnMut(&T) -> Command + 'static,
    {
        fn start(&mut self, parameter: &T) -> Option<Program> {
            let mut command = (self.f)(parameter);
            Some(Program::spawn(&mut command))
        }
        fn bench(
            &mut self,
            program: &mut Option<Program>,
            iters: &[u64],
            _: &T,
        ) -> (Vec<f64>, Option<BTreeMap<EventName, Vec<u64>>>) {
            let program = program.as_mut().unwrap();
            (program.bench(iters), None)
        }
        fn warm_up(
            &mut self,
            program: &mut Option<Program>,
            how_long_ns: Duration,
            _: &T,
        ) -> (u64, u64) {
            let program = program.as_mut().unwrap();
            program.warm_up(how_long_ns)
        }
    }
}
mod report {
    use estimate::{Distributions, Estimates, Statistic};
    use format;
    use metrics::EventName;
    use stats::bivariate::regression::Slope;
    use stats::bivariate::Data;
    use stats::univariate::outliers::tukey::LabeledSample;
    use stats::univariate::Sample;
    use stats::Distribution;
    use std::cell::Cell;
    use std::collections::BTreeMap;
    use std::fmt;
    use std::io::stdout;
    use std::io::Write;
    use std::path::PathBuf;
    use Estimate;
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
            &self.full_id
        }
        pub fn as_number(&self) -> Option<f64> {
            match self.throughput {
                Some(Throughput::Bytes(n)) | Some(Throughput::Elements(n)) => Some(f64::from(n)),
                None => self
                    .value_str
                    .as_ref()
                    .and_then(|string| string.parse::<f64>().ok()),
            }
        }
        pub fn value_type(&self) -> Option<ValueType> {
            match self.throughput {
                Some(Throughput::Bytes(_)) => Some(ValueType::Bytes),
                Some(Throughput::Elements(_)) => Some(ValueType::Elements),
                None => self
                    .value_str
                    .as_ref()
                    .and_then(|string| string.parse::<f64>().ok())
                    .map(|_| ValueType::Value),
            }
        }
    }
    impl fmt::Display for BenchmarkId {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str(self.id())
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
            for report in &self.reports {
                report.benchmark_start(id, context);
            }
        }
        fn warmup(&self, id: &BenchmarkId, context: &ReportContext, warmup_ns: f64) {
            for report in &self.reports {
                report.warmup(id, context, warmup_ns);
            }
        }
        fn analysis(&self, id: &BenchmarkId, context: &ReportContext) {
            for report in &self.reports {
                report.analysis(id, context);
            }
        }
        fn measurement_start(
            &self,
            id: &BenchmarkId,
            context: &ReportContext,
            sample_count: u64,
            estimate_ns: f64,
            iter_count: u64,
        ) {
            for report in &self.reports {
                report.measurement_start(id, context, sample_count, estimate_ns, iter_count);
            }
        }
        fn measurement_complete(
            &self,
            id: &BenchmarkId,
            context: &ReportContext,
            measurements: &MeasurementData,
        ) {
            for report in &self.reports {
                report.measurement_complete(id, context, measurements);
            }
        }
        fn summarize(&self, context: &ReportContext, all_ids: &[BenchmarkId]) {
            for report in &self.reports {
                report.summarize(context, all_ids);
            }
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
        fn text_overwrite(&self) {
            if self.enable_text_overwrite {
                print!("\r");
                for _ in 0..self.last_line_len.get() {
                    print!(" ");
                }
                print!("\r");
            }
        }
        #[cfg_attr(feature = "cargo-clippy", allow(needless_pass_by_value))]
        fn print_overwritable(&self, s: String) {
            if self.enable_text_overwrite {
                self.last_line_len.set(s.len());
                print!("{}", s);
                stdout().flush().unwrap();
            } else {
                println!("{}", s);
            }
        }
        fn green(&self, s: String) -> String {
            if self.enable_text_coloring {
                format!("\x1B[32m{}\x1B[39m", s)
            } else {
                s
            }
        }
        fn yellow(&self, s: String) -> String {
            if self.enable_text_coloring {
                format!("\x1B[33m{}\x1B[39m", s)
            } else {
                s
            }
        }
        fn red(&self, s: String) -> String {
            if self.enable_text_coloring {
                format!("\x1B[31m{}\x1B[39m", s)
            } else {
                s
            }
        }
        fn bold(&self, s: String) -> String {
            if self.enable_text_coloring {
                format!("\x1B[1m{}\x1B[22m", s)
            } else {
                s
            }
        }
        fn faint(&self, s: String) -> String {
            if self.enable_text_coloring {
                format!("\x1B[2m{}\x1B[22m", s)
            } else {
                s
            }
        }
        pub fn outliers(&self, sample: &LabeledSample<f64>) {
            let (los, lom, _, him, his) = sample.count();
            let noutliers = los + lom + him + his;
            let sample_size = sample.as_slice().len();
            if noutliers == 0 {
                return;
            }
            let percent = |n: usize| 100. * n as f64 / sample_size as f64;
            println!(
                "{}",
                self.yellow(format!(
                    "Found {} outliers among {} measurements ({:.2}%)",
                    noutliers,
                    sample_size,
                    percent(noutliers)
                ))
            );
            let print = |n, label| {
                if n != 0 {
                    println!("  {} ({:.2}%) {}", n, percent(n), label);
                }
            };
            print(los, "low severe");
            print(lom, "low mild");
            print(him, "high mild");
            print(his, "high severe");
        }
    }
    impl Report for CliReport {
        fn benchmark_start(&self, id: &BenchmarkId, _: &ReportContext) {
            self.print_overwritable(format!("Benchmarking {}", id));
        }
        fn warmup(&self, id: &BenchmarkId, _: &ReportContext, warmup_ns: f64) {
            self.text_overwrite();
            self.print_overwritable(format!(
                "Benchmarking {}: Warming up for {}",
                id,
                format::time(warmup_ns)
            ));
        }
        fn analysis(&self, id: &BenchmarkId, _: &ReportContext) {
            self.text_overwrite();
            self.print_overwritable(format!("Benchmarking {}: Analyzing", id));
        }
        fn measurement_start(
            &self,
            id: &BenchmarkId,
            _: &ReportContext,
            sample_count: u64,
            estimate_ns: f64,
            iter_count: u64,
        ) {
            self.text_overwrite();
            let iter_string = if self.verbose {
                format!("{} iterations", iter_count)
            } else {
                format::iter_count(iter_count)
            };
            self.print_overwritable(format!(
                "Benchmarking {}: Collecting {} samples in estimated {} ({})",
                id,
                sample_count,
                format::time(estimate_ns),
                iter_string
            ));
        }
        fn measurement_complete(
            &self,
            id: &BenchmarkId,
            _: &ReportContext,
            meas: &MeasurementData,
        ) {
            self.text_overwrite();
            let slope_estimate = meas.absolute_estimates[&Statistic::Slope];
            {
                let mut id = id.id().to_owned();
                if id.len() > 23 {
                    if id.len() > 80 {
                        id.truncate(77);
                        id.push_str("...");
                    }
                    println!("{}", self.green(id.clone()));
                    id.clear();
                }
                let id_len = id.len();
                println!(
                    "{}{}time:   [{} {} {}]",
                    self.green(id),
                    " ".repeat(24 - id_len),
                    self.faint(format::time(slope_estimate.confidence_interval.lower_bound)),
                    self.bold(format::time(slope_estimate.point_estimate)),
                    self.faint(format::time(slope_estimate.confidence_interval.upper_bound))
                );
            }
            if let Some(ref throughput) = meas.throughput {
                println!(
                    "{}thrpt:  [{} {} {}]",
                    " ".repeat(24),
                    self.faint(format::throughput(
                        throughput,
                        slope_estimate.confidence_interval.upper_bound
                    )),
                    self.bold(format::throughput(
                        throughput,
                        slope_estimate.point_estimate
                    )),
                    self.faint(format::throughput(
                        throughput,
                        slope_estimate.confidence_interval.lower_bound
                    )),
                )
            }
            if let Some(ref comp) = meas.comparison {
                let different_mean = comp.p_value < comp.significance_threshold;
                let mean_est = comp.relative_estimates[&Statistic::Mean];
                let point_estimate = mean_est.point_estimate;
                let mut point_estimate_str = format::change(point_estimate, true);
                let explanation_str: String;
                if !different_mean {
                    explanation_str = "No change in performance detected.".to_owned();
                } else {
                    let comparison = compare_to_threshold(&mean_est, comp.noise_threshold);
                    match comparison {
                        ComparisonResult::Improved => {
                            point_estimate_str = self.green(self.bold(point_estimate_str));
                            explanation_str =
                                format!("Performance has {}.", self.green("improved".to_owned()));
                        }
                        ComparisonResult::Regressed => {
                            point_estimate_str = self.red(self.bold(point_estimate_str));
                            explanation_str =
                                format!("Performance has {}.", self.red("regressed".to_owned()));
                        }
                        ComparisonResult::NonSignificant => {
                            explanation_str = "Change within noise threshold.".to_owned();
                        }
                    }
                }
                println!(
                    "{}change: [{} {} {}] (p = {:.2} {} {:.2})",
                    " ".repeat(24),
                    self.faint(format::change(
                        mean_est.confidence_interval.lower_bound,
                        true
                    )),
                    point_estimate_str,
                    self.faint(format::change(
                        mean_est.confidence_interval.upper_bound,
                        true
                    )),
                    comp.p_value,
                    if different_mean { "<" } else { ">" },
                    comp.significance_threshold
                );
                println!("{}{}", " ".repeat(24), explanation_str);
            }
            self.outliers(&meas.avg_times);
            if self.verbose {
                let data = Data::new(meas.iter_counts.as_slice(), meas.sample_times.as_slice());
                let slope_estimate = &meas.absolute_estimates[&Statistic::Slope];
                fn format_short_estimate(estimate: &Estimate) -> String {
                    format!(
                        "[{} {}]",
                        format::time(estimate.confidence_interval.lower_bound),
                        format::time(estimate.confidence_interval.upper_bound)
                    )
                }
                println!(
                    "{:<7}{} {:<15}[{:0.7} {:0.7}]",
                    "slope",
                    format_short_estimate(slope_estimate),
                    "R^2",
                    Slope(slope_estimate.confidence_interval.lower_bound).r_squared(data),
                    Slope(slope_estimate.confidence_interval.upper_bound).r_squared(data),
                );
                println!(
                    "{:<7}{} {:<15}{}",
                    "mean",
                    format_short_estimate(&meas.absolute_estimates[&Statistic::Mean]),
                    "std. dev.",
                    format_short_estimate(&meas.absolute_estimates[&Statistic::StdDev]),
                );
                println!(
                    "{:<7}{} {:<15}{}",
                    "median",
                    format_short_estimate(&meas.absolute_estimates[&Statistic::Median]),
                    "med. abs. dev.",
                    format_short_estimate(&meas.absolute_estimates[&Statistic::MedianAbsDev]),
                );
            }
        }
        fn summarize(&self, _: &ReportContext, _: &[BenchmarkId]) {}
    }
    enum ComparisonResult {
        Improved,
        Regressed,
        NonSignificant,
    }
    fn compare_to_threshold(estimate: &Estimate, noise: f64) -> ComparisonResult {
        let ci = estimate.confidence_interval;
        let lb = ci.lower_bound;
        let ub = ci.upper_bound;
        if lb < -noise && ub < -noise {
            ComparisonResult::Improved
        } else if lb > noise && ub > noise {
            ComparisonResult::Regressed
        } else {
            ComparisonResult::NonSignificant
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
            None
        }
        fn bench(
            &mut self,
            _: &mut Option<Program>,
            iters: &[u64],
            parameter: &T,
        ) -> (Vec<f64>, Option<BTreeMap<EventName, Vec<u64>>>) {
            let mut b = Bencher {
                iters: 0,
                elapsed: Duration::from_secs(0),
            };
            let mut times = Vec::new();
            let mut metrics = BTreeMap::new();
            for &i in iters {
                b.iters = i;
                let (time, run_metrics) = measure_fn(|| {
                    (self.f)(&mut b, parameter);
                    b.elapsed.to_nanos() as f64
                });
                times.push(time);
                if let Some(run_metrics) = run_metrics {
                    for (n, v) in run_metrics {
                        metrics.entry(n).or_insert_with(|| Vec::new()).push(v);
                    }
                }
            }
            let mut return_recorded_metrics = false;
            {
                let mut lengths = metrics.iter().map(|(_, v)| v.len());
                let first = lengths.next();
                if let Some(first) = first {
                    if !lengths.all(|l| l == first) {
                        panic!("metrics out of sync!");
                    } else {
                        return_recorded_metrics = true;
                    }
                }
            }
            if return_recorded_metrics {
                (times, Some(metrics))
            } else {
                (times, None)
            }
        }
        fn warm_up(
            &mut self,
            _: &mut Option<Program>,
            how_long: Duration,
            parameter: &T,
        ) -> (u64, u64) {
            let f = &mut self.f;
            let mut b = Bencher {
                iters: 1,
                elapsed: Duration::from_secs(0),
            };
            let mut total_iters = 0;
            let start = Instant::now();
            loop {
                (*f)(&mut b, parameter);
                total_iters += b.iters;
                let elapsed = start.elapsed();
                if elapsed > how_long {
                    return (elapsed.to_nanos(), total_iters);
                }
                b.iters *= 2;
            }
        }
    }
}
#[cfg(feature = "html_reports")]
mod kde {
    use stats::univariate::kde::kernel::Gaussian;
    use stats::univariate::kde::{Bandwidth, Kde};
    use stats::univariate::Sample;
    pub fn sweep(
        sample: &Sample<f64>,
        npoints: usize,
        range: Option<(f64, f64)>,
    ) -> (Box<[f64]>, Box<[f64]>) {
        let (xs, ys, _) = sweep_and_estimate(sample, npoints, range, sample.as_slice()[0]);
        (xs, ys)
    }
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
    use kde;
    use report::BenchmarkId;
    use stats::bivariate::regression::Slope;
    use stats::bivariate::Data;
    use stats::univariate::outliers::tukey::LabeledSample;
    use stats::univariate::Sample;
    use stats::Distribution;
    use std::iter;
    use std::path::Path;
    use std::process::Child;
    pub mod both {
        use super::{debug_script, escape_underscores, scale_time};
        use super::{DARK_BLUE, DARK_RED, DEFAULT_FONT, KDE_POINTS, LINEWIDTH, SIZE};
        use criterion_plot::prelude::*;
        use estimate::Estimates;
        use estimate::Statistic::Slope;
        use kde;
        use report::BenchmarkId;
        use stats::bivariate::Data;
        use stats::univariate::Sample;
        use std::iter;
        use std::path::Path;
        use std::process::Child;
        use {ConfidenceInterval, Estimate};
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
            let max_iters = base_data.x().max().max(data.x().max());
            let max_elapsed = base_data.y().max().max(data.y().max());
            let (y_scale, prefix) = scale_time(max_elapsed);
            let exponent = (max_iters.log10() / 3.).floor() as i32 * 3;
            let x_scale = 10f64.powi(-exponent);
            let x_label = if exponent == 0 {
                "Iterations".to_owned()
            } else {
                format!("Iterations (x 10^{})", exponent)
            };
            let Estimate {
                confidence_interval:
                    ConfidenceInterval {
                        lower_bound: base_lb,
                        upper_bound: base_ub,
                        ..
                    },
                point_estimate: base_point,
                ..
            } = base_estimates[&Slope];
            let Estimate {
                confidence_interval:
                    ConfidenceInterval {
                        lower_bound: lb,
                        upper_bound: ub,
                        ..
                    },
                point_estimate: point,
                ..
            } = estimates[&Slope];
            let mut figure = Figure::new();
            if !thumbnail_mode {
                figure.set(Title(escape_underscores(id.id())));
            }
            figure
                .set(Font(DEFAULT_FONT))
                .set(size.unwrap_or(SIZE))
                .configure(Axis::BottomX, |a| {
                    a.configure(Grid::Major, |g| g.show())
                        .set(Label(x_label))
                        .set(ScaleFactor(x_scale))
                })
                .configure(Axis::LeftY, |a| {
                    a.configure(Grid::Major, |g| g.show())
                        .set(Label(format!("Total time ({}s)", prefix)))
                        .set(ScaleFactor(y_scale))
                })
                .configure(Key, |k| {
                    if thumbnail_mode {
                        k.hide();
                    }
                    k.set(Justification::Left)
                        .set(Order::SampleText)
                        .set(Position::Inside(Vertical::Top, Horizontal::Left))
                })
                .plot(
                    FilledCurve {
                        x: &[0., max_iters],
                        y1: &[0., base_lb],
                        y2: &[0., base_ub],
                    },
                    |c| c.set(DARK_RED).set(Opacity(0.25)),
                )
                .plot(
                    FilledCurve {
                        x: &[0., max_iters],
                        y1: &[0., lb],
                        y2: &[0., ub],
                    },
                    |c| c.set(DARK_BLUE).set(Opacity(0.25)),
                )
                .plot(
                    Lines {
                        x: &[0., max_iters],
                        y: &[0., base_point],
                    },
                    |c| {
                        c.set(DARK_RED)
                            .set(LINEWIDTH)
                            .set(Label("Base sample"))
                            .set(LineType::Solid)
                    },
                )
                .plot(
                    Lines {
                        x: &[0., max_iters],
                        y: &[0., point],
                    },
                    |c| {
                        c.set(DARK_BLUE)
                            .set(LINEWIDTH)
                            .set(Label("New sample"))
                            .set(LineType::Solid)
                    },
                );
            debug_script(&path, &figure);
            figure.set(Output(path.as_ref().to_owned())).draw().unwrap()
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
                figure.set(Title(escape_underscores(id.id())));
            }
            figure
                .set(Font(DEFAULT_FONT))
                .set(size.unwrap_or(SIZE))
                .configure(Axis::BottomX, |a| {
                    a.set(Label(format!("Average time ({}s)", prefix)))
                        .set(ScaleFactor(x_scale))
                })
                .configure(Axis::LeftY, |a| a.set(Label("Density (a.u.)")))
                .configure(Axis::RightY, |a| a.hide())
                .configure(Key, |k| {
                    if thumbnail_mode {
                        k.hide();
                    }
                    k.set(Justification::Left)
                        .set(Order::SampleText)
                        .set(Position::Outside(Vertical::Top, Horizontal::Right))
                })
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
    pub mod summary {
        use super::{debug_script, escape_underscores, scale_time};
        use super::{DARK_BLUE, DEFAULT_FONT, KDE_POINTS, LINEWIDTH, POINT_SIZE, SIZE};
        use criterion_plot::prelude::*;
        use itertools::Itertools;
        use kde;
        use report::{BenchmarkId, ValueType};
        use stats::univariate::Sample;
        use std::cmp::Ordering;
        use std::path::Path;
        use std::process::Child;
        use AxisScale;
        const NUM_COLORS: usize = 8;
        static COMPARISON_COLORS: [Color; NUM_COLORS] = [
            Color::Rgb(178, 34, 34),
            Color::Rgb(46, 139, 87),
            Color::Rgb(0, 139, 139),
            Color::Rgb(255, 215, 0),
            Color::Rgb(0, 0, 139),
            Color::Rgb(220, 20, 60),
            Color::Rgb(139, 0, 139),
            Color::Rgb(0, 255, 127),
        ];
        impl AxisScale {
            fn to_gnuplot(&self) -> Scale {
                match *self {
                    AxisScale::Linear => Scale::Linear,
                    AxisScale::Logarithmic => Scale::Logarithmic,
                }
            }
        }
        #[cfg_attr(feature = "cargo-clippy", allow(explicit_counter_loop))]
        pub fn line_comparison<P: AsRef<Path>>(
            group_id: &str,
            all_curves: &[&(BenchmarkId, Vec<f64>)],
            path: P,
            value_type: ValueType,
            axis_scale: AxisScale,
        ) -> Child {
            let mut f = Figure::new();
            let input_suffix = match value_type {
                ValueType::Bytes => " Size (Bytes)",
                ValueType::Elements => " Size (Elements)",
                ValueType::Value => "",
            };
            f.set(Font(DEFAULT_FONT))
                .set(SIZE)
                .configure(Key, |k| {
                    k.set(Justification::Left)
                        .set(Order::SampleText)
                        .set(Position::Outside(Vertical::Top, Horizontal::Right))
                })
                .set(Title(format!(
                    "{}: Comparison",
                    escape_underscores(group_id)
                )))
                .configure(Axis::BottomX, |a| {
                    a.set(Label(format!("Input{}", input_suffix)))
                        .set(axis_scale.to_gnuplot())
                });
            let mut max = 0.0;
            let mut i = 0;
            for (key, group) in &all_curves
                .into_iter()
                .group_by(|&&&(ref id, _)| &id.function_id)
            {
                let mut tuples: Vec<_> = group
                    .into_iter()
                    .map(|&&(ref id, ref sample)| {
                        let x = id.as_number().unwrap();
                        let y = Sample::new(sample).mean();
                        if y > max {
                            max = y;
                        }
                        (x, y)
                    })
                    .collect();
                tuples
                    .sort_by(|&(ax, _), &(bx, _)| (ax.partial_cmp(&bx).unwrap_or(Ordering::Less)));
                let (xs, ys): (Vec<_>, Vec<_>) = tuples.into_iter().unzip();
                let function_name = key
                    .as_ref()
                    .map(|string| escape_underscores(string))
                    .unwrap();
                f.plot(Lines { x: &xs, y: &ys }, |c| {
                    c.set(LINEWIDTH)
                        .set(Label(function_name))
                        .set(LineType::Solid)
                        .set(COMPARISON_COLORS[i % NUM_COLORS])
                })
                .plot(Points { x: &xs, y: &ys }, |p| {
                    p.set(PointType::FilledCircle)
                        .set(POINT_SIZE)
                        .set(COMPARISON_COLORS[i % NUM_COLORS])
                });
                i += 1;
            }
            let (scale, prefix) = scale_time(max);
            f.configure(Axis::LeftY, |a| {
                a.configure(Grid::Major, |g| g.show())
                    .configure(Grid::Minor, |g| g.hide())
                    .set(Label(format!("Average time ({}s)", prefix)))
                    .set(axis_scale.to_gnuplot())
                    .set(ScaleFactor(scale))
            });
            debug_script(&path, &f);
            f.set(Output(path.as_ref().to_owned())).draw().unwrap()
        }
        pub fn violin<P: AsRef<Path>>(
            group_id: &str,
            all_curves: &[&(BenchmarkId, Vec<f64>)],
            path: P,
            axis_scale: AxisScale,
        ) -> Child {
            let all_curves_vec = all_curves.iter().rev().map(|&t| t).collect::<Vec<_>>();
            let all_curves: &[&(BenchmarkId, Vec<f64>)] = &*all_curves_vec;
            let kdes = all_curves
                .iter()
                .map(|&&(_, ref sample)| {
                    let (x, mut y) = kde::sweep(Sample::new(sample), KDE_POINTS, None);
                    let y_max = Sample::new(&y).max();
                    for y in y.iter_mut() {
                        *y /= y_max;
                    }
                    (x, y)
                })
                .collect::<Vec<_>>();
            let mut xs = kdes
                .iter()
                .flat_map(|&(ref x, _)| x.iter())
                .filter(|&&x| x > 0.);
            let (mut min, mut max) = {
                let &first = xs.next().unwrap();
                (first, first)
            };
            for &e in xs {
                if e < min {
                    min = e;
                } else if e > max {
                    max = e;
                }
            }
            let (scale, prefix) = scale_time(max);
            let tics = || (0..).map(|x| (f64::from(x)) + 0.5);
            let size = Size(1280, 200 + (25 * all_curves.len()));
            let mut f = Figure::new();
            f.set(Font(DEFAULT_FONT))
                .set(size)
                .set(Title(format!(
                    "{}: Violin plot",
                    escape_underscores(group_id)
                )))
                .configure(Axis::BottomX, |a| {
                    a.configure(Grid::Major, |g| g.show())
                        .configure(Grid::Minor, |g| g.hide())
                        .set(Label(format!("Average time ({}s)", prefix)))
                        .set(axis_scale.to_gnuplot())
                        .set(ScaleFactor(scale))
                })
                .configure(Axis::LeftY, |a| {
                    a.set(Label("Input"))
                        .set(Range::Limits(0., all_curves.len() as f64))
                        .set(TicLabels {
                            positions: tics(),
                            labels: all_curves
                                .iter()
                                .map(|&&(ref id, _)| escape_underscores(id.id())),
                        })
                });
            let mut is_first = true;
            for (i, &(ref x, ref y)) in kdes.iter().enumerate() {
                let i = i as f64 + 0.5;
                let y1 = y.iter().map(|&y| i + y * 0.5);
                let y2 = y.iter().map(|&y| i - y * 0.5);
                f.plot(
                    FilledCurve {
                        x: &**x,
                        y1: y1,
                        y2: y2,
                    },
                    |c| {
                        if is_first {
                            is_first = false;
                            c.set(DARK_BLUE).set(Label("PDF")).set(Opacity(0.25))
                        } else {
                            c.set(DARK_BLUE).set(Opacity(0.25))
                        }
                    },
                );
            }
            debug_script(&path, &f);
            f.set(Output(path.as_ref().to_owned())).draw().unwrap()
        }
    }
    fn escape_underscores(string: &str) -> String {
        string.replace("_", "\\_")
    }
    fn scale_time(ns: f64) -> (f64, &'static str) {
        if ns < 10f64.powi(0) {
            (10f64.powi(3), "p")
        } else if ns < 10f64.powi(3) {
            (10f64.powi(0), "n")
        } else if ns < 10f64.powi(6) {
            (10f64.powi(-3), "u")
        } else if ns < 10f64.powi(9) {
            (10f64.powi(-6), "m")
        } else {
            (10f64.powi(-9), "")
        }
    }
    static DEFAULT_FONT: &'static str = "Helvetica";
    static KDE_POINTS: usize = 500;
    static SIZE: Size = Size(1280, 720);
    const LINEWIDTH: LineWidth = LineWidth(2.);
    const POINT_SIZE: PointSize = PointSize(0.75);
    const DARK_BLUE: Color = Color::Rgb(31, 120, 180);
    const DARK_ORANGE: Color = Color::Rgb(255, 127, 0);
    const DARK_RED: Color = Color::Rgb(227, 26, 28);
    fn debug_script<P: AsRef<Path>>(path: P, figure: &Figure) {
        if ::debug_enabled() {
            let mut script_path = path.as_ref().to_owned();
            script_path.set_extension("gnuplot");
            println!("Writing gnuplot script to {:?}", script_path);
            let result = figure.save(script_path.as_path());
            if let Err(e) = result {
                error!("Failed to write debug output: {}", e);
            }
        }
    }
    pub fn pdf_small<P: AsRef<Path>>(sample: &Sample<f64>, path: P, size: Option<Size>) -> Child {
        let (x_scale, prefix) = scale_time(sample.max());
        let mean = sample.mean();
        let (xs, ys, mean_y) = kde::sweep_and_estimate(sample, KDE_POINTS, None, mean);
        let xs_ = Sample::new(&xs);
        let ys_ = Sample::new(&ys);
        let y_limit = ys_.max() * 1.1;
        let zeros = iter::repeat(0);
        let mut figure = Figure::new();
        figure
            .set(Font(DEFAULT_FONT))
            .set(size.unwrap_or(SIZE))
            .configure(Axis::BottomX, |a| {
                a.set(Label(format!("Average time ({}s)", prefix)))
                    .set(Range::Limits(xs_.min() * x_scale, xs_.max() * x_scale))
                    .set(ScaleFactor(x_scale))
            })
            .configure(Axis::LeftY, |a| {
                a.set(Label("Density (a.u.)"))
                    .set(Range::Limits(0., y_limit))
            })
            .configure(Axis::RightY, |a| a.hide())
            .configure(Key, |k| k.hide())
            .plot(
                FilledCurve {
                    x: &*xs,
                    y1: &*ys,
                    y2: zeros,
                },
                |c| {
                    c.set(Axes::BottomXRightY)
                        .set(DARK_BLUE)
                        .set(Label("PDF"))
                        .set(Opacity(0.25))
                },
            )
            .plot(
                Lines {
                    x: &[mean, mean],
                    y: &[0., mean_y],
                },
                |c| c.set(DARK_BLUE).set(LINEWIDTH).set(Label("Mean")),
            );
        debug_script(&path, &figure);
        figure.set(Output(path.as_ref().to_owned())).draw().unwrap()
    }
    pub fn pdf<P: AsRef<Path>>(
        data: Data<f64, f64>,
        labeled_sample: LabeledSample<f64>,
        id: &BenchmarkId,
        path: P,
        size: Option<Size>,
    ) -> Child {
        let (x_scale, prefix) = scale_time(labeled_sample.max());
        let mean = labeled_sample.mean();
        let &max_iters = data
            .x()
            .as_slice()
            .iter()
            .max_by_key(|&&iters| iters as u64)
            .unwrap();
        let exponent = (max_iters.log10() / 3.).floor() as i32 * 3;
        let y_scale = 10f64.powi(-exponent);
        let y_label = if exponent == 0 {
            "Iterations".to_owned()
        } else {
            format!("Iterations (x 10^{})", exponent)
        };
        let (xs, ys) = kde::sweep(&labeled_sample, KDE_POINTS, None);
        let xs_ = Sample::new(&xs);
        let (lost, lomt, himt, hist) = labeled_sample.fences();
        let vertical = &[0., max_iters];
        let zeros = iter::repeat(0);
        let mut figure = Figure::new();
        figure
            .set(Font(DEFAULT_FONT))
            .set(size.unwrap_or(SIZE))
            .configure(Axis::BottomX, |a| {
                a.set(Label(format!("Average time ({}s)", prefix)))
                    .set(Range::Limits(xs_.min() * x_scale, xs_.max() * x_scale))
                    .set(ScaleFactor(x_scale))
            })
            .configure(Axis::LeftY, |a| {
                a.set(Label(y_label))
                    .set(Range::Limits(0., max_iters * y_scale))
                    .set(ScaleFactor(y_scale))
            })
            .configure(Axis::RightY, |a| a.set(Label("Density (a.u.)")))
            .configure(Key, |k| {
                k.set(Justification::Left)
                    .set(Order::SampleText)
                    .set(Position::Outside(Vertical::Top, Horizontal::Right))
            })
            .plot(
                FilledCurve {
                    x: &*xs,
                    y1: &*ys,
                    y2: zeros,
                },
                |c| {
                    c.set(Axes::BottomXRightY)
                        .set(DARK_BLUE)
                        .set(Label("PDF"))
                        .set(Opacity(0.25))
                },
            )
            .plot(
                Lines {
                    x: &[mean, mean],
                    y: vertical,
                },
                |c| {
                    c.set(DARK_BLUE)
                        .set(LINEWIDTH)
                        .set(LineType::Dash)
                        .set(Label("Mean"))
                },
            )
            .plot(
                Points {
                    x: labeled_sample.iter().filter_map(|(t, label)| {
                        if label.is_outlier() {
                            None
                        } else {
                            Some(t)
                        }
                    }),
                    y: labeled_sample
                        .iter()
                        .zip(data.x().as_slice().iter())
                        .filter_map(
                            |((_, label), i)| {
                                if label.is_outlier() {
                                    None
                                } else {
                                    Some(i)
                                }
                            },
                        ),
                },
                |c| {
                    c.set(DARK_BLUE)
                        .set(Label("\"Clean\" sample"))
                        .set(PointType::FilledCircle)
                        .set(POINT_SIZE)
                },
            )
            .plot(
                Points {
                    x: labeled_sample.iter().filter_map(|(x, label)| {
                        if label.is_mild() {
                            Some(x)
                        } else {
                            None
                        }
                    }),
                    y: labeled_sample
                        .iter()
                        .zip(data.x().as_slice().iter())
                        .filter_map(
                            |((_, label), i)| {
                                if label.is_mild() {
                                    Some(i)
                                } else {
                                    None
                                }
                            },
                        ),
                },
                |c| {
                    c.set(DARK_ORANGE)
                        .set(Label("Mild outliers"))
                        .set(POINT_SIZE)
                        .set(PointType::FilledCircle)
                },
            )
            .plot(
                Points {
                    x: labeled_sample.iter().filter_map(|(x, label)| {
                        if label.is_severe() {
                            Some(x)
                        } else {
                            None
                        }
                    }),
                    y: labeled_sample
                        .iter()
                        .zip(data.x().as_slice().iter())
                        .filter_map(
                            |((_, label), i)| {
                                if label.is_severe() {
                                    Some(i)
                                } else {
                                    None
                                }
                            },
                        ),
                },
                |c| {
                    c.set(DARK_RED)
                        .set(Label("Severe outliers"))
                        .set(POINT_SIZE)
                        .set(PointType::FilledCircle)
                },
            )
            .plot(
                Lines {
                    x: &[lomt, lomt],
                    y: vertical,
                },
                |c| c.set(DARK_ORANGE).set(LINEWIDTH).set(LineType::Dash),
            )
            .plot(
                Lines {
                    x: &[himt, himt],
                    y: vertical,
                },
                |c| c.set(DARK_ORANGE).set(LINEWIDTH).set(LineType::Dash),
            )
            .plot(
                Lines {
                    x: &[lost, lost],
                    y: vertical,
                },
                |c| c.set(DARK_RED).set(LINEWIDTH).set(LineType::Dash),
            )
            .plot(
                Lines {
                    x: &[hist, hist],
                    y: vertical,
                },
                |c| c.set(DARK_RED).set(LINEWIDTH).set(LineType::Dash),
            );
        figure.set(Title(escape_underscores(id.id())));
        debug_script(&path, &figure);
        figure.set(Output(path.as_ref().to_owned())).draw().unwrap()
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
        let (max_iters, max_elapsed) = (data.x().max(), data.y().max());
        let (y_scale, prefix) = scale_time(max_elapsed);
        let exponent = (max_iters.log10() / 3.).floor() as i32 * 3;
        let x_scale = 10f64.powi(-exponent);
        let x_label = if exponent == 0 {
            "Iterations".to_owned()
        } else {
            format!("Iterations (x 10^{})", exponent)
        };
        let lb = lb.0 * max_iters;
        let point = point.0 * max_iters;
        let ub = ub.0 * max_iters;
        let max_iters = max_iters;
        let mut figure = Figure::new();
        figure
            .set(Font(DEFAULT_FONT))
            .set(size.unwrap_or(SIZE))
            .configure(Key, |k| {
                if thumbnail_mode {
                    k.hide();
                }
                k.set(Justification::Left)
                    .set(Order::SampleText)
                    .set(Position::Inside(Vertical::Top, Horizontal::Left))
            })
            .configure(Axis::BottomX, |a| {
                a.configure(Grid::Major, |g| g.show())
                    .set(Label(x_label))
                    .set(ScaleFactor(x_scale))
            })
            .configure(Axis::LeftY, |a| {
                a.configure(Grid::Major, |g| g.show())
                    .set(Label(format!("Total time ({}s)", prefix)))
                    .set(ScaleFactor(y_scale))
            })
            .plot(
                Points {
                    x: data.x().as_slice(),
                    y: data.y().as_slice(),
                },
                |c| {
                    c.set(DARK_BLUE)
                        .set(Label("Sample"))
                        .set(PointSize(0.5))
                        .set(PointType::FilledCircle)
                },
            )
            .plot(
                Lines {
                    x: &[0., max_iters],
                    y: &[0., point],
                },
                |c| {
                    c.set(DARK_BLUE)
                        .set(LINEWIDTH)
                        .set(Label("Linear regression"))
                        .set(LineType::Solid)
                },
            )
            .plot(
                FilledCurve {
                    x: &[0., max_iters],
                    y1: &[0., lb],
                    y2: &[0., ub],
                },
                |c| {
                    c.set(DARK_BLUE)
                        .set(Label("Confidence interval"))
                        .set(Opacity(0.25))
                },
            );
        if !thumbnail_mode {
            figure.set(Title(escape_underscores(id.id())));
        }
        debug_script(&path, &figure);
        figure.set(Output(path.as_ref().to_owned())).draw().unwrap()
    }
    pub(crate) fn abs_distributions<P: AsRef<Path>>(
        distributions: &Distributions,
        estimates: &Estimates,
        id: &BenchmarkId,
        output_directory: P,
    ) -> Vec<Child> {
        distributions
            .iter()
            .map(|(&statistic, distribution)| {
                let path = output_directory
                    .as_ref()
                    .join(id.to_string())
                    .join("report")
                    .join(format!("{}.svg", statistic));
                let estimate = estimates[&statistic];
                let ci = estimate.confidence_interval;
                let (lb, ub) = (ci.lower_bound, ci.upper_bound);
                let start = lb - (ub - lb) / 9.;
                let end = ub + (ub - lb) / 9.;
                let (xs, ys) = kde::sweep(distribution, KDE_POINTS, Some((start, end)));
                let xs_ = Sample::new(&xs);
                let (x_scale, prefix) = scale_time(xs_.max());
                let y_scale = x_scale.recip();
                let p = estimate.point_estimate;
                let n_p = xs.iter().enumerate().find(|&(_, &x)| x >= p).unwrap().0;
                let y_p = ys[n_p - 1]
                    + (ys[n_p] - ys[n_p - 1]) / (xs[n_p] - xs[n_p - 1]) * (p - xs[n_p - 1]);
                let zero = iter::repeat(0);
                let start = xs.iter().enumerate().find(|&(_, &x)| x >= lb).unwrap().0;
                let end = xs
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|&(_, &x)| x <= ub)
                    .unwrap()
                    .0;
                let len = end - start;
                let mut figure = Figure::new();
                figure
                    .set(Font(DEFAULT_FONT))
                    .set(SIZE)
                    .set(Title(format!(
                        "{}: {}",
                        escape_underscores(id.id()),
                        statistic
                    )))
                    .configure(Axis::BottomX, |a| {
                        a.set(Label(format!("Average time ({}s)", prefix)))
                            .set(Range::Limits(xs_.min() * x_scale, xs_.max() * x_scale))
                            .set(ScaleFactor(x_scale))
                    })
                    .configure(Axis::LeftY, |a| {
                        a.set(Label("Density (a.u.)")).set(ScaleFactor(y_scale))
                    })
                    .configure(Key, |k| {
                        k.set(Justification::Left)
                            .set(Order::SampleText)
                            .set(Position::Outside(Vertical::Top, Horizontal::Right))
                    })
                    .plot(Lines { x: &*xs, y: &*ys }, |c| {
                        c.set(DARK_BLUE)
                            .set(LINEWIDTH)
                            .set(Label("Bootstrap distribution"))
                            .set(LineType::Solid)
                    })
                    .plot(
                        FilledCurve {
                            x: xs.iter().skip(start).take(len),
                            y1: ys.iter().skip(start),
                            y2: zero,
                        },
                        |c| {
                            c.set(DARK_BLUE)
                                .set(Label("Confidence interval"))
                                .set(Opacity(0.25))
                        },
                    )
                    .plot(
                        Lines {
                            x: &[p, p],
                            y: &[0., y_p],
                        },
                        |c| {
                            c.set(DARK_BLUE)
                                .set(LINEWIDTH)
                                .set(Label("Point estimate"))
                                .set(LineType::Dash)
                        },
                    );
                debug_script(&path, &figure);
                figure.set(Output(path)).draw().unwrap()
            })
            .collect::<Vec<_>>()
    }
    pub(crate) fn rel_distributions<P: AsRef<Path>>(
        distributions: &Distributions,
        estimates: &Estimates,
        id: &BenchmarkId,
        output_directory: P,
        nt: f64,
    ) -> Vec<Child> {
        let mut figure = Figure::new();
        figure
            .set(Font(DEFAULT_FONT))
            .set(SIZE)
            .configure(Axis::LeftY, |a| a.set(Label("Density (a.u.)")))
            .configure(Key, |k| {
                k.set(Justification::Left)
                    .set(Order::SampleText)
                    .set(Position::Outside(Vertical::Top, Horizontal::Right))
            });
        distributions
            .iter()
            .map(|(&statistic, distribution)| {
                let path = output_directory
                    .as_ref()
                    .join(id.to_string())
                    .join("report")
                    .join("change")
                    .join(format!("{}.svg", statistic));
                let estimate = estimates[&statistic];
                let ci = estimate.confidence_interval;
                let (lb, ub) = (ci.lower_bound, ci.upper_bound);
                let start = lb - (ub - lb) / 9.;
                let end = ub + (ub - lb) / 9.;
                let (xs, ys) = kde::sweep(distribution, KDE_POINTS, Some((start, end)));
                let xs_ = Sample::new(&xs);
                let p = estimate.point_estimate;
                let n_p = xs.iter().enumerate().find(|&(_, &x)| x >= p).unwrap().0;
                let y_p = ys[n_p - 1]
                    + (ys[n_p] - ys[n_p - 1]) / (xs[n_p] - xs[n_p - 1]) * (p - xs[n_p - 1]);
                let one = iter::repeat(1);
                let zero = iter::repeat(0);
                let start = xs.iter().enumerate().find(|&(_, &x)| x >= lb).unwrap().0;
                let end = xs
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|&(_, &x)| x <= ub)
                    .unwrap()
                    .0;
                let len = end - start;
                let x_min = xs_.min();
                let x_max = xs_.max();
                let (fc_start, fc_end) = if nt < x_min || -nt > x_max {
                    let middle = (x_min + x_max) / 2.;
                    (middle, middle)
                } else {
                    (
                        if -nt < x_min { x_min } else { -nt },
                        if nt > x_max { x_max } else { nt },
                    )
                };
                let mut figure = figure.clone();
                figure
                    .set(Title(format!(
                        "{}: {}",
                        escape_underscores(id.id()),
                        statistic
                    )))
                    .configure(Axis::BottomX, |a| {
                        a.set(Label("Relative change (%)"))
                            .set(Range::Limits(x_min * 100., x_max * 100.))
                            .set(ScaleFactor(100.))
                    })
                    .plot(Lines { x: &*xs, y: &*ys }, |c| {
                        c.set(DARK_BLUE)
                            .set(LINEWIDTH)
                            .set(Label("Bootstrap distribution"))
                            .set(LineType::Solid)
                    })
                    .plot(
                        FilledCurve {
                            x: xs.iter().skip(start).take(len),
                            y1: ys.iter().skip(start),
                            y2: zero.clone(),
                        },
                        |c| {
                            c.set(DARK_BLUE)
                                .set(Label("Confidence interval"))
                                .set(Opacity(0.25))
                        },
                    )
                    .plot(
                        Lines {
                            x: &[p, p],
                            y: &[0., y_p],
                        },
                        |c| {
                            c.set(DARK_BLUE)
                                .set(LINEWIDTH)
                                .set(Label("Point estimate"))
                                .set(LineType::Dash)
                        },
                    )
                    .plot(
                        FilledCurve {
                            x: &[fc_start, fc_end],
                            y1: one,
                            y2: zero,
                        },
                        |c| {
                            c.set(Axes::BottomXRightY)
                                .set(DARK_RED)
                                .set(Label("Noise threshold"))
                                .set(Opacity(0.1))
                        },
                    );
                debug_script(&path, &figure);
                figure.set(Output(path)).draw().unwrap()
            })
            .collect::<Vec<_>>()
    }
    pub fn t_test<P: AsRef<Path>>(
        t: f64,
        distribution: &Distribution<f64>,
        id: &BenchmarkId,
        output_directory: P,
    ) -> Child {
        let path = output_directory
            .as_ref()
            .join(id.to_string())
            .join("report")
            .join("change")
            .join("t-test.svg");
        let (xs, ys) = kde::sweep(distribution, KDE_POINTS, None);
        let zero = iter::repeat(0);
        let mut figure = Figure::new();
        figure
            .set(Font(DEFAULT_FONT))
            .set(SIZE)
            .set(Title(format!(
                "{}: Welch t test",
                escape_underscores(id.id())
            )))
            .configure(Axis::BottomX, |a| a.set(Label("t score")))
            .configure(Axis::LeftY, |a| a.set(Label("Density")))
            .configure(Key, |k| {
                k.set(Justification::Left)
                    .set(Order::SampleText)
                    .set(Position::Outside(Vertical::Top, Horizontal::Right))
            })
            .plot(
                FilledCurve {
                    x: &*xs,
                    y1: &*ys,
                    y2: zero,
                },
                |c| {
                    c.set(DARK_BLUE)
                        .set(Label("t distribution"))
                        .set(Opacity(0.25))
                },
            )
            .plot(
                Lines {
                    x: &[t, t],
                    y: &[0, 1],
                },
                |c| {
                    c.set(Axes::BottomXRightY)
                        .set(DARK_BLUE)
                        .set(LINEWIDTH)
                        .set(Label("t statistic"))
                        .set(LineType::Solid)
                },
            );
        debug_script(&path, &figure);
        figure.set(Output(path)).draw().unwrap()
    }
    trait Append<T> {
        fn append_(self, item: T) -> Self;
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
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::process::Child;
    use Estimate;
    const THUMBNAIL_SIZE: Size = Size(450, 300);
    fn wait_on_gnuplot(children: Vec<Child>) {
        let start = ::std::time::Instant::now();
        let child_count = children.len();
        for child in children {
            match child.wait_with_output() {
                Ok(ref out) if out.status.success() => {}
                Ok(out) => error!("Error in Gnuplot: {}", String::from_utf8_lossy(&out.stderr)),
                Err(e) => error!("Got IO error while waiting for Gnuplot to complete: {}", e),
            }
        }
        let elapsed = &start.elapsed();
        info!(
            "Waiting for {} gnuplot processes took {}",
            child_count,
            ::format::time(::DurationExt::to_nanos(elapsed) as f64)
        );
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
    impl IndividualBenchmark {
        fn new(path_prefix: &str, id: &BenchmarkId) -> IndividualBenchmark {
            IndividualBenchmark {
                name: id.id().to_owned(),
                path: format!("{}/{}", path_prefix, id.id()),
            }
        }
    }
    #[derive(Serialize)]
    struct SummaryContext {
        group_id: String,
        thumbnail_width: usize,
        thumbnail_height: usize,
        violin_plot: Option<PathBuf>,
        line_chart: Option<PathBuf>,
        benchmarks: Vec<IndividualBenchmark>,
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
            Plot {
                name: name.to_owned(),
                url: url.to_owned(),
            }
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
        fn benchmark_start(&self, _: &BenchmarkId, _: &ReportContext) {}
        fn warmup(&self, _: &BenchmarkId, _: &ReportContext, _: f64) {}
        fn analysis(&self, _: &BenchmarkId, _: &ReportContext) {}
        fn measurement_start(&self, _: &BenchmarkId, _: &ReportContext, _: u64, _: f64, _: u64) {}
        fn measurement_complete(
            &self,
            id: &BenchmarkId,
            report_context: &ReportContext,
            measurements: &MeasurementData,
        ) {
            if !report_context.plotting.is_enabled() {
                return;
            }
            try_else_return!(fs::mkdirp(
                &report_context
                    .output_directory
                    .join(id.to_string())
                    .join("report")
            ));
            let slope_estimate = &measurements.absolute_estimates[&Statistic::Slope];
            fn time_interval(est: &Estimate) -> ConfidenceInterval {
                ConfidenceInterval {
                    lower: format::time(est.confidence_interval.lower_bound),
                    point: format::time(est.point_estimate),
                    upper: format::time(est.confidence_interval.upper_bound),
                }
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
            if !context.plotting.is_enabled() {
                return;
            }
            let mut all_plots = vec![];
            let group_id = &all_ids[0].group_id;
            let mut function_ids = BTreeSet::new();
            for id in all_ids.iter() {
                if let Some(ref function_id) = id.function_id {
                    function_ids.insert(function_id);
                }
            }
            let data: Vec<(BenchmarkId, Vec<f64>)> =
                self.load_summary_data(&context.output_directory, all_ids);
            for function_id in function_ids {
                let samples_with_function: Vec<_> = data
                    .iter()
                    .by_ref()
                    .filter(|&&(ref id, _)| id.function_id.as_ref() == Some(function_id))
                    .collect();
                if samples_with_function.len() > 1 {
                    let subgroup_id = format!("{}/{}", group_id, function_id);
                    all_plots.extend(self.generate_summary(
                        &subgroup_id,
                        &*samples_with_function,
                        context,
                        false,
                    ));
                }
            }
            all_plots.extend(self.generate_summary(
                group_id,
                &*(data.iter().by_ref().collect::<Vec<_>>()),
                context,
                true,
            ));
            wait_on_gnuplot(all_plots)
        }
    }
    impl Html {
        fn comparison(&self, measurements: &MeasurementData) -> Option<Comparison> {
            if let Some(ref comp) = measurements.comparison {
                let different_mean = comp.p_value < comp.significance_threshold;
                let mean_est = comp.relative_estimates[&Statistic::Mean];
                let explanation_str: String;
                if !different_mean {
                    explanation_str = "No change in performance detected.".to_owned();
                } else {
                    let comparison = compare_to_threshold(&mean_est, comp.noise_threshold);
                    match comparison {
                        ComparisonResult::Improved => {
                            explanation_str = "Performance has improved.".to_owned();
                        }
                        ComparisonResult::Regressed => {
                            explanation_str = "Performance has regressed.".to_owned();
                        }
                        ComparisonResult::NonSignificant => {
                            explanation_str = "Change within noise threshold.".to_owned();
                        }
                    }
                }
                let comp = Comparison {
                    p_value: format!("{:.2}", comp.p_value),
                    inequality: (if different_mean { "<" } else { ">" }).to_owned(),
                    significance_level: format!("{:.2}", comp.significance_threshold),
                    explanation: explanation_str,
                    change: ConfidenceInterval {
                        point: format::change(mean_est.point_estimate, true),
                        lower: format::change(mean_est.confidence_interval.lower_bound, true),
                        upper: format::change(mean_est.confidence_interval.upper_bound, true),
                    },
                    additional_plots: vec![
                        Plot::new("Change in mean", "change/mean.svg"),
                        Plot::new("Change in median", "change/median.svg"),
                        Plot::new("T-Test", "change/t-test.svg"),
                    ],
                };
                Some(comp)
            } else {
                None
            }
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
        fn load_summary_data<P: AsRef<Path>>(
            &self,
            output_dir: P,
            all_ids: &[BenchmarkId],
        ) -> Vec<(BenchmarkId, Vec<f64>)> {
            let output_dir = output_dir.as_ref();
            all_ids
                .iter()
                .filter_map(|id| {
                    let entry = output_dir.join(id.id()).join("new");
                    let (iters, times): (Vec<f64>, Vec<f64>) =
                        try_else_return!(fs::load(&entry.join("sample.json")), || None);
                    let avg_times = iters
                        .into_iter()
                        .zip(times.into_iter())
                        .map(|(iters, time)| time / iters)
                        .collect::<Vec<_>>();
                    Some((id.clone(), avg_times))
                })
                .collect::<Vec<_>>()
        }
        fn generate_summary(
            &self,
            group_id: &str,
            data: &[&(BenchmarkId, Vec<f64>)],
            report_context: &ReportContext,
            full_summary: bool,
        ) -> Vec<Child> {
            let mut gnuplots = vec![];
            let report_dir = report_context
                .output_directory
                .join(group_id)
                .join("report");
            try_else_return!(fs::mkdirp(&report_dir), || gnuplots);
            let violin_path = report_dir.join("violin.svg");
            gnuplots.push(plot::summary::violin(
                group_id,
                data,
                &violin_path,
                report_context.plot_config.summary_scale,
            ));
            let value_types: Vec<_> = data.iter().map(|&&(ref id, _)| id.value_type()).collect();
            let function_types: BTreeSet<_> =
                data.iter().map(|&&(ref id, _)| &id.function_id).collect();
            let mut line_path = None;
            if value_types.iter().all(|x| x == &value_types[0]) && function_types.len() > 1 {
                if let Some(value_type) = value_types[0] {
                    let path = report_dir.join("lines.svg");
                    gnuplots.push(plot::summary::line_comparison(
                        group_id,
                        data,
                        &path,
                        value_type,
                        report_context.plot_config.summary_scale,
                    ));
                    line_path = Some(path);
                }
            }
            let path_prefix = if full_summary {
                "../../.."
            } else {
                "../../../.."
            };
            let benchmarks = data
                .iter()
                .map(|&&(ref id, _)| IndividualBenchmark::new(path_prefix, id))
                .collect();
            let context = SummaryContext {
                group_id: group_id.to_owned(),
                thumbnail_width: THUMBNAIL_SIZE.0,
                thumbnail_height: THUMBNAIL_SIZE.1,
                violin_plot: Some(violin_path),
                line_chart: line_path,
                benchmarks: benchmarks,
            };
            let text = self
                .handlebars
                .render("summary_report", &context)
                .expect("Failed to render summary report template");
            try_else_return!(
                fs::save_string(&text, &report_dir.join("index.html")),
                || gnuplots
            );
            gnuplots
        }
    }
    enum ComparisonResult {
        Improved,
        Regressed,
        NonSignificant,
    }
    fn compare_to_threshold(estimate: &Estimate, noise: f64) -> ComparisonResult {
        let ci = estimate.confidence_interval;
        let lb = ci.lower_bound;
        let ub = ci.upper_bound;
        if lb < -noise && ub < -noise {
            ComparisonResult::Improved
        } else if lb > noise && ub > noise {
            ComparisonResult::Regressed
        } else {
            ComparisonResult::NonSignificant
        }
    }
}
use benchmark::BenchmarkConfig;
use benchmark::NamedRoutine;
pub use benchmark::{Benchmark, BenchmarkDefinition, ParameterizedBenchmark};
use estimate::{Distributions, Estimates, Statistic};
#[cfg(feature = "html_reports")]
use html::Html;
use plotting::Plotting;
use report::{CliReport, Report, Reports};
use routine::Function;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};
use std::{fmt, mem};
fn debug_enabled() -> bool {
    std::env::vars().any(|(key, _)| key == "CRITERION_DEBUG")
}
#[cfg(not(feature = "real_blackbox"))]
pub fn black_box<T>(dummy: T) -> T {
    unsafe {
        let ret = std::ptr::read_volatile(&dummy);
        std::mem::forget(dummy);
        ret
    }
}
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
        let start = Instant::now();
        for _ in 0..self.iters {
            black_box(routine());
        }
        self.elapsed = start.elapsed();
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
                Plotting::Enabled
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
impl Criterion {
    pub fn with_filter<S: Into<String>>(mut self, filter: S) -> Criterion {
        self.filter = Some(filter.into());
        self
    }
    fn filter_matches(&self, id: &str) -> bool {
        match self.filter {
            Some(ref string) => id.contains(string),
            None => true,
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
            match *self {
                Plotting::Enabled => true,
                _ => false,
            }
        }
    }
}
trait DurationExt {
    fn to_nanos(&self) -> u64;
}
const NANOS_PER_SEC: u64 = 1_000_000_000;
impl DurationExt for Duration {
    fn to_nanos(&self) -> u64 {
        self.as_secs() * NANOS_PER_SEC + u64::from(self.subsec_nanos())
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
impl Estimate {}
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
