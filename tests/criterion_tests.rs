#[macro_use]
extern crate criterion;
extern crate serde_json;
extern crate tempdir;
extern crate walkdir;

use std::fs::File;
use criterion::{Benchmark, Criterion, Fun, ParameterizedBenchmark, Throughput};
use criterion::BenchmarkDefinition;
use std::time::Duration;
use std::path::PathBuf;
use walkdir::WalkDir;
use std::rc::Rc;
use std::cell::RefCell;
use std::process::{Command, Stdio};
use serde_json::value::Value;
use tempdir::TempDir;

/*
 * Please note that these tests are not complete examples of how to use
 * Criterion.rs. See the benches folder for actual examples.
 */
fn temp_dir() -> TempDir {
    TempDir::new("").unwrap()
}

// Configure a Criterion struct to perform really fast benchmarks. This is not
// recommended for real benchmarking, only for testing.
fn short_benchmark(dir: &TempDir) -> Criterion {
    Criterion::default()
        .output_directory(dir.path())
        .warm_up_time(Duration::from_millis(250))
        .measurement_time(Duration::from_millis(500))
        .nresamples(1000)
}

#[test]
fn test_bench_function() {
    let dir = temp_dir();
    Benchmark::new("test_bench_function", move |b| b.iter(|| 10)).run("test_bench_function", &short_benchmark(&dir));
}
