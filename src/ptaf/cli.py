from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import (
    ESNET_SIGNALS,
    MIT_GPU_METRICS,
    TELECOM_KPIS,
    load_prepared,
    prepare_esnet,
    prepare_mit_supercloud_gpu,
    prepare_telecomts,
    summarize_series,
)
from .experiment import ExperimentConfig, run_experiment
from .correlation import write_correlation_analysis
from .precursors import write_event_precursor_analysis
from .robustness import write_results_sensitivity
from .baselines import write_trivial_baselines
from .breadth import build_trace_manifest
from .models import SUPPORTED_MODELS
from .paths import data_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptaf",
        description="Excursion forecastability experiments for systems telemetry",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    paths = sub.add_parser("paths")
    paths.set_defaults(handler=_cmd_paths)

    prepare = sub.add_parser("prepare", help="prepare benchmark data")
    prep = prepare.add_subparsers(dest="dataset", required=True)

    telecom = prep.add_parser("telecomts")
    telecom.add_argument("--root", type=Path, default=data_root()/"raw"/"telecomts")
    telecom.add_argument("--output", type=Path, default=data_root()/"processed"/"telecomts.csv")
    telecom.add_argument("--scope", choices=["normal","anomalous","all"], default="normal")
    telecom.add_argument("--kpis", nargs="+", choices=TELECOM_KPIS, default=list(TELECOM_KPIS))
    telecom.add_argument("--resample-ms", type=int, default=1000)
    telecom.set_defaults(handler=_cmd_prepare_telecom)

    mit = prep.add_parser("mit-gpu")
    mit.add_argument("--root", type=Path, default=data_root()/"raw"/"mit_supercloud"/"gpu_labeled_sample")
    mit.add_argument("--metric", choices=MIT_GPU_METRICS, default="utilization_gpu_pct")
    mit.add_argument("--output", type=Path, default=None)
    mit.add_argument("--resample-ms", type=int, default=1000)
    mit.set_defaults(handler=_cmd_prepare_mit)

    esnet = prep.add_parser("esnet")
    esnet.add_argument("--raw", type=Path, required=True)
    esnet.add_argument("--output", type=Path, default=data_root()/"processed"/"esnet_r-nova-fd-01_delta.csv")
    esnet.add_argument("--scenario", default="r-nova-fd-01")
    esnet.add_argument("--signals", nargs="+", default=list(ESNET_SIGNALS))
    esnet.set_defaults(handler=_cmd_prepare_esnet)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--data", type=Path, required=True)
    inspect.add_argument("--top", type=int, default=30)
    inspect.add_argument("--prominence-fraction", type=float, default=.10)
    inspect.add_argument("--prominence-window", type=int, default=31)
    inspect.add_argument("--train-fraction", type=float, default=.7)
    inspect.add_argument("--event-distance", type=int, default=3)
    inspect.set_defaults(handler=_cmd_inspect)

    analyze = sub.add_parser("correlate", help="measure Pearson lead correlation versus horizon")
    analyze.add_argument("--data", type=Path, required=True)
    analyze.add_argument("--series-id", required=True)
    analyze.add_argument("--horizons", nargs="+", type=int, default=[96,192,336,720])
    analyze.add_argument("--signals", nargs="+", default=None)
    analyze.add_argument("--train-fraction", type=float, default=.7)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(handler=_cmd_correlate)

    precursor = sub.add_parser(
        "event-precursors",
        help="measure event-specific precursor structure using only the training prefix",
    )
    precursor.add_argument("--data", type=Path, required=True)
    precursor.add_argument("--series-id", required=True)
    precursor.add_argument("--horizons", nargs="+", type=int, default=[96,192,336,720])
    precursor.add_argument("--signals", nargs="+", default=None)
    precursor.add_argument("--input-size", type=int, default=336)
    precursor.add_argument("--train-fraction", type=float, default=.7)
    precursor.add_argument("--analysis-val-fraction", type=float, default=.3)
    precursor.add_argument("--prominence-fraction", type=float, default=.10)
    precursor.add_argument("--prominence-window", type=int, default=31)
    precursor.add_argument("--event-distance", type=int, default=3)
    precursor.add_argument("--event-tolerance", type=int, default=3)
    precursor.add_argument("--output", type=Path, required=True)
    precursor.set_defaults(handler=_cmd_event_precursors)

    sensitivity = sub.add_parser(
        "event-sensitivity",
        help="re-evaluate completed forecasts across event-definition settings",
    )
    sensitivity.add_argument("--results-root", type=Path, required=True)
    sensitivity.add_argument("--prominence-fractions", nargs="+", type=float, default=[.05,.10,.20])
    sensitivity.add_argument("--event-tolerances", nargs="+", type=int, default=[1,3,5])
    sensitivity.add_argument("--output", type=Path, required=True)
    sensitivity.set_defaults(handler=_cmd_event_sensitivity)

    baselines = sub.add_parser(
        "trivial-baselines",
        help="evaluate persistence, recent-window replay, and event-rate random baselines",
    )
    baselines.add_argument("--data", type=Path, required=True)
    baselines.add_argument("--series-id", required=True)
    baselines.add_argument("--horizons", nargs="+", type=int, default=[96,192,336,720])
    baselines.add_argument("--input-size", type=int, default=336)
    baselines.add_argument("--train-fraction", type=float, default=.7)
    baselines.add_argument("--val-fraction", type=float, default=.1)
    baselines.add_argument("--prominence-fraction", type=float, default=.10)
    baselines.add_argument("--prominence-window", type=int, default=31)
    baselines.add_argument("--event-distance", type=int, default=3)
    baselines.add_argument("--event-tolerance", type=int, default=3)
    baselines.add_argument("--random-trials", type=int, default=1000)
    baselines.add_argument("--random-seed", type=int, default=1)
    baselines.add_argument("--output", type=Path, required=True)
    baselines.set_defaults(handler=_cmd_trivial_baselines)

    manifest = sub.add_parser(
        "trace-manifest",
        help="select longest eligible traces for the breadth study without using event or forecast outcomes",
    )
    manifest.add_argument("--data", type=Path, required=True)
    manifest.add_argument("--target-signal", required=True)
    manifest.add_argument("--n-traces", type=int, default=3)
    manifest.add_argument("--mode", choices=["univariate","multivariate"], default="univariate")
    manifest.add_argument("--signals", nargs="+", default=None)
    manifest.add_argument("--input-size", type=int, default=336)
    manifest.add_argument("--max-horizon", type=int, default=720)
    manifest.add_argument("--train-fraction", type=float, default=.7)
    manifest.add_argument("--val-fraction", type=float, default=.1)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(handler=_cmd_trace_manifest)

    run = sub.add_parser("run")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--series-id", required=True)
    run.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    run.add_argument("--loss", choices=["mse","ep","mep"], required=True)
    run.add_argument("--horizon", type=int, default=96)
    run.add_argument("--input-size", type=int, default=336)
    run.add_argument("--max-steps", type=int, default=1000)
    run.add_argument("--tuning-max-steps", type=int, default=100)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--prominence-fraction", type=float, default=.10)
    run.add_argument("--prominence-window", type=int, default=31)
    run.add_argument("--event-distance", type=int, default=3)
    run.add_argument("--event-tolerance", type=int, default=3)
    run.add_argument("--ep-quantile", type=float, default=.90)
    run.add_argument("--mep-low-quantile", type=float, default=.10)
    run.add_argument("--mep-high-quantile", type=float, default=.90)
    run.add_argument("--ep-under-weight", type=float, default=2.0)
    run.add_argument("--ep-over-weight", type=float, default=1.0)
    run.add_argument("--ep-quantile-grid", nargs="+", type=float, default=[.90,.95])
    run.add_argument("--mep-low-quantile-grid", nargs="+", type=float, default=[.05,.10])
    run.add_argument("--mep-high-quantile-grid", nargs="+", type=float, default=[.90,.95])
    run.add_argument("--ep-under-weight-grid", nargs="+", type=float, default=[1.0,2.0])
    run.add_argument("--ep-over-weight-grid", nargs="+", type=float, default=[1.0,2.0])
    run.add_argument("--tuning-metric", choices=["event_macro_f1","mse"], default="event_macro_f1")
    run.add_argument("--multivariate-signals", nargs="+", default=None)
    run.add_argument("--train-fraction", type=float, default=.7)
    run.add_argument("--val-fraction", type=float, default=.1)
    run.add_argument("--output-dir", type=Path, default=None)
    run.set_defaults(handler=_cmd_run)
    return parser


def _cmd_paths(args):
    del args
    print(f"data_root={data_root()}")
    return 0


def _cmd_prepare_telecom(args):
    out=prepare_telecomts(args.root,args.output,scope=args.scope,kpis=tuple(args.kpis),resample_ms=args.resample_ms)
    print(f"wrote {len(out.frame):,} rows to {args.output}")
    return 0


def _cmd_prepare_mit(args):
    output=args.output or data_root()/"processed"/f"mit_supercloud_gpu_{args.metric}.csv"
    out=prepare_mit_supercloud_gpu(args.root,output,metric=args.metric,resample_ms=args.resample_ms)
    print(f"wrote {len(out.frame):,} rows to {output}")
    return 0


def _cmd_prepare_esnet(args):
    out=prepare_esnet(args.raw,args.output,signals=tuple(args.signals),scenario=args.scenario)
    print(f"wrote {len(out.frame):,} rows to {args.output}")
    return 0


def _cmd_inspect(args):
    frame=load_prepared(args.data)
    summary=summarize_series(frame,train_fraction=args.train_fraction,prominence_fraction=args.prominence_fraction,
        prominence_window=args.prominence_window,event_distance=args.event_distance)
    summary=summary.sort_values(["events_per_1000","n"],ascending=[False,False])
    print(summary.head(args.top).to_string(index=False))
    return 0


def _cmd_correlate(args):
    frame=load_prepared(args.data)
    out=write_correlation_analysis(
        frame, args.series_id, tuple(args.horizons), args.output,
        signals=tuple(args.signals or ()), train_fraction=args.train_fraction,
    )
    print(out)
    return 0


def _cmd_event_precursors(args):
    frame = load_prepared(args.data)
    out = write_event_precursor_analysis(
        frame,
        args.series_id,
        tuple(args.horizons),
        args.output,
        signals=tuple(args.signals or ()),
        input_size=args.input_size,
        train_fraction=args.train_fraction,
        analysis_val_fraction=args.analysis_val_fraction,
        prominence_fraction=args.prominence_fraction,
        prominence_window=args.prominence_window,
        event_distance=args.event_distance,
        event_tolerance=args.event_tolerance,
    )
    print(out)
    return 0


def _cmd_trivial_baselines(args):
    frame = load_prepared(args.data)
    out = write_trivial_baselines(
        frame,
        args.series_id,
        tuple(args.horizons),
        args.output,
        input_size=args.input_size,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        prominence_fraction=args.prominence_fraction,
        prominence_window=args.prominence_window,
        event_distance=args.event_distance,
        event_tolerance=args.event_tolerance,
        random_trials=args.random_trials,
        random_seed=args.random_seed,
    )
    print(out)
    return 0


def _cmd_trace_manifest(args):
    frame = load_prepared(args.data)
    out = build_trace_manifest(
        frame,
        args.data,
        args.target_signal,
        args.output,
        n_traces=args.n_traces,
        mode=args.mode,
        required_signals=tuple(args.signals or ()),
        input_size=args.input_size,
        max_horizon=args.max_horizon,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    print(out)
    return 0


def _cmd_event_sensitivity(args):
    out = write_results_sensitivity(
        args.results_root,
        args.output,
        prominence_fractions=tuple(args.prominence_fractions),
        event_tolerances=tuple(args.event_tolerances),
    )
    print(out)
    return 0


def _cmd_run(args):
    out=run_experiment(ExperimentConfig(
        data=args.data,series_id=args.series_id,model=args.model,loss=args.loss,horizon=args.horizon,
        input_size=args.input_size,max_steps=args.max_steps,tuning_max_steps=args.tuning_max_steps,
        learning_rate=args.learning_rate,seed=args.seed,
        prominence_fraction=args.prominence_fraction,prominence_window=args.prominence_window,
        event_distance=args.event_distance,event_tolerance=args.event_tolerance,
        ep_quantile=args.ep_quantile,mep_low_quantile=args.mep_low_quantile,mep_high_quantile=args.mep_high_quantile,
        ep_under_weight=args.ep_under_weight,ep_over_weight=args.ep_over_weight,
        ep_quantile_grid=tuple(args.ep_quantile_grid),
        mep_low_quantile_grid=tuple(args.mep_low_quantile_grid),
        mep_high_quantile_grid=tuple(args.mep_high_quantile_grid),
        ep_under_weight_grid=tuple(args.ep_under_weight_grid),
        ep_over_weight_grid=tuple(args.ep_over_weight_grid),tuning_metric=args.tuning_metric,
        train_fraction=args.train_fraction,val_fraction=args.val_fraction,
        multivariate_signals=tuple(args.multivariate_signals or ())), output_dir=args.output_dir)
    print(out)
    return 0


def main(argv=None):
    parser=_build_parser(); args=parser.parse_args(argv); return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
