#!/usr/bin/env python3
"""Redo the threshold/window choice of trained SWAT-CLEAN detectors on a wider
grid, without retraining and without changing any model file.

For each model folder (models/swatchrono_<model>_s<seed>) this:
  1. loads the saved model and rebuilds the training data split exactly as
     main_train.py does (chronological validation; last 20% of the training file);
  2. calls main_train.hyperparameter_search, the function used in training, with
     the training grid. It must give back the theta, window and final F1 stored in
     the model .json ("reproduces stored: YES"); otherwise the script exits with
     code 3 and nothing else is trusted;
  3. calls it again with a wider grid: the default grid of the original
     repository (main_train.py, --detect_params_percentile / _windows) plus the
     values of the training grid. Same metric (F1) and same FPR cap (5%) on the
     tuning part of the test file.

It writes only outputs/retune_v1/<run>.json and the feature-error reports of the
two searches in outputs/retune_v1/<run>/; the model folders and the scaler file
are never written (the training data are loaded untransformed and scaled in
memory with the saved scaler, which is checked against a fresh fit).

    python3 retune_thresholds.py models/swatchrono_ae_s7 [more model folders]
    python3 retune_thresholds.py --summary      # table of all results so far
"""
import argparse
import glob
import json
import os
import pickle
import sys

OUT_ROOT = "outputs/retune_v1"
TRAIN_PERCENTILES = [0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
TRAIN_WINDOWS = [1, 3, 5, 10, 20, 30]
REPO_PERCENTILES = [0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995,
                    0.996, 0.997, 0.998, 0.999, 0.9995, 0.99995]
REPO_WINDOWS = [1, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
WIDE_PERCENTILES = sorted(set(REPO_PERCENTILES) | set(TRAIN_PERCENTILES))
WIDE_WINDOWS = sorted(set(REPO_WINDOWS) | set(TRAIN_WINDOWS))
FIELDS = ["threshold_percentile", "threshold_window", "threshold_theta",
          "validation_selection_value", "validation_fpr", "final_f1", "final_recall",
          "final_precision", "final_fpr"]


def search(main_train, utils, detector, model_type, xfull, xtest, ytest, dataset,
           percentiles, windows, metric, max_fpr, run_label, sensor_cols):
    """One call of the training's own search; returns the fields it stores."""
    config = {"name": detector.params.get("_retune_name", "model"),
              "grid_search": {"percentile": percentiles, "window": windows,
                              "metrics": [metric], "max-fpr": max_fpr,
                              "pr-plot": False, "detection-plots": False,
                              "save-metric-info": False, "save-theta": False}}
    if model_type == "AE":
        from sklearn.model_selection import train_test_split
        _, xval, _, _ = train_test_split(xfull, xfull, test_size=0.2,
                                         random_state=42, shuffle=False)
        main_train.hyperparameter_search(detector, model_type, config, xval, xtest, ytest,
                                         dataset, test_split=0.7, run_name=run_label,
                                         verbose=0, feature_names=sensor_cols)
    else:
        history = int(detector.params.get("history", 0))
        _, val_idxs = utils.train_val_history_idx_split(xfull, history, shuffle=False)
        main_train.hyperparameter_search(detector, model_type, config, xfull, xtest, ytest,
                                         dataset, val_idxs=val_idxs, test_split=0.7,
                                         run_name=run_label, verbose=0,
                                         feature_names=sensor_cols)
    return {k: detector.params.get(k) for k in FIELDS}


def retune(model_dir, dataset):
    import numpy as np
    import main_train
    import utils
    from data_loader import load_train_data, load_test_data
    from sklearn.preprocessing import StandardScaler

    run = os.path.basename(os.path.normpath(model_dir))
    jsons = sorted(glob.glob(os.path.join(model_dir, "*.json")))
    if len(jsons) != 1:
        raise SystemExit(f"{model_dir}: expected one .json, found {len(jsons)}")
    name = os.path.basename(jsons[0])[:-5]
    model_type = name.split("-")[0]
    stored = json.load(open(jsons[0]))

    raw, sensor_cols = load_train_data(dataset, no_transform=True)  # writes nothing
    scaler = pickle.load(open(f"models/{dataset}_scaler.pkl", "rb"))
    fresh = StandardScaler().fit(raw)
    scaler_ok = bool(np.allclose(fresh.mean_, scaler.mean_) and np.allclose(fresh.scale_, scaler.scale_))
    if not scaler_ok:
        raise SystemExit("saved scaler differs from a fresh fit on the training data")
    xfull = scaler.transform(raw)
    xtest, ytest, _ = load_test_data(dataset)

    metric_list = stored.get("threshold_selection_metric") or ["F1"]
    metric = metric_list[0] if isinstance(metric_list, list) else metric_list
    max_fpr = stored.get("threshold_max_fpr", 0.05)

    os.makedirs(os.path.join(OUT_ROOT, run, "train_grid"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, run, "wide_grid"), exist_ok=True)
    detector = main_train.load_saved_model(model_type, run, name)
    detector.params["_retune_name"] = name
    print(f"=== {run} ({name}) metric={metric} max_fpr={max_fpr}", flush=True)

    again = search(main_train, utils, detector, model_type, xfull, xtest, ytest, dataset,
                   TRAIN_PERCENTILES, TRAIN_WINDOWS, metric, max_fpr,
                   f"retune_v1/{run}/train_grid", sensor_cols)
    reproduced = (
        again["threshold_window"] == stored.get("best_window")
        and np.isclose(again["threshold_percentile"], stored.get("threshold_percentile", np.nan))
        and np.isclose(again["threshold_theta"], stored.get("best_theta"), rtol=1e-4, atol=0)
        and abs(again["final_f1"] - stored.get("final_f1", np.nan)) <= 1e-3)
    print(f"stored : percentile={stored.get('threshold_percentile')} window={stored.get('best_window')} "
          f"theta={stored.get('best_theta'):.6g} F1={stored.get('final_f1')}")
    print(f"again  : percentile={again['threshold_percentile']} window={again['threshold_window']} "
          f"theta={again['threshold_theta']:.6g} F1={again['final_f1']:.6f}")
    print(f"reproduces stored: {'YES' if reproduced else 'NO'}", flush=True)

    wide = None
    if reproduced:
        wide = search(main_train, utils, detector, model_type, xfull, xtest, ytest, dataset,
                      WIDE_PERCENTILES, WIDE_WINDOWS, metric, max_fpr,
                      f"retune_v1/{run}/wide_grid", sensor_cols)
        print(f"wide   : percentile={wide['threshold_percentile']} window={wide['threshold_window']} "
              f"theta={wide['threshold_theta']:.6g} F1={wide['final_f1']:.6f} "
              f"recall={wide['final_recall']:.4f} FPR={wide['final_fpr']:.4f}", flush=True)

    result = dict(run=run, model=name, model_type=model_type, metric=metric, max_fpr=max_fpr,
                  stored={k: stored.get(k) for k in FIELDS + ["best_theta", "best_window"]},
                  train_grid=dict(percentiles=TRAIN_PERCENTILES, windows=TRAIN_WINDOWS, **again),
                  reproduces_stored=bool(reproduced),
                  wide_grid=(dict(percentiles=WIDE_PERCENTILES, windows=WIDE_WINDOWS, **wide)
                             if wide else None))
    with open(os.path.join(OUT_ROOT, f"{run}.json"), "w") as fd:
        json.dump(result, fd, indent=1, default=float)
    return reproduced


def summary():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT_ROOT, "*.json")))]
    print("run                    repro  train: pct   w    F1     | wide: pct     w    F1     recall  FPR    edge")
    for r in rows:
        t, w = r["train_grid"], r["wide_grid"]
        line = "%-22s %-5s  %-8s %3s %.4f |" % (r["run"], "YES" if r["reproduces_stored"] else "NO",
                                              t["threshold_percentile"], t["threshold_window"], t["final_f1"])
        if w:
            edge = []
            if w["threshold_percentile"] == min(WIDE_PERCENTILES):
                edge.append("pct-min")
            if w["threshold_window"] == max(WIDE_WINDOWS):
                edge.append("w-max")
            line += " %-8s %3s %.4f  %.4f  %.4f %s" % (
                w["threshold_percentile"], w["threshold_window"], w["final_f1"],
                w["final_recall"], w["final_fpr"], " ".join(edge))
        print(line)
    print(f"{len(rows)} models; wide grid: percentiles {WIDE_PERCENTILES}, windows {WIDE_WINDOWS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dirs", nargs="*")
    ap.add_argument("--dataset", default="SWAT-CLEAN")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        return 0
    if not args.model_dirs:
        ap.error("give model folders or --summary")
    ok = [retune(d, args.dataset) for d in args.model_dirs]
    if not all(ok):
        print("ERROR: at least one model did not reproduce its stored values", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
