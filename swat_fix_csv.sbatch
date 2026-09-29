#!/bin/bash
#SBATCH --job-name=swat_csv_fix
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --partition=a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH -o slurm.swat_csv_fix.%j.out
#SBATCH -e slurm.swat_csv_fix.%j.err

# No model training, no dataset relabeling unless BOTH reference hashes match.
# Raw input CSVs are never replaced. Existing processed CSVs are not overwritten.
set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
export APPTAINER_CACHEDIR="$REPO"
cd "$REPO"

test -r "$IMAGE"
test -r "$REPO/data/SWAT/hash_SWaT.py"
test -r "$REPO/data/SWAT/process_SWaT.py"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)_$$}"
CANDIDATE="$REPO/data/SWAT/csv_normalized_$RUN_ID"
echo "Candidate directory: $CANDIDATE"

if apptainer exec \
    --bind "$REPO:$REPO" \
    --pwd "$REPO" \
    "$IMAGE" python3 -u - \
    --source-dir "$REPO/data/SWAT" --output-dir "$CANDIDATE" <<'PY_NORMALIZE'
#!/usr/bin/env python3
"""Normalize the user's semicolon/decimal-comma SWaT exports without relabeling.

Inputs are never modified. Candidate CSVs are written to a new directory.
Exit 0: both repository hashes match; 2: valid conversion but hash mismatch;
1: structural or conversion error. Requires pandas and numpy.

Reference hash algorithm:
https://github.com/pwwl/ics-anomaly-detection/blob/main/data/SWAT/hash_SWaT.py
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

SPECS = (
    ("SWaT_Dataset_Normal_v1.csv", 1,
     "3710e89098e099fc22ba25457a4c45428865fe94d9a868c937ed0feb232c8916"),
    ("SWaT_Dataset_Attack_v0.csv", 0,
     "59556a9c0d03739e7644c1b669d5d16137e65b7b3df723a406956ec062c743a9"),
)
NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"


def repository_hash(path):
    """Reproduce the reference CSV-cell hash, not a raw-file SHA256."""
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            text = "".join(row)
            digest.update((text or "EMPTY_ROW_SEPARATOR").encode("utf-8"))
    return digest.hexdigest()


def convert(source, destination, header_index, chunk_size=20000):
    """Keep cell strings, replacing decimal commas only in feature columns."""
    count = 0
    labels = Counter()
    changed = 0
    first_timestamp = None
    last_timestamp = None
    with source.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.reader(src, delimiter=";", strict=True)
        prefix = list(islice(reader, header_index + 1))
        if len(prefix) != header_index + 1:
            raise ValueError("{}: missing header".format(source.name))
        header = prefix[-1]
        names = [name.replace(" ", "").strip() for name in header]
        if (len(names) < 3 or names[0] != "Timestamp"
                or names[-1] != "Normal/Attack"):
            raise ValueError(
                "{}: expected semicolon-delimited header at CSV record {}".format(
                    source.name, header_index + 1))
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("{}: empty or duplicate column names".format(source.name))
        width = len(header)
        with destination.open("x", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst, delimiter=",", lineterminator="\n")
            # Keep the normal file's title row and all original header spelling.
            writer.writerows(prefix)
            while True:
                rows = list(islice(reader, chunk_size))
                if not rows:
                    break
                for offset, row in enumerate(rows):
                    if len(row) != width:
                        raise ValueError(
                            "{}: CSV record {} has {} fields, expected {}; "
                            "no rows will be silently dropped".format(
                                source.name, header_index + 2 + count + offset,
                                len(row), width))
                frame = pd.DataFrame(rows, dtype=str)
                for col in (0, width - 1):
                    if frame[col].str.strip().eq("").any():
                        raise ValueError("{}: empty timestamp or label".format(source.name))
                for col in range(1, width - 1):
                    values = frame[col].str.replace(",", ".", regex=False)
                    valid = values.str.strip().str.fullmatch(NUMBER, na=False)
                    if not valid.all():
                        offset = int(np.flatnonzero(~valid.to_numpy())[0])
                        raise ValueError(
                            "{}: nonnumeric/ambiguous value {!r} in {}, CSV record {}. "
                            "Missing values and thousands separators require review.".format(
                                source.name, frame.iloc[offset, col], names[col],
                                header_index + 2 + count + offset))
                    numeric = pd.to_numeric(values.str.strip(), errors="raise")
                    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                        raise ValueError("{}: nonfinite values in {}".format(source.name, names[col]))
                    changed += int(frame[col].ne(values).sum())
                    # Write the normalized strings, NOT the temporary floats.
                    frame[col] = values
                if header_index == 1 and not frame[width - 1].str.strip().eq("Normal").all():
                    raise ValueError("{}: unexpected non-Normal training label".format(source.name))
                if first_timestamp is None:
                    first_timestamp = frame.iloc[0, 0]
                last_timestamp = frame.iloc[-1, 0]
                labels.update(frame[width - 1].tolist())
                writer.writerows(frame.itertuples(index=False, name=None))
                count += len(frame)
    if count == 0:
        raise ValueError("{}: no data records".format(source.name))
    return {
        "file": source.name,
        "header_record": header_index + 1,
        "data_rows": count,
        "columns": width,
        "feature_count": width - 2,
        "feature_names": names[1:-1],
        "decimal_comma_cells_converted": changed,
        "first_timestamp_unchanged": first_timestamp,
        "last_timestamp_unchanged": last_timestamp,
        "label_counts_unchanged": dict(labels),
        "normalized_csv_cell_hash": repository_hash(destination),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/SWAT"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    # A fresh destination prevents accidental reuse of stale candidates.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print("Candidate directory: {}".format(args.output_dir.resolve()), flush=True)
    reports = []
    for filename, header_index, expected in SPECS:
        report = convert(args.source_dir / filename,
                         args.output_dir / filename, header_index)
        report["expected_csv_cell_hash"] = expected
        report["hash_matches"] = report["normalized_csv_cell_hash"] == expected
        reports.append(report)
        print(json.dumps({key: value for key, value in report.items()
                          if key != "feature_names"}, indent=2), flush=True)
    if reports[0]["feature_names"] != reports[1]["feature_names"]:
        raise ValueError("Normal/attack feature names or column order differ")
    with (args.output_dir / "conversion_report.json").open("x", encoding="utf-8") as out:
        json.dump(reports, out, indent=2)
    print("CSV conversion and structural checks passed. Original files are unchanged.", flush=True)
    if not all(report["hash_matches"] for report in reports):
        print("HASH MISMATCH REMAINS: no relabeling or training was performed.", flush=True)
        print("Inspect conversion_report.json before using row-index attack corrections.", flush=True)
        return 2
    print("Both reference hashes match.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, UnicodeError, csv.Error) as exc:
        print("STOP: {}".format(exc), file=sys.stderr, flush=True)
        print("Original CSVs are unchanged. Do not use partial candidate files.", file=sys.stderr)
        sys.exit(1)

PY_NORMALIZE
then
    echo "Normalized candidates match both reference hashes."
else
    status=$?
    echo "STOP: conversion/hash gate returned $status. No relabeling or training was run."
    echo "Original files remain unchanged. Inspect this log and the candidate report."
    exit "$status"
fi

# Independently run the repository's unchanged guard in the candidate directory.
apptainer exec --bind "$REPO:$REPO" --pwd "$CANDIDATE" \
    "$IMAGE" python3 -u "$REPO/data/SWAT/hash_SWaT.py"

# This uses the repository's corrected-label protocol, NOT unmodified source labels.
apptainer exec --bind "$REPO:$REPO" --pwd "$CANDIDATE" \
    "$IMAGE" python3 -u "$REPO/data/SWAT/process_SWaT.py"

# Validate the processed candidates before exposing them to the training loader.
apptainer exec --bind "$REPO:$REPO" --pwd "$REPO" \
    "$IMAGE" python3 -u - "$CANDIDATE" "$REPO/data/SWAT" <<'PY_PUBLISH'
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

candidate, target = map(Path, sys.argv[1:])
names = ("SWATv0_train.csv", "SWATv0_test.csv")
features = None
for name in names:
    first_chunk = True
    rows = 0
    label_values = set()
    for frame in pd.read_csv(candidate / name, chunksize=20000):
        if not {"Timestamp", "Normal/Attack"}.issubset(frame.columns):
            raise ValueError("Missing metadata columns: " + name)
        cols = [c for c in frame.columns if c not in ("Timestamp", "Normal/Attack")]
        if features is None:
            features = cols
        if not features or cols != features:
            raise ValueError("Inconsistent feature names/order: " + name)
        values = frame[cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite features: " + name)
        labels = pd.to_numeric(frame["Normal/Attack"], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError("Nonbinary labels: " + name)
        label_values.update(labels.unique().tolist())
        rows += len(frame)
    if not rows:
        raise ValueError("Empty processed data: " + name)
    expected_labels = {0} if name == names[0] else {0, 1}
    if label_values != expected_labels:
        raise ValueError("Unexpected label classes: " + name)
    print("{}: {:,} rows, {} features; structural checks passed".format(name, rows, len(features)))

# Same-filesystem hard links publish complete files without overwriting anything.
# The candidate directory and destination are inside the same data/SWAT directory.
if any(os.path.lexists(str(target / name)) for name in names):
    print("Processed files already exist in data/SWAT; NOT overwriting them.")
    print("Validated candidates remain in: {}".format(candidate))
    sys.exit(3)
created = []
try:
    for name in names:
        os.link(str(candidate / name), str(target / name))
        created.append(target / name)
except OSError:
    for path in created:
        path.unlink()
    raise
print("Published SWATv0_train.csv and SWATv0_test.csv in data/SWAT.")
print("Raw source CSVs remain unchanged. No model training was performed.")
PY_PUBLISH

echo "SWaT CSV preparation completed successfully."
