"""Pure NumPy helpers: frozen eligibility and metrics with explicit denominators."""
import numpy as np


def freeze_eligible(target_indices, attack_labels, clean_detected):
    targets = np.asarray(target_indices)
    if clean_detected is None:
        raise ValueError("Clean window detection is required")
    clean = np.asarray(clean_detected, dtype=bool)
    if clean.shape != targets.shape or not np.all(attack_labels[targets]):
        raise ValueError("Evasion protocol requires only attack-labelled evaluation targets")
    eligible = clean.copy()
    if not np.any(eligible):
        raise ValueError("No clean-detected eligible targets; ASR is undefined")
    eligible.setflags(write=False)
    return eligible


def iteration_metrics(eligible, detected):
    detected = np.asarray(detected, dtype=bool)
    if eligible.shape != detected.shape:
        raise ValueError("Detection/eligibility shapes differ")
    count = int(np.sum(eligible))
    if count == 0:
        raise ValueError("ASR denominator is zero")
    evaded = int(np.sum(eligible & ~detected))
    return dict(evasions=evaded, asr=evaded / count,
                detection_rate=float(np.mean(detected)),
                detected_targets=int(np.sum(detected)),
                eligible_targets=count, targets=int(detected.size))
