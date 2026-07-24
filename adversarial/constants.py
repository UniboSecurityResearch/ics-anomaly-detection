"""Model-family constants shared across the adversarial package."""

# Point-wise reconstruction models (input == target timestep).
POINT_MODELS = {"AE", "DNN"}

# Sequence prediction models (input = history window, target = future timestep).
SEQUENCE_MODELS = {"CNN", "GRU", "LSTM"}
