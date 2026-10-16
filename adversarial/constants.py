"""Model-family constants shared across the adversarial package."""

# Point-wise reconstruction models (input == target timestep).
POINT_MODELS = {"AE"}

# Sequence prediction models (input = history window, target = future timestep).
SEQUENCE_MODELS = {"DNN", "CNN", "GRU", "LSTM", "LIN"}

# One-step models: the Keras model takes a single row X[t-1] (shape n x features)
# and predicts X[t]. They run as sequence models with history 1 and target offset
# 0; the (n, 1, features) history window is reshaped to (n, features) before the
# model call. Training uses the same pairing (main_train.py: X[t] -> X[t+1]).
ONE_STEP_MODELS = {"LIN"}
