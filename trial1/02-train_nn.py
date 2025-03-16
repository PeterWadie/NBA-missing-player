import pandas as pd
import numpy as np
import pickle

# If using TensorFlow Keras:
import tensorflow as tf
from tensorflow import keras

# Or if you prefer PyTorch, you would import torch, nn, etc.

from sklearn.model_selection import train_test_split

# ----------------------------
# 1) Load your LabelEncoders
# ----------------------------
with open("player_encoder.pkl", "rb") as f:
    player_encoder = pickle.load(f)

with open("team_encoder.pkl", "rb") as f:
    team_encoder = pickle.load(f)

# ----------------------------
# 2) Define feature columns
#    (same as your LightGBM code)
# ----------------------------
feature_cols = (
    [f"home_{i}" for i in range(4)]
    + [f"away_{i}" for i in range(5)]
    + ["home_team", "away_team", "starting_min"]
)

# ----------------------------
# 3) Load Encoded Dataset
# ----------------------------
df_encoded = pd.read_csv("01-encoded_data.csv")
X = df_encoded[feature_cols].values
y = df_encoded["removed_player"].values  # integer-encoded labels

# Number of unique classes == total distinct players in player_encoder
num_classes = len(player_encoder.classes_)

# ----------------------------
# 4) Train/validation split
# ----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5) Define a Neural Network
# ----------------------------
model = keras.Sequential()

# Input layer matches your input dimension
input_dim = X_train.shape[1]  # typically 4+5+2+1 = 12
model.add(keras.layers.Input(shape=(input_dim,)))

# Hidden layers (tweak sizes/activations as needed)
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.2))         # dropout can help generalize
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.2))

# Output layer: num_classes with softmax
# This is a multiclass classification
model.add(keras.layers.Dense(num_classes, activation="softmax"))

# ----------------------------
# 6) Compile the model
# ----------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",  # y is integer-encoded
    metrics=["accuracy"]
)

# ----------------------------
# 7) Train the model
# ----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=10,            # Increase epochs as needed
    batch_size=1024,      # Adjust based on data size/your RAM
    validation_data=(X_valid, y_valid),
    verbose=1
)

# ----------------------------
# 8) Save the model
# ----------------------------
model.save("nn_model.h5")
print("Neural network model saved as nn_model.h5")
