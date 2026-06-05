import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ==================================
# LOAD DATA
# ==================================

df = pd.read_csv(
    "outputs/emotion_features.csv"
)

X = df.drop(
    "emotion",
    axis=1
)

y = df["emotion"]

# ==================================
# ENCODE LABELS
# ==================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

y_categorical = to_categorical(
    y_encoded
)

# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ==================================
# FEATURE SCALING
# ==================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# ==================================
# BUILD MODEL
# ==================================

model = Sequential()

model.add(
    Dense(
        256,
        activation="relu",
        input_shape=(40,)
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        8,
        activation="softmax"
    )
)

# ==================================
# COMPILE MODEL
# ==================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==================================
# EARLY STOPPING
# ==================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==================================
# TRAIN MODEL
# ==================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=50,
    batch_size=32,
    callbacks=[
        early_stop
    ]
)

# ==================================
# SAVE MODEL
# ==================================

model.save(
    "models/emotion_model.h5"
)

print(
    "\nModel Saved Successfully"
)

# ==================================
# MODEL EVALUATION
# ==================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

# ==================================
# ACCURACY GRAPH
# ==================================

os.makedirs(
    "outputs",
    exist_ok=True
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Emotion Recognition Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/accuracy_plot.png"
)

plt.close()

# ==================================
# CONFUSION MATRIX
# ==================================

predictions = model.predict(
    X_test
)

predicted_labels = predictions.argmax(
    axis=1
)

actual_labels = y_test.argmax(
    axis=1
)

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

print(
    "Saved: outputs/accuracy_plot.png"
)

print(
    "Saved: outputs/confusion_matrix.png"
)