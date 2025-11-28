import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
# NN-ready data from features.py
from features import (
    X_train_nn, X_val_nn, X_test_nn,
    y_class_train_enc, y_class_val_enc, y_class_test_enc,
    y_reg_train_nn, y_reg_val_nn, y_reg_test_nn
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_classification_nn(input_dim, num_classes):
    """
    Simple feedforward neural network for multi-class crop classification.
    input_dim: number of features (7 in this project).
    num_classes: number of crop labels (22).
    """
    model = Sequential()
    # Hidden layer 1
    model.add(Dense(64, activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(0.2))  # regularization

    # Hidden layer 2
    model.add(Dense(32, activation="relu"))

    # Output layer: one neuron per class, softmax for probabilities
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_regression_nn(input_dim):
    """
    Simple feedforward neural network for regression on synthetic yield.
    """
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))  # single numeric output

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",  # mean squared error
    )
    return model

if __name__ == "__main__":
    # ----- Build and train classification NN -----
    print("\nTraining classification neural network...")
    num_features = X_train_nn.shape[1]
    num_classes = len(np.unique(y_class_train_enc))

    clf_nn = build_classification_nn(num_features, num_classes)

    history = clf_nn.fit(
        X_train_nn,
        y_class_train_enc,
        validation_data=(X_val_nn, y_class_val_enc),
        epochs=40,
        batch_size=32,
        verbose=1,
    )

    # ----- Compute validation & test metrics (for Table 1) -----
    # Validation predictions
    y_val_pred_probs = clf_nn.predict(X_val_nn)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)

    # Test predictions
    y_test_pred_probs = clf_nn.predict(X_test_nn)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)

    # Metrics
    acc_val = accuracy_score(y_class_val_enc, y_val_pred)
    f1_val = f1_score(y_class_val_enc, y_val_pred, average="weighted")
    acc_test = accuracy_score(y_class_test_enc, y_test_pred)
    f1_test = f1_score(y_class_test_enc, y_test_pred, average="weighted")

    print(f"NN Classifier – Validation: Accuracy = {acc_val:.3f}, F1 = {f1_val:.3f}")
    print(f"NN Classifier – Test:       Accuracy = {acc_test:.3f}, F1 = {f1_test:.3f}")


    # ----- Plot 1: Learning curve for classification NN -----
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification NN – Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===== REGRESSION NN =====
    print("\nTraining regression neural network...")

    reg_nn = build_regression_nn(num_features)

    history_reg = reg_nn.fit(
        X_train_nn,
        y_reg_train_nn,
        validation_data=(X_val_nn, y_reg_val_nn),
        epochs=40,
        batch_size=32,
        verbose=1,
    )

    # Validation and test predictions
    y_val_pred_reg = reg_nn.predict(X_val_nn).ravel()
    y_test_pred_reg = reg_nn.predict(X_test_nn).ravel()

    # Metrics for Table 2
    mae_val = mean_absolute_error(y_reg_val_nn, y_val_pred_reg)
    rmse_val = np.sqrt(mean_squared_error(y_reg_val_nn, y_val_pred_reg))
    mae_test = mean_absolute_error(y_reg_test_nn, y_test_pred_reg)
    rmse_test = np.sqrt(mean_squared_error(y_reg_test_nn, y_test_pred_reg))

    print(f"NN Regressor – Validation: MAE = {mae_val:.3f}, RMSE = {rmse_val:.3f}")
    print(f"NN Regressor – Test:       MAE = {mae_test:.3f}, RMSE = {rmse_test:.3f}")

    # ----- Plot 2: learning curve for regression NN -----
    plt.figure(figsize=(8, 6))
    plt.plot(history_reg.history["loss"], label="Train Loss")
    plt.plot(history_reg.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Regression NN – Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

