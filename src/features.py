from sklearn.preprocessing import StandardScaler, LabelEncoder
from data import (
    X_train, X_val, X_test,
    y_class_train, y_class_val, y_class_test,
    y_reg_train, y_reg_val, y_reg_test,
)

# ===== Preprocessing for Neural Networks =====

# Scale numeric features (important for NNs)
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train)
X_val_nn = scaler.transform(X_val)
X_test_nn = scaler.transform(X_test)


# converts class labels to integers for NN classifier
le = LabelEncoder()
y_class_train_enc = le.fit_transform(y_class_train)
y_class_val_enc = le.transform(y_class_val)
y_class_test_enc = le.transform(y_class_test)

# Regression targets just as numpy arrays
y_reg_train_nn = y_reg_train.values
y_reg_val_nn = y_reg_val.values
y_reg_test_nn = y_reg_test.values


__all__ = [
    # NN-ready inputs
    "X_train_nn", "X_val_nn", "X_test_nn",
    "y_class_train_enc", "y_class_val_enc", "y_class_test_enc",
    "y_reg_train_nn", "y_reg_val_nn", "y_reg_test_nn",
    # objects might reuse (optional)
    "scaler", "le",
]
