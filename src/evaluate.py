from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from data import X_train, X_val, y_class_val, y_class_train
from data import y_reg_val, y_reg_train
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
import mlflow.sklearn
import mlflow
import matplotlib.pyplot as plt

from train_baselines import y_predic_class,y_predic_reg,best_class_model

model = best_class_model

y_pred = model.predict(X_val)


cm = confusion_matrix(y_class_val, y_pred)

# --- Plot it ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix – Decision Tree Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()