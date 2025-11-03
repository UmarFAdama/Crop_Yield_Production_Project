import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data import data
from data import X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
from sklearn.metrics import confusion_matrix
from src.train_baselines import best_reg_model
from train_baselines import best_class_model

#Plot 1 – Target distribution plot for classification (bar plot of class counts)
plt.figure(figsize=(10,6))
sns.countplot(x=data["label"], order=data["label"].value_counts().index)
plt.title("Class Distribution (Crop Frequency)")
plt.ylabel("Count")
plt.xlabel("Crop Type")
plt.tight_layout()
plt.show()

#Plot 2 – Correlation heatmap for key numeric features.
plt.figure(figsize=(10,6))
correlations = data.corr(numeric_only=True)
top_4 = correlations["yield"].drop("yield").abs().sort_values(ascending=False).head(4)
top_features = top_4.index.tolist()

features_to_plot = top_features + ["yield"]
plt.figure(figsize=(10,6))

top_correlations = data[features_to_plot].corr()
mask = np.triu(np.ones_like(top_correlations, dtype=bool))
sns.heatmap(top_correlations, cmap="coolwarm", annot=True)

plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Plot 3 – Confusion matrix for the best current classification baseline on the test set.
model = best_class_model
model.fit(X_train, y_class_train)

y_class_pred = model.predict(X_test)

cm = confusion_matrix(y_class_test, y_class_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix – Decision Tree Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

#Plot 4 – Residuals vs predicted for the best current regression baseline on the test set
best_reg_model.fit(X_train, y_reg_train)
y_reg_pred = best_reg_model.predict(X_test)
residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8,6))
plt.scatter(y_reg_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted – Best Regression Baseline')
plt.xlabel('Predicted Crop Yield (kg/ha)')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()
