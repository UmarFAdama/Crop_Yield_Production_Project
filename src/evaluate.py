import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data import data
from data import X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

#Plot 1 – Target distribution plot for classification (bar plot of class counts)
plt.figure(figsize=(10,6))
sns.countplot(x=data["label"], order=data["label"].value_counts().index)
plt.title("Class Distribution (Crop Frequency)")
plt.xticks(rotation=45, ha="right")
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
nb = GaussianNB()
# Train the best model
nb.fit(X_train, y_class_train)

#predict on test set
y_class_prediction = nb.predict(X_test)

#generate confusion matrix
cm = confusion_matrix(y_class_test, y_class_prediction, labels=nb.classes_)

#plot confusion matrix
plt.figure(figsize=(14, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.title("Confusion Matrix – Naive Bayes")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

#Plot 4 – Residuals vs predicted for the best current regression baseline on the test set
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_reg_train)

#predict on test set
y_reg_prediction = lin_reg.predict(X_test)

#residuals = actual-predicted
residuals = y_reg_test - y_reg_prediction

plt.figure(figsize=(8,6))
plt.scatter(y_reg_prediction, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals vs Predicted – Best Regression Baseline: Linear Regression')
plt.xlabel('Predicted Crop Yield (kg/ha)')
plt.ylabel('Residuals (Actual − Predicted)')
plt.tight_layout()
plt.show()

#plot 5
## 1. Train the best regression model (Linear Regression) on the training set
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_reg_train)

# 2. Compute permutation importance on the test set
result = permutation_importance(
    lin_reg,
    X_test,
    y_reg_test,
    n_repeats=30,
    random_state=42,
    scoring="neg_mean_squared_error",  # lower MSE = better
)

importances = result.importances_mean
feature_names = X_train.columns.to_numpy()

# 3. Sort features by importance (descending)
idx = np.argsort(importances)[::-1]
sorted_importances = importances[idx]
sorted_features = feature_names[idx]

# 4. Plot – bar chart
plt.figure(figsize=(8, 6))
plt.bar(range(len(sorted_features)), sorted_importances)
plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
plt.ylabel("Mean decrease in score\n(permutation importance)")
plt.title("Permutation Feature Importance – Linear Regression (Yield Regression)")
plt.tight_layout()
plt.show()
