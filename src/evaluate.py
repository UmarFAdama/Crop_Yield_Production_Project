import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data import data
from data import X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

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
plt.title('Residuals vs Predicted – Best Regression Baseline')
plt.xlabel('Predicted Crop Yield (kg/ha)')
plt.ylabel('Residuals (Actual − Predicted)')
plt.tight_layout()
plt.show()
