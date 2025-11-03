import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data import data
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from data import X_train, X_val, y_class_val, y_class_train
from data import y_reg_val, y_reg_train
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
import mlflow.sklearn
import mlflow

from train_baselines import y_predic_class,y_predic_reg,best_class_model
plt.figure(figsize=(10,6))
sns.countplot(x=data["label"], order=data["label"].value_counts().index)
plt.title("Class Distribution (Crop Frequency)")
plt.ylabel("Count")
plt.xlabel("Crop Type")
plt.tight_layout()
#plt.show()


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


model = best_class_model

y_pred = model.predict(X_val)


cm = confusion_matrix(y_class_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix – Decision Tree Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()
