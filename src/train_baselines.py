from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from data import X_train, X_val, y_class_val, y_class_train
from data import y_reg_val, y_reg_train
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
import mlflow.sklearn
import mlflow
model_class = {
    "Naive Bays": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)}



for name, model in model_class.items():
    with mlflow.start_run(run_name=f"Classification - {name}"):
        model.fit(X_train, y_class_train)
        y_predic_class = model.predict(X_val)
        acc = accuracy_score(y_class_val, y_predic_class)
        f1 = f1_score(y_class_val, y_predic_class, average='weighted')

        print(f"{name} :: Accuracy: {acc:.3f}, F1: {f1:.3f}")

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, name =f"{name}_classifier")

#Best classifier: Naive Bayes

model_reg = {
    "Linear Regression":LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
}

for name, model in model_reg.items():
    with mlflow.start_run(run_name=f"Regression - {name}"):
        model.fit(X_train, y_reg_train)
        y_predic_reg = model.predict(X_val)
        mae = mean_absolute_error(y_reg_val,y_predic_reg)
        rmse = np.sqrt(mean_squared_error(y_reg_val,y_predic_reg))
        print(f"{name} :: MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, name=f"{name}_regressor")


best_class_model = GaussianNB()



'''# confusion matrix for visualization
cm = confusion_matrix(y_class_val, y_predic_value)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes (Validation Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
'''
