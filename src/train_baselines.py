from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from data import X_train, X_val, X_test, y_class_val, y_class_train, y_class_test
from data import y_reg_val, y_reg_train, y_reg_test
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
import mlflow
import pandas as pd


classification_results = []
regression_results = []

model_class = {
    "Naive Bays": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=0)}



for name, model in model_class.items():
    with mlflow.start_run(run_name=f"Classification - {name}"):
        model.fit(X_train, y_class_train)
        # validation
        y_class_prediction = model.predict(X_val)
        acc = accuracy_score(y_class_val, y_class_prediction)
        f1 = f1_score(y_class_val, y_class_prediction, average='weighted')

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_score", f1)
        mlflow.sklearn.log_model(model, name =f"{name}_classifier")

        # adding to array to put in table
        classification_results.append({
            "Model": name,
            "Accuracy (Validation)": round(acc, 3),
            "F1 (Validation)": round(f1, 3)
        })

# Best classifier: Naive Bayes

model_reg = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=0)
}

for name, model in model_reg.items():
    with mlflow.start_run(run_name=f"Regression - {name}"):
        model.fit(X_train, y_reg_train)

        #validation
        y_reg_prediction = model.predict(X_val)
        mae = mean_absolute_error(y_reg_val,y_reg_prediction)
        rmse = np.sqrt(mean_squared_error(y_reg_val,y_reg_prediction))


        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.sklearn.log_model(model, name=f"{name}_regressor")

        #adding to array to put in table
        regression_results.append({
            "Model": name,
            "MAE (Validation)": round(mae, 3),
            "RMSE (Validation)": round(rmse, 3),
        })



table1 = pd.DataFrame(classification_results) #Thes tables will be in our pdf
table2 = pd.DataFrame(regression_results)

print("\nTable 1 – Classification Metrics")
print(table1.to_string(index=False))

print("\nTable 2 – Regression Metrics")
print(table2.to_string(index=False))

table1.to_csv("Table1_Classification_Metrics.csv", index=False)
table2.to_csv("Table2_Regression_Metrics.csv", index=False)
