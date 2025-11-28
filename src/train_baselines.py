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
        acc_val = accuracy_score(y_class_val, y_class_prediction)
        f1_val = f1_score(y_class_val, y_class_prediction, average='weighted')

        #test
        y_class_test_prediction = model.predict(X_test)
        acc_test = accuracy_score(y_class_test, y_class_test_prediction)
        f1_test = f1_score(y_class_test, y_class_test_prediction, average='weighted')

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy_val", acc_val)
        mlflow.log_metric("f1_score_val", f1_val)
        mlflow.log_metric("accuracy_test", acc_test)
        mlflow.log_metric("f1_score_test", f1_test)
        mlflow.sklearn.log_model(model, name =f"{name}_classifier")

        # adding to array to put in table
        classification_results.append({
            "Model": name,
            "Accuracy (Validation)": round(acc_val, 3),
            "F1 (Validation)": round(f1_val, 3),
            "Accuracy (Test)": round(acc_test, 3),
            "F1 (Test)": round(f1_test, 3)
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
        mae_val = mean_absolute_error(y_reg_val,y_reg_prediction)
        rmse_val = np.sqrt(mean_squared_error(y_reg_val,y_reg_prediction))

        #test
        y_reg_test_prediction = model.predict(X_test)
        mae_test = mean_absolute_error(y_reg_test, y_reg_test_prediction)
        rmse_test = np.sqrt(mean_squared_error(y_reg_test, y_reg_test_prediction))



        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("mae_val", mae_val)
        mlflow.log_metric("rmse_val", rmse_val)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.sklearn.log_model(model, name=f"{name}_regressor")

        #adding to array to put in table
        regression_results.append({
            "Model": name,
            "MAE (Validation)": round(mae_val, 3),
            "RMSE (Validation)": round(rmse_val, 3),
            "MAE (Test)": round(mae_test, 3),
            "RMSE (Test)": round(rmse_test, 3),
        })


table1 = pd.DataFrame(classification_results) #Thes tables will be in our pdf
table2 = pd.DataFrame(regression_results)

print("\nTable 1 – Classification Metrics")
print(table1.to_string(index=False))

print("\nTable 2 – Regression Metrics")
print(table2.to_string(index=False))

table1.to_csv("Table1_Classification_baseline.csv", index=False)
table2.to_csv("Table2_Regression_baseline.csv", index=False)
