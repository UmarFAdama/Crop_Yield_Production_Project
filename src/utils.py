import pandas as pd

# 1. Classification metrics (Table 1)

# Example placeholders – replace with your real numbers
table1_data = [
    {
        "Model": "Naive Bayes",
        "Val Accuracy": 0.995,   # <- replace
        "Test Accuracy": 0.993,  # <- replace
        "Val F1": 0.995,         # <- replace
        "Test F1": 0.993,        # <- replace
    },
    {
        "Model": "Decision Tree",
        "Val Accuracy": 0.964,
        "Test Accuracy": 0.977,
        "Val F1": 0.964,
        "Test F1": 0.977,
    },
    {
        "Model": "NN Classifier (MLP)",
        "Val Accuracy": 0.980,
        "Test Accuracy": 0.977,
        "Val F1": 0.980,
        "Test F1": 0.978,
    },
]

# 2. Regression metrics (Table 2)

table2_data = [
    {
        "Model": "Linear Regression",
        "Val MAE": 3.164,
        "Test MAE": 3.165,
        "Val RMSE": 3.939,
        "Test RMSE": 3.997,
    },
    {
        "Model": "Decision Tree Regressor (Classical)",
        "Val MAE": 4.387,
        "Test MAE": 4.670,
        "Val RMSE": 5.477,
        "Test RMSE": 5.817,
    },
    {
        "Model": "NN Regressor (MLP)",
        "Val MAE": 3.345,
        "Test MAE": 3.480,
        "Val RMSE": 4.144,
        "Test RMSE": 4.418,
    },
]


def main():
    table1 = pd.DataFrame(table1_data)
    table2 = pd.DataFrame(table2_data)

    print("\nTable 1 – Classification comparison (Classical vs NN)")
    print(table1.to_string(index=False))

    print("\nTable 2 – Regression comparison (Classical vs NN)")
    print(table2.to_string(index=False))


    table1.to_csv("Table1_Classification.csv", index=False)
    table2.to_csv("Table2_Regression.csv", index=False)


if __name__ == "__main__":
    main()
