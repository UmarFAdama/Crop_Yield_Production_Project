import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"data\Crop_recommendation.csv")
print("Data loaded successfully!")

data["yield"] = (0.3 * data["N"] + 0.25 * data["P"] + 0.2 * data["K"] + 0.15 * data["rainfall"] + 0.1 * data["temperature"])

numeric_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

X = data.drop(columns=["label", "yield"])
y_class = data["label"]
y_reg = data["yield"]

# splitting into training and a temporary set
X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.3, random_state=42, stratify=y_class
    )

#splitting the temporary set into validation and testing set
X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.5, random_state=42, stratify=y_class_temp
    )

