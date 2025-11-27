import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"\Users\LAURA NWENEKA\Documents\Crop_Yield_Production_Project\data\Crop_recommendation.csv")
print("Data loaded successfully!")

#data["yield"] = (0.3 * data["N"] + 0.25 * data["P"] + 0.2 * data["K"] + 0.15 * data["rainfall"] + 0.1 * data["temperature"])
data["yield"] = (
    0.3 * data["N"] +
    0.25 * np.sqrt(data["P"]) +                           # nonlinear nutrient relationship
    0.2 * np.log1p(data["K"]) +                           # diminishing returns on potassium
    0.1 * data["rainfall"] * np.exp(-0.001 * data["rainfall"]) +  # rainfall penalty
    0.1 * (1 - np.abs(data["ph"] - 6.5)) * 10 +           # optimal pH around 6.5
    np.random.normal(0, 4, size=len(data))                # realistic random noise
)

numeric_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

X = data.drop(columns=["label", "yield"])
y_class = data["label"]
y_reg = data["yield"]

# splitting into training and a temporary set
X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.4, random_state=42, stratify=y_class
    )

#splitting the temporary set into validation and testing set
X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.5, random_state=42, stratify=y_class_temp
    )


__all__ = [
    "data",
    "X_train", "X_val", "X_test",
    "y_class_train", "y_class_val", "y_class_test",
    "y_reg_train", "y_reg_val", "y_reg_test",
]
