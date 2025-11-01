import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\LAURA NWENEKA\Documents\Crop_Yield_Production_Project\data\Crop_recommendation.csv")
print("Data loaded successfully!")
print(data.head())


data["yield"] = (0.3 * data["N"] + 0.25 * data["P"] + 0.2 * data["K"] + 0.15 * data["rainfall"] + 0.1 * data["temperature"])
print(data.head())

numeric_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

for feature in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=data[feature], y=data["yield"], alpha=0.7)
    plt.title(f"{feature} vs Synthetic Yield")
    plt.xlabel(feature)
    plt.ylabel("Yield")
    plt.show()


X = data.drop(columns=["label", "yield"])
print(type(X))
print(X.shape)

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
