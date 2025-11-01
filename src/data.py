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
