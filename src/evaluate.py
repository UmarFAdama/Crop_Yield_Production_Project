import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data import data


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



