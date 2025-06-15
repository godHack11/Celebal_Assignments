import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print("\nShape of data:", data.shape)
print("\nData types:\n", data.dtypes)
print("\nSummary statistics:\n", data.describe())
print("\nMissing values:\n", data.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cmap='magma', cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

data.hist(bins=20, color='skyblue', edgecolor='black', figsize=(15, 10))
plt.suptitle("Histograms of Numeric Columns", fontsize=16)
plt.tight_layout()
plt.show()

num_cols = data.select_dtypes(include=np.number).columns

for column in num_cols:
    plt.figure()
    sns.boxplot(x=data[column], color='lightgreen')
    plt.title(f"Boxplot of {column}")
    plt.show()
    
# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Between Numeric Features")
plt.show()

# Pairplot for selected columns
features = ["Age", "Fare", "Pclass", "Survived"]
df_plot = data[features].dropna()

sns.pairplot(df_plot, hue="Survived", palette="husl")
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Count plots for categorical features
cat_cols = ["Sex", "Embarked", "Pclass", "Survived"]

for col in cat_cols:
    plt.figure()
    sns.countplot(x=col, data=data, palette="pastel")
    plt.title(f"Count Plot of {col}")
    plt.show()           

# Final Summary of the Analysis
print("Summary of Findings:\n")
print("""
1. Dataset has 891 rows and 12 columns.
2. Missing values found in 'Age', 'Cabin', and 'Embarked'.
3. 'Fare' is highly skewed with some high-paying passengers.
4. Outliers are present in 'Fare' and 'Age'.
5. 'Sex' and 'Pclass' strongly affect survival chances.
6. 'Fare' and 'Pclass' are correlated (negative relation).
7. Pairplot shows visible separation between survivors and non-survivors.
8. Categorical features like 'Sex' and 'Embarked' influence outcomes.
""")
























