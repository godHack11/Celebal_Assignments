# Take Any dataset of your choice and do the visualization

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
titanic = sns.load_dataset('titanic')
print(titanic.head())
# Count Plot – Number of passengers by sex
sns.countplot(data=titanic, x='sex')
plt.title("Number of Passengers by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# Bar Plot – Survival rate by class
sns.barplot(data=titanic, x='class', y='survived')
plt.title("Survival Rate by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# Pie Chart – Proportion of survivors vs non-survivors
survived_counts = titanic['survived'].value_counts()
labels = ['Not Survived', 'Survived']
plt.pie(survived_counts, labels=labels, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title("Survival Distribution")
plt.axis('equal')

plt.show()

# Histogram – Age distribution of passengers
plt.hist(titanic['age'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# Box Plot – Age vs Class
sns.boxplot(data=titanic, x='class', y='age')
plt.title("Age Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Age")
plt.show()

# Violin Plot – Age distribution by gender
sns.violinplot(data=titanic, x='sex', y='age')
plt.title("Age Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Age")
plt.show()

# Heatmap – Correlation between numeric variables
plt.figure(figsize=(8, 6))
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# FacetGrid – Survival by age and sex
g = sns.FacetGrid(titanic, col='survived', row='sex')
g.map_dataframe(sns.histplot, x='age', bins=20)
g.fig.suptitle("Age Distribution by Sex and Survival", y=1.03)
plt.show()
