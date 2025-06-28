Train multiple machine learning models and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score. Implement hyperparameter tuning techniques like 
GridSearchCV and RandomizedSearchCV to optimize model parameters. Analyze the results to select the best-performing model.


import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import randint

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} :")
    print(classification_report(y_test, y_pred))

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 'scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro')
grid.fit(X_train, y_train)
print("\nBest Grid params:", grid.best_params_)
y_pred_grid = grid.predict(X_test)
print("Grid F1:", f1_score(y_test, y_pred_grid, average='macro'))

param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': [10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': randint(2, 10)
}

rand = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=20, cv=5, scoring='f1_macro', random_state=42)
rand.fit(X_train, y_train)
print("\nBest Random params:", rand.best_params_)
y_pred_rand = rand.predict(X_test)
print("Random F1:", f1_score(y_test, y_pred_rand, average='macro'))
























