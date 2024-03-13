import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Function to load dataset
def load_dataset(filename):
    data = pd.read_csv(filename)
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == object:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
    return data

# Function to split dataset
def split_dataset(data):
    X = data.drop('Label', axis=1)
    y = data['Label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to evaluate model
def evaluate_model(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    precision = precision_score(y_val, preds, average='weighted')  # Update for multiclass
    recall = recall_score(y_val, preds, average='weighted')        # Update for multiclass
    f1 = f1_score(y_val, preds, average='weighted')                # Update for multiclass
    return precision, recall, f1

# List datasets
datasets = os.listdir('../demo_datasets')
print("Available datasets:")
for i, dataset in enumerate(datasets):
    print(f"{i}: {dataset}")

# Select dataset
dataset_index = int(input("Enter the dataset number you want to use: "))
data = load_dataset(f'demo_datasets/{datasets[dataset_index]}')

# Split dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(data)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Evaluate models
results = {}
for name, model in models.items():
    precision, recall, f1 = evaluate_model(model, X_train, X_val, y_train, y_val)
    results[name] = f1
    print(f"{name}: Precision={precision:.3f}, Recall={recall:.3f}, F1 Score={f1:.3f}")

# Visualize the results with shades of blue and purple
colors = LinearSegmentedColormap.from_list("Custom", ["blue", "purple"])
color_range = colors(np.linspace(0, 1, len(results)))
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(results)), list(results.values()), align='center', color=color_range)
plt.xticks(range(len(results)), list(results.keys()), rotation=45)
plt.title('Model Comparison on F1 Score')
plt.ylabel('F1 Score')
plt.show()

# Model descriptions
model_descriptions = {
    "Logistic Regression": "A linear model for classification rather than regression. It is useful for cases where the outcome is a binary.",
    "Gaussian Naive Bayes": "Based on Bayes' theorem with the assumption of independence among predictors. Good for large feature sets.",
    "Support Vector Machine": "A powerful classifier that works well on a wide range of classification problems, even ones with complex boundaries.",
    "K-Nearest Neighbors": "A simple, instance-based learning algorithm where the class of a sample is determined by the majority class among its k nearest neighbors.",
    "Decision Tree": "A model that uses a tree-like graph of decisions and their possible consequences. Easy to interpret and understand.",
    "Random Forest": "An ensemble method that uses multiple decision trees to improve classification accuracy. Reduces overfitting risk.",
    "Gradient Boosting": "An ensemble technique that combines multiple weak learners to form a strong learner. Sequentially adds predictors to correct errors made by previous predictors.",
    "AdaBoost": "A boosting algorithm that combines multiple weak classifiers to create a strong classifier. Adjusts the weights of incorrectly classified instances so that subsequent classifiers focus more on difficult cases."
}

# Report the model with the highest F1 score with text graphic and include model description
best_model = max(results, key=results.get)
best_score = results[best_model]
separator = "=" * 50
title = "BEST MODEL REPORT"
report = f"The model with the highest F1 score is: {best_model}\nWith an F1 score of: {best_score:.2f}\n\nDescription: {model_descriptions[best_model]}"

# Text graphic
print("\n" + separator)
print(f"{title.center(len(separator))}")
print(separator)
print(report)
print(separator + "\n")
