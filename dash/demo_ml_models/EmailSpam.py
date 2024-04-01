import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import shap


def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    data = pd.read_csv(file_path, encoding='latin-1')
    return data


def display_sample_messages(data):
    """
    Display random sample messages from the dataset.
    """
    print("\nRandom 15 Spam Messages:")
    print(data[data['Category'] == 'spam'].sample(15))

    print("\nRandom 15 Ham Messages:")
    print(data[data['Category'] == 'ham'].sample(15))
    print()


def visualize_message_distribution(data):
    """
    Visualize the distribution of spam and ham messages.
    """
    data['Category'].value_counts().plot.bar()
    plt.show()


def preprocess_data(data):
    """
    Preprocess the data by filtering spam and ham messages and converting text to lowercase.
    """
    data = data[data['Category'].isin(['spam', 'ham'])]
    data['Message'] = data['Message'].str.lower()
    return data


def split_data(data):
    """
    Split the dataset into training, development, and test sets.
    """
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(data['Message'], data['Category'], test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.25, random_state=42)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def transform_data(X_train, X_dev, X_test):
    """
    Transform text data into numerical format using CountVectorizer.
    """
    vectorizer = CountVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_dev_transformed = vectorizer.transform(X_dev)
    X_test_transformed = vectorizer.transform(X_test)
    return X_train_transformed, X_dev_transformed, X_test_transformed, vectorizer


def train_eval_model(model, X_train, y_train, X_dev, y_dev):
    """
    Train and evaluate the given model.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)
    f1 = metrics.f1_score(y_dev, predictions, average='weighted')
    return f1


def main():
    # Load data
    file_path = 'spam.csv'
    data = load_data(file_path)

    # Display sample messages
    display_sample_messages(data)

    # Visualize message distribution
    visualize_message_distribution(data)

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(data)

    # Transform data
    X_train_transformed, X_dev_transformed, X_test_transformed, vectorizer = transform_data(X_train, X_dev, X_test)

    # Train and evaluate models
    models = {
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(solver='liblinear')
    }
    for name, model in models.items():
        f1_score = train_eval_model(model, X_train_transformed, y_train, X_dev_transformed, y_dev)
        print(f"{name} model, F1 Score: {f1_score}")

    # LimeTextExplainer for Logistic Regression Model
    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(X_train_transformed, y_train)
    explainer = LimeTextExplainer(class_names=['ham', 'spam'])

    idx = 0
    text_instance = X_dev.iloc[idx]

    predict_function = lambda texts: lr_model.predict_proba(vectorizer.transform(texts))
    exp = explainer.explain_instance(text_instance, predict_function, num_features=6)
    print(f"\nLogistic Regression Explanation for Document id: {idx}")
    print('Probability=', lr_model.predict_proba(vectorizer.transform([text_instance]))[0, 1])
    print('True class: %s' % y_dev.iloc[idx])
    print(exp.as_list())

    fig = exp.as_pyplot_figure()
    plt.show()


if __name__ == "__main__":
    main()

