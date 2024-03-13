import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

# Load dataset
df = pd.read_csv('../demo_datasets/suspicious_webapps_dataset.csv')  # Update this path

# Fill NaN values in the 'WebString' column with an empty string
df['WebString'] = df['WebString'].fillna('')

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(df['WebString'], df['Label'], test_size=0.4, random_state=42)
X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_tfidf, y_train)

# Dimensionality Reduction
svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X_train_tfidf)


# Function to plot user input in the context of training data
def plot_user_input(user_input):
    user_input_tfidf = vectorizer.transform([user_input])
    user_input_reduced = svd.transform(user_input_tfidf)

    distances, indices = knn.kneighbors(user_input_tfidf)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train, cmap='viridis', alpha=0.5, label='Training Points')
    plt.scatter(user_input_reduced[:, 0], user_input_reduced[:, 1], c='red', label='User Input')

    # Highlight the nearest neighbor
    for i in indices[0]:
        plt.scatter(X_reduced[i, 0], X_reduced[i, 1], s=100, facecolors='none', edgecolors='r')

    plt.legend()
    plt.title('User Input in the Context of Training Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


# Ask user for input
user_input = input("Enter a string to test: ")
plot_user_input(user_input)

# Predict and print the classification for the user input
prediction = knn.predict(vectorizer.transform([user_input]))
print(f"Predicted label for input '{user_input}': {label_encoder.inverse_transform(prediction)[0]}")
