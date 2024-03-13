import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree


def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    data = pd.read_csv(file_path)
    return data


def encode_labels(data):
   
    # Encode class labels using LabelEncoder.
    
    label_encoder = LabelEncoder()
    data['class'] = label_encoder.fit_transform(data['class'])
    return label_encoder


def extract_features(data):
    
    # Extract features from domain names.
    
    data['Length'] = data['host'].apply(len)
    data['ContainsDigits'] = data['host'].apply(lambda x: any(char.isdigit() for char in x))
    data['ContainsHyphen'] = data['host'].apply(lambda x: '-' in x)
    return data[['Length', 'ContainsDigits', 'ContainsHyphen']], data['class']


def train_model(X_train, y_train):

    #Train the Decision Tree classifier.
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):

    #Evaluate the trained model.

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm


def visualize_length_distribution(data):
    
    #Visualize the domain length distribution by class.
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Length', hue='class', kde=True, bins=20)
    plt.title('Domain Length Distribution by Class')
    plt.xlabel('Domain Length')
    plt.ylabel('Frequency')
    plt.show()


def visualize_tld_distribution(data):
    
    #Visualize the top 10 TLDs by class.

    data['TLD'] = data['host'].apply(lambda x: x.split('.')[-1])

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=data, x='TLD', hue='class', order=data['TLD'].value_counts().index[:10])
    plt.title('Top 10 TLDs by Class')
    plt.xlabel('TLD')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Customize legend labels
    legend_labels = ['Legit', 'Malicious']  # Replace with your desired labels
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels)

    plt.show()



def main():
    # Load data
    data = load_data("legit-malware_domains.csv")


    # Encode class labels
    label_encoder = encode_labels(data)

    # Extract features
    X, y = extract_features(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    clf = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, cm = evaluate_model(clf, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)

    # Visualize domain length distribution
    visualize_length_distribution(data)

    # Visualize top-level domain distribution
    visualize_tld_distribution(data)

    plt.figure(figsize=(20, 10), dpi=300)
    plot_tree(clf, feature_names=X.columns, class_names=label_encoder.classes_, filled=True)

    # Training the Decision Tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    plt.show()

    # Making predictions
    y_pred = clf.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



if __name__ == "__main__":
    main()
