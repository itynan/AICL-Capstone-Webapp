import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import re
import string

# Define the feature extraction functions
def count_sequential_chars(password):
    count = 0
    for i in range(len(password) - 2):
        if ord(password[i]) + 1 == ord(password[i + 1]) and ord(password[i + 1]) + 1 == ord(password[i + 2]):
            count += 1
    return count

def count_repetitive_patterns(password):
    return len(re.findall(r'(.)\1+', password))

def extract_features(passwords):
    # Ensure all passwords are strings
    passwords = passwords.fillna('').astype(str)

    features = pd.DataFrame()
    features['length'] = passwords.apply(len)
    features['uppercase_count'] = passwords.apply(lambda x: sum(1 for c in x if c.isupper()))
    features['lowercase_count'] = passwords.apply(lambda x: sum(1 for c in x if c.islower()))
    features['digit_count'] = passwords.apply(lambda x: sum(1 for c in x if c.isdigit()))
    features['special_char_count'] = passwords.apply(lambda x: sum(1 for c in x if c in string.punctuation))
    features['sequential_char_count'] = passwords.apply(count_sequential_chars)
    features['repetitive_pattern_count'] = passwords.apply(count_repetitive_patterns)
    return features

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, header=None, skiprows=1, names=['Password', 'Label'])
    df_filtered = df[df['Label'].isin(['Strong', 'Weak'])]
    password_features = extract_features(df_filtered['Password'])
    y = (df_filtered['Label'] == 'Strong').astype(int)
    return password_features, y

# Define and fit the GMM Clustering model
def fit_gmm_model(features):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(features)
    return gmm

# Model Prediction for User Input
# Modify your function to return more detailed information
def evaluate_password_strength_and_get_features(gmm_model, password):
    try:
        user_password_features = extract_features(pd.Series([password]))

        # Normalize features if needed (uncomment and adjust as necessary)
        # scaler = StandardScaler().fit(user_password_features)
        # user_password_features = scaler.transform(user_password_features)

        user_cluster_label = gmm_model.predict(user_password_features)
        user_cluster_proba = gmm_model.predict_proba(user_password_features)[0]

        # Convert features and probabilities to a more readable format
        features_and_proba = {
            'cluster_label': int(user_cluster_label[0]),
            'probability': user_cluster_proba,
            'features': user_password_features.iloc[0].to_dict()
        }
        return features_and_proba
    except Exception as e:
        # Handle unexpected inputs or errors during prediction
        print(f"An error occurred: {e}")
        return None

def calculate_aggregate_features(dataframe, labels):
    """
    Calculate the aggregate features for strong and weak passwords.

    Args:
        dataframe (pd.DataFrame): DataFrame containing password features.
        labels (pd.Series): Series containing labels for each password (e.g., 0 for weak, 1 for strong).

    Returns:
        dict: A dictionary containing the aggregate features for strong and weak passwords.
    """
    # Split the dataset based on the labels
    strong_passwords = dataframe[labels == 1]
    weak_passwords = dataframe[labels == 0]

    # Calculate average features for strong passwords
    avg_strong = strong_passwords.mean().to_dict()

    # Calculate average features for weak passwords
    avg_weak = weak_passwords.mean().to_dict()

    return {
        'strong': avg_strong,
        'weak': avg_weak
    }

def training_data_split(features, labels):
    # Split into Training, Development, and Test Sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def f1_precision_recall(y_true, y_pred):
    # Calculate F1, Precision, and Recall
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return f1, precision, recall