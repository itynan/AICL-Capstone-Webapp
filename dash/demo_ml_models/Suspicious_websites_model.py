import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from lime.lime_text import LimeTextExplainer
import plotly.graph_objects as go

class TextModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.vectorizer = TfidfVectorizer(min_df=5)
        self.model = LogisticRegression(random_state=42)
        self.data = None
        self.X_train, self.X_dev, self.X_test = None, None, None
        self.y_train, self.y_dev, self.y_test = None, None, None

    def load_and_prepare_data(self):
        self.data = pd.read_csv(self.dataset_path, low_memory=False)
        self.data.dropna(subset=['Label', 'WebString'], inplace=True)

        X = self.data['WebString'].fillna("")  # Handling missing values if any remain
        y = self.data['Label']

        X_vectorized = self.vectorizer.fit_transform(X)
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.25, random_state=42)

        self.X_train, self.X_dev, self.X_test = X_train, X_dev, X_test
        self.y_train, self.y_dev, self.y_test = y_train, y_dev, y_test

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Assumes you have self.X_test and self.y_test already set
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, pos_label='Suspicious', average='binary')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


    def predict_and_explain(self, user_input):
        # Assuming your model has been trained and can predict probabilities
        explainer = LimeTextExplainer(class_names=['Normal', 'Suspicious'])

        # Wrap the predict_proba function for compatibility with Lime
        def predict_proba_wrapper(texts):
            return self.model.predict_proba(self.vectorizer.transform(texts))

        # Generate explanation
        exp = explainer.explain_instance(user_input, predict_proba_wrapper, num_features=10)

        # Get the list of explanations
        exp_list = exp.as_list()

        # Extract feature names and their weights
        features, weights = zip(*exp_list)

        # Return the data necessary for visualizations
        return features, weights, exp

    def generate_lime_explanation(self, user_input):
        # Initialize Lime for text
        explainer = LimeTextExplainer(class_names=['Normal', 'Suspicious'])

        # Vectorize the user input to match model's expected input format
        vectorized_input = self.vectorizer.transform([user_input])

        # Predict probabilities for the vectorized input
        probas = self.model.predict_proba(vectorized_input)

        # Generate explanation for the Suspicious class
        exp = explainer.explain_instance(user_input, lambda x: self.model.predict_proba(self.vectorizer.transform(x)), num_features=10, labels=[1])

        # Extract the feature names and their weights for the 'Suspicious' class
        exp_list = exp.as_list(label=1)
        features, weights = zip(*exp_list)

        # Convert to Plotly figure for Dash
        fig = go.Figure([go.Bar(x=weights, y=features, orientation='h')])
        fig.update_layout(title_text='Top contributing features to the prediction',
                          xaxis_title='Weight',
                          yaxis_title='Features')

        # Extract probabilities for easier understanding
        normal_proba = round(probas[0][0] * 100, 2)
        suspicious_proba = round(probas[0][1] * 100, 2)

        return fig, {'Normal': normal_proba, 'Suspicious': suspicious_proba}