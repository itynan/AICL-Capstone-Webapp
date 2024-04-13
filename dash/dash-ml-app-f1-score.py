# Required libraries for Dash and data handling1
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os

# Machine Learning and Evaluation Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Load dataset
def load_dataset(filename):
    data = pd.read_csv(filename)
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == object:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
    return data

# Split dataset
def split_dataset(data):
    X = data.drop('Label', axis=1)
    y = data['Label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Evaluate model
def evaluate_model(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    precision = precision_score(y_val, preds, average='weighted')
    recall = recall_score(y_val, preds, average='weighted')
    f1 = f1_score(y_val, preds, average='weighted')
    return precision, recall, f1

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],routes_pathname_prefix='/f1score/',requests_pathname_prefix='/f1score/')

datasets = [f for f in os.listdir('./demo_datasets') if os.path.isfile(os.path.join('./demo_datasets', f))]


#Get the absolute path of the directory where this script is located
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# datasets_dir = os.path.join(BASE_DIR, 'demo_datasets')
# datasets = [f for f in os.listdir(datasets_dir) if os.path.isfile(os.path.join(datasets_dir, f))]


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Start with your Dataset!"),
            html.P("""
                A dataset is a collection of data used to train, test, and validate models.""",
                className="mb-4"),
        ]),
        dbc.Col([
            html.H2("Machine Learning Model?"),
            html.P("""
                A model is an algorithm that learns patterns from your selected dataset to make predictions or decisions without being explicitly programmed.""",
                className="mb-4"),
        ]),
        dbc.Col([
            html.H2("Why F1 Score?"),
            html.P("""
                The F1 score is useful because it evaluates the accuracy of a test by considering its ability to minimize false positives.""",
                className="mb-4"),
        ])
    ], className="mb-5"),
    dbc.Row([
        dbc.Col([
            html.H4("Dataset Selection"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[{'label': dataset, 'value': dataset} for dataset in datasets],
                value=datasets[0] if datasets else None
            ),
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button("Evaluate Models", id="evaluate-button", n_clicks=0, className="mt-3"),
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-results",
                type="default",
                children=html.Div(id="evaluation-results")
            ),
        ], width=12),
    ]),
], fluid=True)

@app.callback(
    Output('evaluation-results', 'children'),
    [Input('evaluate-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')]
)
def update_evaluation_results(n_clicks, selected_dataset):
    if n_clicks > 0 and selected_dataset:
        data = load_dataset(f'./demo_datasets/{selected_dataset}')
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(data)

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

         # Model descriptions
        model_descriptions = {
            "Logistic Regression": "A linear model for classification rather than regression. It is useful for cases where the outcome is binary.",
            "Gaussian Naive Bayes": "Based on Bayes' theorem with the assumption of independence among predictors. Good for large feature sets.",
            "Support Vector Machine": "A powerful classifier that works well on a wide range of classification problems, even ones with complex boundaries.",
            "K-Nearest Neighbors": "A simple, instance-based learning algorithm where the class of a sample is determined by the majority class among its k nearest neighbors.",
            "Decision Tree": "A model that uses a tree-like graph of decisions and their possible consequences. Easy to interpret and understand.",
            "Random Forest": "An ensemble method that uses multiple decision trees to improve classification accuracy. Reduces overfitting risk.",
            "Gradient Boosting": "An ensemble technique that combines multiple weak learners to form a strong learner. Sequentially adds predictors to correct errors made by previous predictors.",
            "AdaBoost": "A boosting algorithm that combines multiple weak classifiers to create a strong classifier. Adjusts the weights of incorrectly classified instances so that subsequent classifiers focus more on difficult cases."
        }

        results = {}
        for name, model in models.items():
            precision, recall, f1 = evaluate_model(model, X_train, X_val, y_train, y_val)
            results[name] = {'f1': f1, 'precision': precision, 'recall': recall}

        best_model_name = max(results, key=lambda x: results[x]['f1'])
        best_model_results = results[best_model_name]
        best_model_description = model_descriptions[best_model_name]


        fig = px.bar(x=list(results.keys()), y=[result['f1'] for result in results.values()], labels={'x': 'Model', 'y': 'F1 Score'}, title='F1 Scores of Evaluated Models')
        
        best_model_report = dbc.Card([
            dbc.CardBody([
                html.H5("Best Model", className="card-title"),
                html.P(f"Model: {best_model_name}", className="card-text"),
                html.P(f"F1 Score: {best_model_results['f1']:.3f}", className="card-text"),
                html.P(f"Precision: {best_model_results['precision']:.3f}", className="card-text"),
                html.P(f"Description: {best_model_description}", className="card-text"),  # Include the model description
            ])
        ], style={"width": "auto"})

        return [
            dcc.Graph(figure=fig),
            html.Div(best_model_report, style={'marginTop': 20})
        ]
    else:
        return html.Div("Select a dataset and click 'Evaluate Models' to start.")

server=app.server
if __name__ == '__main__':
    port = int(os.environ.get("PORT",7002))
    #port = int(os.environ.get("PORT",host='0.0.0.0', port=7002))
    app.run_server(debug=True, port=port)