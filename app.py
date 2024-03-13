import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px  # Added for using Plotly Express
import pandas as pd  # Added for using Pandas
from demo_ml_models.test2_password_gmm_clustering import load_and_prepare_data, fit_gmm_model, evaluate_password_strength_and_get_features, calculate_aggregate_features
import numpy as np
from flask import Flask
from flask_cors import CORS
import os


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# Add CORS support
CORS(app.server)

# Load the dataset and fit the model once, outside of the callback to avoid reloading on every callback
features, y = load_and_prepare_data('demo_datasets/password_dataset.csv')
gmm_model = fit_gmm_model(features)
# Calculate average features for strong and weak passwords
aggregate_features = calculate_aggregate_features(features, y)
# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src='download.jpg', height='100px'), width={"size": 6, "offset": 3}),  # Adjust 'height' as needed and center the logo
    ], justify="center"),  # Center the logo row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H4("Dataset/Model Selection", className="card-title"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[
                    {'label': 'Password Dataset', 'value': 'Dataset1'},
                    {'label': 'Dataset 2', 'value': 'Dataset2'}
                ],
                placeholder="Select a Dataset",
                value='Dataset1'
            ),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Password Strength Evaluator Model', 'value': 'Model1'},
                    {'label': 'Model 2', 'value': 'Model2'}
                ],
                placeholder="Select a Model",
                value='Model1'
            )
        ], body=True, className="mt-3 w-100")),
    ]),
    dbc.Row([
        dbc.Col(dbc.Input(id='malicious-string', type='text', placeholder="Enter a Password", className="mt-3 w-100")),
    ]),
    dbc.Row([
        dbc.Col(html.Button('Run', id='submit-val', n_clicks=0, className="btn mt-3 w-100",
            style={'backgroundColor': '#06066A', 'color': 'white', 'border': 'none'})),
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(id='output-container', className="mt-3 w-100")),
    ]),
    dbc.Row([  # This row is for the graph visualization
        dbc.Col(id='graph-container', className="mt-3 w-100"),
    ]),
], fluid=True, style={'backgroundColor': '#f0f0f0'})  # Add the style parameter here)
@app.callback(
    Output('output-container', 'children'),
    [Input('submit-val', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('dataset-dropdown', 'value'),
     State('malicious-string', 'value')]
)

def update_output(n_clicks, selected_model, selected_dataset, malicious_string):
    if n_clicks > 0 and selected_model == 'Model1' and selected_dataset == 'Dataset1':
        result = evaluate_password_strength_and_get_features(gmm_model, malicious_string)
        if result is not None:
            label, input_features = result['cluster_label'], result['features']

            # Load all dataset features and their GMM cluster assignments
            dataset_features, _ = load_and_prepare_data('demo_datasets/password_dataset.csv')

            # Load the original dataset with labels for distribution visualization
            original_data = pd.read_csv('demo_datasets/password_dataset.csv')

            # Visualization for label distribution
            label_fig = px.bar(original_data['Label'].value_counts(), title='Label Distribution')

            all_cluster_labels = gmm_model.predict(dataset_features)

            # Clustering Visualization
            cluster_fig = go.Figure()

            # Color map for clusters
            cluster_colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}  # Modify as needed based on your number of clusters

            # Plot all points from your dataset with cluster colors
            for cluster_label in np.unique(all_cluster_labels):
                cluster_indices = np.where(all_cluster_labels == cluster_label)[0]
                cluster_features = dataset_features.iloc[cluster_indices]
                cluster_fig.add_trace(go.Scatter(
                    x=cluster_features['length'], y=cluster_features['digit_count'],
                    mode='markers',
                    name=f'Cluster {cluster_label}',
                    marker=dict(color=cluster_colors[cluster_label])
                ))

            # Add the input password as a distinguished point
            cluster_fig.add_trace(go.Scatter(
                x=[input_features['length']], y=[input_features['digit_count']],
                mode='markers',
                name='Input Password',
                marker=dict(size=12, color='black', symbol='star')
            ))

            # Set titles and labels for the plot
            cluster_fig.update_layout(
                title='Password Feature Clustering',
                xaxis_title='Length',
                yaxis_title='Digit Count'
            )

            # Determine if the password is strong or weak and format the output accordingly
            prediction_text = "Model prediction for the entered password: "
            prediction = html.B("Strong") if label == 1 else html.B("Weak")

            # Return both graphs and prediction text
            return html.Div([
                html.Div([prediction_text, prediction]),
                dcc.Graph(figure=cluster_fig),  # Your clustering graph
                dcc.Graph(figure=label_fig)  # Your label distribution graph
            ])
        else:
            return html.Div('An error occurred, please try again.')
    else:
        return html.Div('Click "Run" to process.')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, port=port)