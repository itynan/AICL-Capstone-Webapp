import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from demo_ml_models.test2_password_gmm_clustering import (load_and_prepare_data, fit_gmm_model,
                                                          evaluate_password_strength_and_get_features,
                                                          calculate_aggregate_features)
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load the dataset and fit the model
features, y = load_and_prepare_data('demo_datasets/password_dataset.csv')
gmm_model = fit_gmm_model(features)

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("ML Model Selection"),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Gaussian Mixture Model', 'value': 'gmm'},
                    # Add more models here as needed
                ],
                placeholder="Select a Model",
                value='gmm'
            ),
        ], width=6),

        dbc.Col([
            html.H4("Dataset Selection"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[
                    {'label': 'Password Dataset', 'value': 'password_dataset'},
                    # Add more datasets here as needed
                ],
                placeholder="Select a Dataset",
                value='password_dataset'
            ),
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.H4("Enter a Password"),
            dbc.Input(id='password-input', type='text', placeholder="Enter a password"),
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Button('Run', id='run-button', n_clicks=0, color='primary', className="mt-3"),
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='output-container', className="mt-3"),
        ], width=12),
    ]),
], fluid=True)


# Define the callback to update the output based on the inputs
@app.callback(
    Output('output-container', 'children'),
    [Input('run-button', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('dataset-dropdown', 'value'),
     State('password-input', 'value')]
)
def update_output(n_clicks, selected_model, selected_dataset, password):
    if n_clicks > 0:
        if selected_model == 'gmm' and selected_dataset == 'password_dataset':
            result = evaluate_password_strength_and_get_features(gmm_model, password)
            if result is not None:
                label, input_features = result['cluster_label'], result['features']

                # Visualization for label distribution
                label_fig = px.bar(pd.DataFrame({'Label': ['Strong', 'Weak'], 'Count': [sum(y), len(y) - sum(y)]}),
                                   x='Label', y='Count', title='Label Distribution')

                all_cluster_labels = gmm_model.predict(features)

                # Clustering Visualization
                cluster_fig = go.Figure()

                # Color map for clusters
                cluster_colors = {0: 'red', 1: 'green'}  # Adjust based on your number of clusters

                # Plot all points from your dataset with cluster colors
                for cluster_label in np.unique(all_cluster_labels):
                    cluster_indices = np.where(all_cluster_labels == cluster_label)[0]
                    cluster_features = features.iloc[cluster_indices]
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
                    dcc.Graph(figure=cluster_fig),  # Clustering graph
                    dcc.Graph(figure=label_fig)  # Label distribution graph
                ])
            else:
                return html.Div('An error occurred, please try again.')
        else:
            return html.Div('Please select the Gaussian Mixture Model and the Password Dataset.')
    else:
        return html.Div('Click "Run" to process.')

import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7001))
    app.run(debug=True, port=port)