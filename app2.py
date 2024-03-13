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

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("ML Model Selection"),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Model 1', 'value': 'model1'},
                    {'label': 'Model 2', 'value': 'model2'}
                ],
                placeholder="Select a Model",
                value=None
            ),
        ], width=6),

        dbc.Col([
            html.H4("Dataset Selection"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[
                    {'label': 'Dataset 1', 'value': 'dataset1'},
                    {'label': 'Dataset 2', 'value': 'dataset2'}
                ],
                placeholder="Select a Dataset",
                value=None
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
        # Here you would run your model and return the results
        # For demonstration purposes, we'll just return the input values
        return html.Div([
            html.P(f"Selected Model: {selected_model}"),
            html.P(f"Selected Dataset: {selected_dataset}"),
            html.P(f"Entered Password: {password}"),
            html.P("Model output would go here...")
        ])
    return ""

import os
# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 6001))
    app.run(debug=True, port=port)