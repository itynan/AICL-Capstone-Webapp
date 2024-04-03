import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from demo_ml_models.test2_password_gmm_clustering import (load_and_prepare_data, fit_gmm_model, evaluate_password_strength_and_get_features, calculate_aggregate_features, training_data_split)
from demo_ml_models.Suspicious_websites_model import TextModel

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer


# Load the dataset and fit the model once, outside of the callback to avoid reloading on every callback
features, y = load_and_prepare_data('demo_datasets/password_dataset.csv')
gmm_model = fit_gmm_model(features)

# Initialize use case 2 sus web apps dataset
text_model = TextModel('demo_datasets/suspicious_webapps_dataset.csv')

text_model.load_and_prepare_data()  # This prepares your data
text_model.train_model()  # Ensure this line is added to train your model

# Calculate average features for strong and weak passwords
aggregate_features = calculate_aggregate_features(features, y)

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src='/assets/full_AI_secops.png', height='100px'), width={"size": 6, "offset": 3}),  # Adjust 'height' as needed and center the logo
    ], justify="center"),  # Center the logo row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H4("Dataset/Model Selection", className="card-title"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[
                    {'label': 'Password Dataset', 'value': 'Dataset1'},
                    {'label': 'Suspicious webapps Dataset', 'value': 'Dataset2'}
                ],
                placeholder="Select a Dataset",
                value='Dataset1'
            ),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Password Strength Evaluator Model (GMM clustering)', 'value': 'Model1'},
                    {'label': 'Suspicious webapp Strings', 'value': 'Model2'}
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
    if n_clicks <=0:
         return html.Div('Click "Run" to process.')

    # Check if the selected model and dataset are correctly paired
    if (selected_model == 'Model1' and selected_dataset != 'Dataset1') or \
       (selected_model == 'Model2' and selected_dataset != 'Dataset2'):
        return html.Div('Please select the corresponding dataset and model.')


    if selected_model == 'Model1' and selected_dataset == 'Dataset1':
        result = evaluate_password_strength_and_get_features(gmm_model, malicious_string)
        if result is not None:
            label, input_features = result['cluster_label'], result['features']

            # Data splitting
            X_train, X_dev, X_test, y_train, y_dev, y_test = training_data_split(features, y)

            # Create a pie chart for the dataset split
            split_fig = go.Figure(data=[
                go.Pie(labels=['Train', 'Development', 'Test'],
                       values=[len(X_train), len(X_dev), len(X_test)],
                       hole=.3)
            ])

            # Set pie chart layout
            split_fig.update_layout(
                title_text='Dataset Split',
                annotations=[dict(text='Dataset', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )

            # Load all dataset features and their GMM cluster assignments
            dataset_features, _ = load_and_prepare_data('demo_datasets/password_dataset.csv')

            all_cluster_labels = gmm_model.predict(dataset_features)

            # Clustering Visualization
            cluster_fig = go.Figure()

            # Visualization for label distribution
            original_data = pd.read_csv('demo_datasets/password_dataset.csv')
            label_fig = px.bar(original_data['Label'].value_counts(), title='Label Distribution')

            # Now process the random sample
            sample_data = original_data.sample(15)  # Get random sample
            sample_table = dash_table.DataTable(
                data=sample_data.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in sample_data.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'white', 'fontWeight': 'bold'}
            )

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

            # Convert the samples to Dash DataTable for pretty display
            sample_table = dash_table.DataTable(
                data=sample_data.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in sample_data.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'white', 'fontWeight': 'bold'}
            )

            sample_data_section = dcc.Loading(
                id="loading-sample-data",
                children=[
                    html.Div([
                        html.H4('Random Sample from Selected Dataset'),
                        sample_table
                    ])
                ],
                type="default"
            )

            # Wrap your graphs in the Loading component
            clustering_graph = dcc.Loading(
                id="loading-clustering",
                children=[dcc.Graph(figure=cluster_fig)],
                type="default"
            )

            label_distribution_graph = dcc.Loading(
                id="loading-label-distribution",
                children=[dcc.Graph(figure=label_fig)],
                type="default"
            )

            prediction_text_display = html.Div([html.Div([prediction_text, prediction])])

            # Return both graphs, prediction text, and the sample data table
            return html.Div([
            prediction_text_display,
            clustering_graph,
            label_distribution_graph,
            dcc.Graph(figure=split_fig),  # Add the dataset splits graph
            sample_data_section
            #metrics_div  # Display the metrics
            ])
        else:
            return html.Div('An error occurred, please try again.')

    elif selected_model == 'Model2' and selected_dataset == 'Dataset2':

        try:
            # Dataset split visualization:
            split_fig_text = go.Figure(data=[
                go.Pie(labels=['Train', 'Development', 'Test'],
                       values=[text_model.X_train.shape[0], text_model.X_dev.shape[0], text_model.X_test.shape[0]],
                       # changed from len() to .shape[0]
                       hole=.5)
            ])

            split_fig_text.update_layout(
                title_text='Suspicious String Dataset Split',
                annotations=[dict(text='String Dataset', x=0.5, y=0.5, font_size=15, showarrow=False)]
            )
        except Exception as e:
            return html.Div(f'Error in dataset split visualization: {e}')

            # Continue adding the rest of your visualization logic here
            # Update label distribution visualization
            label_counts = pd.Series(text_model.y_train).value_counts() + pd.Series(text_model.y_dev).value_counts() + pd.Series(text_model.y_test).value_counts()
            label_fig_text = go.Figure(data=[
                go.Bar(x=label_counts.index, y=label_counts.values, marker_color=['blue', 'red'])  # Adjust colors as needed
            ])

            label_fig_text.update_layout(
                title='Label Distribution in Text Dataset',
                xaxis_title='Labels',
                yaxis_title='Count'
            )


            # Update label distribution visualization
            label_counts = pd.Series(text_model.y_train).value_counts() + pd.Series(text_model.y_dev).value_counts() + pd.Series(text_model.y_test).value_counts()
            label_fig_text = go.Figure(data=[
                go.Bar(x=label_counts.index, y=label_counts.values, marker_color=['blue', 'red'])  # Adjust colors as needed
            ])
            label_fig_text.update_layout(
                title='Label Distribution in Suspicious String Dataset',
                xaxis_title='Labels',
                yaxis_title='Count'
            )

            # Now process the random sample from dataset for display
            original_data = pd.read_csv('data_datasets/suspicious_webapps_dataset.csv')
            sample_data = original_data.sample(15)  # Get random sample
            sample_table = dash_table.DataTable(
                data=sample_data.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in sample_data.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'white', 'fontWeight': 'bold'}
            )
            sample_data_section = dcc.Loading(
                id="loading-sample-data",
                children=[
                    html.Div([
                        html.H4('Random Sample from Selected Dataset'),
                        sample_table
                    ])
                ],
                type="default"
            )

            try:
                lime_fig, probas_dict = text_model.generate_lime_explanation(malicious_string)

                # Use the extracted probabilities for display
                probabilities_display = html.Div([
                    html.P([html.B("Model Prediction:")]),
                    html.P(f"Normal website probability: {probas_dict['Normal']}%"),
                    html.P(f"Suspicious probability: {probas_dict['Suspicious']}%")
                ])
            except Exception as e:
                return html.Div(f'Error in LIME explanation generation: {e}')

            # Evaluate the model
            evaluation_metrics = text_model.evaluate_model()

            # HTML components to display evaluation metrics
            metrics_display = dbc.Row([
                dbc.Col(html.Div(f"Accuracy: {evaluation_metrics['accuracy']*100:.2f}%"), width=3),
                dbc.Col(html.Div(f"Precision: {evaluation_metrics['precision']*100:.2f}%"), width=3),
                dbc.Col(html.Div(f"Recall: {evaluation_metrics['recall']*100:.2f}%"), width=3),
                dbc.Col(html.Div(f"F1 Score: {evaluation_metrics['f1']*100:.2f}%"), width=3)
            ])

            # Combine debug info with other visualization components
            return html.Div([
                #debug_info,
                metrics_display,
                html.Br(), #Add a space
                probabilities_display,
                dcc.Graph(figure=lime_fig),
                dcc.Graph(figure=split_fig_text),
                dcc.Graph(figure=label_fig_text),
                sample_data_section
            ])
        except Exception as e:
            # If there's an error, show it along with the debug info
            return html.Div([
                #debug_info,
                html.Div(f'An error occurred, please try again.')
                #to show exception {e}
            ])

    else:
        return html.Div('Click "Run" to process.')

# import os
# server=app.server
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT",7003))
#     #port = int(os.environ.get("PORT",host='0.0.0.0', port=7002))
#     app.run_server(debug=True, port=port)