'''RFM analysis is a customer segmentation technique used by businesses to evaluate customer value based on three key metrics: recency, frequency, and monetary value. This method helps companies identify their most valuable customers and tailor marketing strategies accordingly.

Recency: Measures how recently a customer has made a purchase.
Frequency: Indicates how often a customer makes purchases.
Monetary Value: Refers to the total amount spent by a customer over all transactions.'''

import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime
import plotly.colors

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

pio.templates.default = "plotly_white"

data = pd.read_csv("rfm_data.csv")

data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'])

data['Recency'] = (datetime.now() - data['PurchaseDate']).dt.days

frequency_data = data.groupby('CustomerID')['OrderID'].count().reset_index()
frequency_data.rename(columns={'OrderID': 'Frequency'}, inplace=True)
data = data.merge(frequency_data, on='CustomerID', how ='left')

monetary_data = data.groupby('CustomerID')['TransactionAmount'].sum().reset_index()
monetary_data.rename(columns={'TransactionAmount': 'MonetaryValue'}, inplace=True)
data = data.merge(monetary_data, on='CustomerID', how='left')

recency_scores = [5, 4, 3, 2, 1]
frequency_scores = [1, 2, 3, 4, 5]
monetary_scores =[1, 2, 3, 4, 5]

data['RecencyScore'] = pd.cut(data['Recency'], bins=5, labels=recency_scores)
data['FrequencyScore'] = pd.cut(data['Frequency'], bins=5, labels=frequency_scores)
data['MonetaryScore'] = pd.cut(data['MonetaryValue'], bins=5, labels=monetary_scores)

data['RecencyScore'] = data['RecencyScore'].astype(int)
data['FrequencyScore'] = data['FrequencyScore'].astype(int)
data['MonetaryScore'] = data['MonetaryScore'].astype(int)

data['RFM_Score'] = data['RecencyScore'] + data['FrequencyScore'] + data['MonetaryScore']

segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
data['Value Segment'] = pd.qcut(data['RFM_Score'], q=3, labels=segment_labels)

segment_counts = data['Value Segment'].value_counts().reset_index()
segment_counts.columns = ['Value Segment', 'Count']

pastel_colors = px.colors.qualitative.Pastel

fig_segment_dist = px.bar(segment_counts, x='Value Segment', y='Count', color='Value Segment',
                          color_discrete_sequence=pastel_colors,
                          title='RFM Value Segment Distribution')

fig_segment_dist.update_layout(xaxis_title='RFM Value Segment',
                               yaxis_title='Count',
                               showlegend=True)
#RFM Value Segment
#fig_segment_dist.show()


data['RFM Customer Segments'] = ''

data.loc[data['RFM_Score'] >= 9, 'RFM Customer Segments'] = 'Champions'
data.loc[(data['RFM_Score'] >= 6) & (data['RFM_Score'] < 9), 'RFM Customer Segments'] = 'Potential Loyalists'
data.loc[(data['RFM_Score'] >= 5) & (data['RFM_Score'] < 6), 'RFM Customer Segments'] = 'At Risk Customers'
data.loc[(data['RFM_Score'] >= 4) & (data['RFM_Score'] < 5), 'RFM Customer Segments'] = "Can't Lose"
data.loc[(data['RFM_Score'] >= 3) & (data['RFM_Score'] < 4), 'RFM Customer Segments'] = 'Lost'

segment_product_counts = data.groupby(['Value Segment', 'RFM Customer Segments']).size().reset_index(name='Count')

segment_product_counts = segment_product_counts.sort_values('Count', ascending=False)

fig_treemap_segment_product = px.treemap(segment_product_counts, path=['Value Segment', 'RFM Customer Segments'],
                                         values='Count',
                                         color='Value Segment', color_discrete_sequence=px.colors.qualitative.Pastel,
                                         title='RFM Customer Segments by Value')

#fig_treemap_segment_product.show()


#filter the data to include customers in champions segment only
champions_segment = data[data['RFM Customer Segments'] == 'Champions']

champions_segment_fig = go.Figure()
champions_segment_fig.add_trace(go.Box(y=champions_segment['RecencyScore'], name='Recency'))
champions_segment_fig.add_trace(go.Box(y=champions_segment['FrequencyScore'], name='Frequency'))
champions_segment_fig.add_trace(go.Box(y=champions_segment['MonetaryScore'], name='Monetary'))

champions_segment_fig.update_layout(title='Distribution of RFM Values within Champions Segment',
                                    yaxis_title='RFM Value',
                                    showlegend=True)
#champions_segment_fig.show()

correlation_matrix = champions_segment[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].corr()

#visualize correlation matrix using a heatmap
fig_corr_heatmap = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='RdBu',
    colorbar=dict(title='Correlation')
))
fig_corr_heatmap.update_layout(title='Correlation Matrix of RFM Values within Champions Segment')

#fig_corr_heatmap.show()




pastel_colors = plotly.colors.qualitative.Pastel

segment_counts = data["RFM Customer Segments"].value_counts()

#Bar Chart to compare segment counts
comparision_fig = go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values,
                                         marker=dict(color=pastel_colors))])

#set color
champions_color = 'rgb(158,202,225)'
comparision_fig.update_traces(marker=dict(color=[
    champions_color if segment == 'Champions' else pastel_colors[i % len(pastel_colors)]
    for i, segment in enumerate(segment_counts.index)
]))
comparision_fig.update_layout(title='Comparison of RFM Segments',
                              xaxis_title= 'RFM Segments',
                              yaxis_title='Number fo Customers',
                              showlegend=True)
#comparision_fig.show()


segment_scores = data.groupby('RFM Customer Segments')[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].mean().reset_index()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['RecencyScore'],
    name='Recency Score',
))

fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['FrequencyScore'],
    name='Frequency Score',
))

fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['MonetaryScore'],
    name='Monetary Score',
))

fig.update_layout(
    title='Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores',
    xaxis_title='RFM Segments',
    yaxis_title='Score',
    barmode='group',
    showlegend=True
)

#fig.show()


#Creating a dashborard for the RFM Analysis
app = dash.Dash(__name__, external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'])

app.layout = html.Div([
    html.Header([
        html.H1('RFM Analysis Dashboard', className='text-center mb-4', style={'fontFamily': 'Arial', 'color': '#007bff'}),
        html.P('Analyze customer segments based on RFM scores.', className='text-center mb-4', style={'fontSize': '18px'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #007bff'}),

    html.Div([
        dcc.Dropdown(
            id='chart-type-dropdown',
            options=[
                {'label': 'RFM Value Segment Distribution', 'value': 'segment_distribution'},
                {'label': 'RFM Customer Segments Treemap', 'value': 'RFM_distribution'},
                {'label': 'Correlation Matrix of RFM Values within Champions Segment', 'value': 'correlation_matrix'},
                {'label': 'Comparison of RFM Segments', 'value': 'segment_comparison'},
                {'label': 'Comparison of RFM Segments based on Scores', 'value': 'segment_scores'},
            ],
            value='segment_distribution',
            placeholder='Select a chart to display',
            className="mb-4",
            style={'fontFamily': 'Arial', 'border': '1px solid #ced4da', 'padding': '5px', 'width': '100%', 'margin': 'auto'}
        ),
        html.Div([
            dcc.Graph(id='rfm-chart', className="mb-4", style={'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '10px'})
        ])
    ], style={'padding': '20px'})
])

@app.callback(
    Output('rfm-chart', 'figure'),
    [Input('chart-type-dropdown', 'value')]
)
def update_chart(selected_chart_type):
    if selected_chart_type == 'segment_distribution':
        return fig_segment_dist
    elif selected_chart_type == 'RFM_distribution':
        return fig_treemap_segment_product
    elif selected_chart_type == "correlation_matrix":
        return fig_corr_heatmap
    elif selected_chart_type == 'segment_comparison':
        return comparision_fig
    elif selected_chart_type == 'segment_scores':
        return fig
    return fig_segment_dist

if __name__ == '__main__':
    app.run_server(port=8055)
