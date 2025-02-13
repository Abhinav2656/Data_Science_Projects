import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from wordcloud import WordCloud
import numpy as np

pio.templates.default = "plotly_white"

# Load Dataset
data = pd.read_csv("Instagram data.csv", encoding='latin-1')

# Convert Date to Datetime Format and Set Index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

# Distribution of Impressions
fig = px.histogram(
    data, x='Impressions',
    nbins=10, title='Distribution of Impressions'
)
fig.show()

# Impressions Over Time
fig = px.line(data, x=data.index, y='Impressions', title='Impressions Over Time')
fig.show()

# Metrics Over Time
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Likes'], name='Likes'))
fig.add_trace(go.Scatter(x=data.index, y=data['Saves'], name='Saves'))
fig.add_trace(go.Scatter(x=data.index, y=data['Follows'], name='Follows'))

fig.update_layout(title='Metrics Over Time', xaxis_title='Date', yaxis_title='Count')
fig.show()

# Reach from Different Sources
reach_sources = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
reach_counts = [data[source].sum() for source in reach_sources if source in data.columns]

colors = ['#FFB6C1', '#87CEFA', '#90EE90', '#FFDAB9']

fig = px.pie(data_frame=data, names=reach_sources, values=reach_counts,
             title='Reach from Different Sources', color_discrete_sequence=colors)
fig.show()

# Engagement Sources
engagement_metrics = ['Saves', 'Comments', 'Shares', 'Likes']
engagement_counts = [data[metric].sum() for metric in engagement_metrics if metric in data.columns]

colors = ['#FFB6C1', '#3E8E41', '#C6F4D6', '#FFC394']

fig = px.pie(data_frame=data, names=engagement_metrics, values=engagement_counts,
             title='Engagement Sources', color_discrete_sequence=colors)
fig.show()

# Profile Visits vs Follows
fig = px.scatter(data, x='Profile Visits', y='Follows', trendline='ols', title='Profile Visits vs Follows')
fig.show()

# Hashtags Word Cloud(Not Showing)
hashtags = ' '.join(data['Hashtags'].dropna().astype(str))
wordcloud = WordCloud(background_color="white").generate(hashtags)

# Convert WordCloud to Image Format for Plotly
fig = go.Figure(go.Image(z=np.array(wordcloud)))
fig.update_layout(title='Hashtags Word Cloud')
fig.show()

# Correlation Matrix(Not Showing)
corr_matrix = data.corr()

fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns,
                                y=corr_matrix.index, colorscale='RdBu',
                                zmin=-1, zmax=1))
fig.update_layout(title='Correlation Matrix', xaxis_title='Features', yaxis_title='Features')
fig.show()

# Distribution of Hashtags(Not Showing)
all_hashtags = []
data['Hashtags'] = data['Hashtags'].fillna('')  # Fix missing values

for row in data['Hashtags']:
    hashtags = row.split()
    hashtags = [tag.strip() for tag in hashtags]
    all_hashtags.extend(hashtags)

hashtag_distribution = pd.Series(all_hashtags).value_counts().reset_index()
hashtag_distribution.columns = ['Hashtag', 'Count']

fig = px.bar(hashtag_distribution.head(20), x='Hashtag', y='Count', title='Top 20 Hashtags Distribution')
fig.show()

# Likes and Impressions Distribution for Each Hashtag(Not Showing)
hashtag_likes = {}
hashtag_impressions = {}

for _, row in data.iterrows():
    hashtags = str(row['Hashtags']).split()
    for hashtag in hashtags:
        hashtag = hashtag.strip()
        if hashtag not in hashtag_likes:
            hashtag_likes[hashtag] = 0
            hashtag_impressions[hashtag] = 0
        hashtag_likes[hashtag] += row['Likes']
        hashtag_impressions[hashtag] += row['Impressions']

likes_distribution = pd.DataFrame(list(hashtag_likes.items()), columns=['Hashtag', 'Likes'])
impressions_distribution = pd.DataFrame(list(hashtag_impressions.items()), columns=['Hashtag', 'Impressions'])

fig_likes = px.bar(likes_distribution.head(20), x='Hashtag', y='Likes', title='Top 20 Hashtags by Likes')
fig_impressions = px.bar(impressions_distribution.head(20), x='Hashtag', y='Impressions', title='Top 20 Hashtags by Impressions')

fig_likes.show()
fig_impressions.show()
