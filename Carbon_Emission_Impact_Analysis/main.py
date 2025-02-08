import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


temperature_data = pd.read_csv('temperature.csv')
co2_data = pd.read_csv('carbon_emmission.csv')

temperature_values = temperature_data.filter(regex='^F').stack()
temperature_stats = {
    "Mean": temperature_values.mean(),
    "Median": temperature_values.median(),
    "Variance": temperature_values.var()
}

co2_values = co2_data["Value"]
co2_stats = {
    "Mean": co2_values.mean(),
    "Median": co2_values.median(),
    "Variance": co2_values.var()
}


temperature_years = temperature_data.filter(regex='^F').mean(axis=0)
temperature_years.index = temperature_years.index.str.replace('F', '').astype(int)

co2_data['Year'] = co2_data['Date'].str[:4].astype(int)
co2_yearly = co2_data.groupby('Year')['Value'].mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=temperature_years.index, y=temperature_years.values,
    mode='lines+markers', name="Temperature Change(C)"
))

fig.add_trace(go.Scatter(
    x=co2_yearly.index, y=co2_yearly.values,
    mode='lines+markers', name="CO2 Concentration (ppm)", line=dict(dash='dash')
))

fig.update_layout(
    title='Time-Series of Temperature Change and CO2 Concentrations',
    xaxis_title='Year', yaxis_title='Values',
    template='plotly_white', legend_title='Metrics'
)
fig.show()


#correlation heatmap
merged_data = pd.DataFrame({
    "Temperature Change": temperature_years,
    "CO2 Concentration": co2_yearly
}).dropna()

heatmap_fig = px.imshow(
    merged_data.corr(), text_auto='.2f',
    color_continuous_scale='RdBu',
    title='Correlation Heatmap'
)
heatmap_fig.update_layout(
    template='plotly_white'
)

heatmap_fig.show()


#Scatter Plot: temperature vs CO2 concentrations
scatter_fig = px.scatter(
    merged_data, x='CO2 Concentration', y='Temperature Change',
    labels={"CO2 Concentration": "CO2 Concentration (PPM)", "Temperature Change": "Temperature Change (C)"},
    title='Temperature Change VS CO2 Concentration',
    template='plotly_white'
)

scatter_fig.update_traces(marker=dict(size=10, opacity=0.7))
scatter_fig.show()

#temperature trend
temp_trend = linregress(temperature_years.index, temperature_years.values)
temp_trend_line = temp_trend.slope * temperature_years.index + temp_trend.intercept

# CO2 trend
co2_trend = linregress(co2_yearly.index, co2_yearly.values)
co2_trend_line = co2_trend.slope * co2_yearly.index + co2_trend.intercept

fig_trends = go.Figure()

fig_trends.add_trace(go.Scatter(
    x=temperature_years.index, y=temperature_years.values,
    mode='lines+markers', name='Temperature Change (C)'
))
fig_trends.add_trace(go.Scatter(
    x=temperature_years.index, y=temp_trend_line,
    mode='lines', name=f'Temperature Change (Slope: {temp_trend.slope:.2f})', line=dict(dash='dash')
))
fig_trends.add_trace(go.Scatter(
    x=co2_yearly.index, y=co2_yearly.values,
    mode='lines+markers', name='CO2 Centration (ppm)'
))
fig_trends.add_trace(go.Scatter(
    x=co2_yearly.index, y=co2_trend_line,
    mode='lines', name=f'CO2 Trend (Slope: {co2_trend.slope:.2f})', line=dict(dash='dash')
))

fig_trends.update_layout(
    title='Trends in Temperature Change and CO2 Concentration',
    xaxis_title='Year', yaxis_title='Values',
    template='plotly_white', legend_title='Metrics'
)
fig_trends.show()


#seasonal variations in CO2 concentrations
co2_data['Month'] = co2_data['Date'].str[-2:].astype(int)
co2_monthly = co2_data.groupby('Month')['Value'].mean()

fig_seasonal = px.line(
    co2_monthly,
    x=co2_monthly.index, y=co2_monthly.values,
    labels={'x': 'Month', 'y': 'Co2 Concentration (ppm)'},
    title='Seasonal variations in Co2 Concentrations', markers=True
)

fig_seasonal.update_layout(
    xaxis=dict(tickmode='array', tickvals=list(range(1, 13))),
    template='plotly_white'
)
fig_seasonal.show()


#pearson and spearman correlation coefficients
pearson_corr, _ = pearsonr(merged_data['CO2 Concentration'], merged_data['Temperature Change'])
spearman_corr, _ = spearmanr(merged_data['CO2 Concentration'], merged_data['Temperature Change'])

#granger causality test
granger_data = merged_data.diff().dropna()
granger_results = grangercausalitytests(granger_data, maxlag=3, verbose=False)

granger_p_values = {f"Lag {lag}": round(results[0]['ssr_chit2test'][1], 4)
                    for lag, results in granger_results.items()}


#creating lagged CO2 data to investigate lagged effects
merged_data['CO2 Lag 1'] = merged_data['CO2 Concentration'].shift(1)
merged_data['Co2 Lag 2'] = merged_data['CO2 Concentration'].shift(2)
merged_data['Co2 Lag 3'] = merged_data['CO2 Concentration'].shift(3)

lagged_data = merged_data.dropna()

X = lagged_data[['CO2 Concentration', 'CO2 Lag 1', 'CO2 Lag 2', 'CO2 Lag 3']]
Y = lagged_data['Temperature Change ']
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
model_summary = model.summary()


# K-Means clustering
clustering_data = merged_data[['Temperature Change', 'CO2 Concentration']].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

kmeans = KMeans(n_clusters=3, random_state=42)
clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)
clustering_data['Label'] = clustering_data['Cluster'].map({
    0: 'Moderate Temp & CO2', 1: 'High Temp & CO2', 2: 'Low Temp & CO2'
})

fig_clusters = px.scatter(
    clustering_data,
    x='CO2 Concentration', y='Temperature Change',
    color='Label', color_discrete_sequence=px.colors.qualitative.Set2,
    labels={
        'CO2 Concentration': 'CO2 Concentration (ppm)',
        'Temperature Change': 'Temperature Change (C)',
        'Label': 'Climate Pattern'
    },
    title='Clustering of Years Based on Climate Patterns'
)

fig_clusters.update_layout(
    template='plotly_white', legend_title='Climate Pattern'
)
fig_clusters.show()


#simple predictive model using linear regression
X = merged_data[['CO2 Concentration']].values
Y = merged_data['Temperature Change'].values

model = LinearRegression()
model.fit(X,Y)

#function to simulate all "What -If" scenarios
def simulate_temperature_change(co2_percentage_change):
    #new CO2 Concentration
    current_mean_co2 = merged_data['CO2 Concentration'].mean()
    new_co2 = current_mean_co2 * (1 + co2_percentage_change / 100)

    #predict temperature change
    predicted_temp = model.predict([[new_co2]])
    return predicted_temp[0]

#simulating scenarios
scenarios = {
    "Increase CO2 by 10%": simulate_temperature_change(10),
    "Decrease CO2 by 10%": simulate_temperature_change(-10),
    "Increase CO2 by 20%": simulate_temperature_change(20),
    "Decrease CO2 by 20%": simulate_temperature_change(-20),
}
print(scenarios)