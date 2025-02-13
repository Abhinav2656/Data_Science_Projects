# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from setuptools.command.rotate import rotate
# from scipy.optimize import curve_fit
# import numpy as np
#
# from Other_Tkinter_Widgets import label
#
# ev_data = pd.read_csv("Electric_Vehicle_Population_Data.csv")
# ev_data = ev_data.dropna()
#
# sns.set_style("whitegrid")
#
# # EV Adoption Over Time
# plt.figure(figsize=(12,6))
# ev_adoption_by_year = ev_data['Model Year'].value_counts().sort_index()
# sns.barplot(x=ev_adoption_by_year.index, y=ev_adoption_by_year.values, palette="viridis")
# plt.title('EV Adoption Over Time')
# plt.xlabel('Model Year')
# plt.ylabel('Number of Vehicles Registered')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
#
# # geographical distribution at county level
# ev_county_distribution = ev_data['County'].value_counts()
# top_countries = ev_county_distribution.head(3).index
#
# # filtering dataset for top countries
# top_counties_data = ev_data[ev_data['County'].isin(top_countries)]
#
# ev_city_distribution_top_counties = top_counties_data.groupby(['County', 'City']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
#
# top_cities = ev_city_distribution_top_counties.head(10)
#
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Number of Vehicles', y='City', hue='County', data=top_cities, palette="magma")
# plt.title('Top Cities in Top Counties by EV Registrations')
# plt.xlabel('Number of Vehicles Registered')
# plt.ylabel('City')
# plt.legend(title='County')
# plt.tight_layout()
# plt.show()
#
#
# # analyzing distribution of electric vehicle types
# ev_type_distribution = ev_data['Electric Vehicle Type'].value_counts()
#
# plt.figure(figsize=(10,6))
# sns.barplot(x=ev_type_distribution.values, y=ev_type_distribution.index, palette='rocket')
# plt.title('Distribution of Electric Vehicle Type')
# plt.xlabel('Number of Vehicle Type')
# plt.tight_layout()
# plt.show()
#
#
# # analyzing popularity of EV Manufacturers
# ev_make_distribution = ev_data['Make'].value_counts().head(10)
#
# plt.figure(figsize=(12,6))
# sns.barplot(x=ev_make_distribution.values, y=ev_type_distribution.index, palette="cube helix")
# plt.title('Top 10 Popular EV Makes')
# plt.xlabel('Number of Vehicles Registered')
# plt.ylabel('Make')
# plt.tight_layout()
# plt.show()
#
#
# # selecting top 3 manufacturers based on number of vehicles registered
# top_3_makes = ev_make_distribution.head(3).index
# top_makes_data = ev_data[ev_data['Make'].isin(top_3_makes)]
# ev_model_distribution_top_makes = top_makes_data.groupby(['Make', 'Model']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
# top_models = ev_model_distribution_top_makes.head(10)
#
# plt.figure(figsize=(12,8))
# sns.barplot(x='Number of Vehicles', y='Model', hue='Make', data=top_models, palette='viridis')
# plt.title('Top Models in Top 3 Makes by EV Registrations')
# plt.xlabel('Number of Vehicles Registered')
# plt.ylabel('Model')
# plt.tight_layout()
# plt.show()
#
#
# # analyzing distribution of electric range
# plt.figure(figsize=(12,6))
# sns.histplot(ev_data['Electric Range'], bins=30, kde=True, color='royalblue')
# plt.title('Distribution of Electric Vehicle Ranges')
# plt.xlabel('Electric Range(miles)')
# plt.ylabel('Number of Vehicles')
# plt.axvline(ev_data['Electric Range'].mean(), color='red', linestyle='--', label=f"Mean Range: {ev_data['Electric Range'].mean():.2f} miles")
# plt.legend()
# plt.show()
#
#
# # calculating average electric range by model year
# average_range_by_year = ev_data.groupby('Model Year')['Electric Range'].mean().reset_index()
# plt.figure(figsize=(12, 6))
# sns.lineplot(x='Model Year', y='Electric Range', data=average_range_by_year, marker='o', color='green')
# plt.title('Average Electric Range by Model Year')
# plt.xlabel('Model Year')
# plt.ylabel('Average Electric Range (miles)')
# plt.grid(True)
# plt.show()
#
#
# # top 10 models with highest average electric range
# average_range_by_model = top_makes_data.groupby(['Make', 'Model'])['Electric Range'].mean().sort_values(ascending=False).reset_index()
#
# top_range_models = average_range_by_model.head(10)
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Electric Range', y='Model', hue='Make', data=top_range_models, palette='cool')
# plt.title('Top 10 Models by Average Electric Range in Top Makes')
# plt.xlabel('Average Electric Range (miles)')
# plt.ylabel('Model')
# plt.legend(title='Make', loc='center right')
# plt.show()
#
#
# # calculate number of EVs registered each year in US
# ev_registration_counts = ev_data['Model Year'].value_counts().sort_index()
#
# filtered_years = ev_registration_counts[ev_registration_counts.index <= 2023]
#
#
# def exp_growth(x, a, b):
#     return a * np.exp(b * x)
#
#
# x_data = filtered_years.index - filtered_years.index.min()
# y_data = filtered_years.values
#
# params, covariance = curve_fit(exp_growth, x_data, y_data)
#
# forecast_years = np.arange(2024, 2024 + 6) - filtered_years.index.min()
# forecast_values = exp_growth(forecast_years, *params)
#
# forecasted_evs = dict(zip(forecast_years + filtered_years.index.min(), forecast_values))
#
# # plotting
# years = np.arange(filtered_years.index.min(), 2029 +1)
# actual_years = filtered_years.index
# forecast_years_full = np.arange(2024, 2029+1)
#
# actual_values = filtered_years.values
# forecasted_values_full = [forecasted_evs[year] for year in forecast_years_full]
#
# plt.figure(figsize=(12,8))
# plt.plot(actual_years, actual_values, 'bo-', label='Actual Registrations')
# plt.plot(forecast_years_full, forecasted_values_full, 'ro--', label='Forecasted Registrations')
#
# plt.title('Current & Estimated EV Market')
# plt.xlabel('Year')
# plt.ylabel('Number of EV Registrations')
# plt.legend()
# plt.grid(True)
# plt.show()
#
#


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np

# Load dataset
ev_data = pd.read_csv("Electric_Vehicle_Population_Data.csv")
ev_data = ev_data.dropna()

sns.set_style("whitegrid")

# EV Adoption Over Time
plt.figure(figsize=(12,6))
ev_adoption_by_year = ev_data['Model Year'].value_counts().sort_index()
sns.barplot(x=ev_adoption_by_year.index, y=ev_adoption_by_year.values, palette="viridis")
plt.title('EV Adoption Over Time')
plt.xlabel('Model Year')
plt.ylabel('Number of Vehicles Registered')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Geographical Distribution at County Level
ev_county_distribution = ev_data['County'].value_counts()
top_counties = ev_county_distribution.head(3).index

top_counties_data = ev_data[ev_data['County'].isin(top_counties)]
ev_city_distribution_top_counties = top_counties_data.groupby(['County', 'City']).size().reset_index(name='Number of Vehicles')

top_cities = ev_city_distribution_top_counties.head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Number of Vehicles', y='City', hue='County', data=top_cities, palette="magma")
plt.title('Top Cities in Top Counties by EV Registrations')
plt.xlabel('Number of Vehicles Registered')
plt.ylabel('City')
plt.legend(title='County')
plt.tight_layout()
plt.show()

# Distribution of Electric Vehicle Types
ev_type_distribution = ev_data['Electric Vehicle Type'].value_counts()

plt.figure(figsize=(10,6))
sns.barplot(x=ev_type_distribution.values, y=ev_type_distribution.index, palette='rocket')
plt.title('Distribution of Electric Vehicle Type')
plt.xlabel('Number of Vehicles')
plt.tight_layout()
plt.show()

# Popular EV Manufacturers
ev_make_distribution = ev_data['Make'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.barplot(x=ev_make_distribution.values, y=ev_make_distribution.index, palette="cubehelix")  # Fixed
plt.title('Top 10 Popular EV Makes')
plt.xlabel('Number of Vehicles Registered')
plt.ylabel('Make')
plt.tight_layout()
plt.show()

# Top Models from Top 3 Manufacturers
top_3_makes = ev_make_distribution.head(3).index
top_makes_data = ev_data[ev_data['Make'].isin(top_3_makes)]
ev_model_distribution_top_makes = top_makes_data.groupby(['Make', 'Model']).size().reset_index(name='Number of Vehicles')

top_models = ev_model_distribution_top_makes.head(10)

plt.figure(figsize=(12,8))
sns.barplot(x='Number of Vehicles', y='Model', hue='Make', data=top_models, palette='viridis')
plt.title('Top Models in Top 3 Makes by EV Registrations')
plt.xlabel('Number of Vehicles Registered')
plt.ylabel('Model')
plt.tight_layout()
plt.show()

# Distribution of Electric Range
plt.figure(figsize=(12,6))
sns.histplot(ev_data['Electric Range'], bins=30, kde=True, color='royalblue')
plt.title('Distribution of Electric Vehicle Ranges')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Number of Vehicles')
plt.axvline(ev_data['Electric Range'].mean(), color='red', linestyle='--', label=f"Mean Range: {ev_data['Electric Range'].mean():.2f} miles")
plt.legend()
plt.show()

# Average Electric Range by Model Year
average_range_by_year = ev_data.groupby('Model Year')['Electric Range'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='Model Year', y='Electric Range', data=average_range_by_year, marker='o', color='green')
plt.title('Average Electric Range by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average Electric Range (miles)')
plt.grid(True)
plt.show()

# Top 10 Models by Average Electric Range
average_range_by_model = top_makes_data.groupby(['Make', 'Model'])['Electric Range'].mean().sort_values(ascending=False).reset_index()
top_range_models = average_range_by_model.head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Electric Range', y='Model', hue='Make', data=top_range_models, palette='cool')
plt.title('Top 10 Models by Average Electric Range in Top Makes')
plt.xlabel('Average Electric Range (miles)')
plt.ylabel('Model')
plt.legend(title='Make', loc='center right')
plt.show()

# Forecasting EV Registrations
ev_registration_counts = ev_data['Model Year'].value_counts().sort_index()

filtered_years = ev_registration_counts.loc[ev_registration_counts.index <= 2023]  # Fixed

def exp_growth(x, a, b):
    return a * np.exp(b * x)

x_data = filtered_years.index - filtered_years.index.min()
y_data = filtered_years.values

params, _ = curve_fit(exp_growth, x_data, y_data)

forecast_years = np.arange(2024, 2024 + 6) - filtered_years.index.min()
forecast_values = exp_growth(forecast_years, *params)

forecasted_evs = dict(zip(forecast_years + filtered_years.index.min(), forecast_values))

# Plotting Forecast
years = np.arange(filtered_years.index.min(), 2029 +1)
actual_years = filtered_years.index
forecast_years_full = np.arange(2024, 2029+1)

actual_values = filtered_years.values
forecasted_values_full = [forecasted_evs.get(year, 0) for year in forecast_years_full]  # Fixed

plt.figure(figsize=(12,8))
plt.plot(actual_years, actual_values, 'bo-', label='Actual Registrations')
plt.plot(forecast_years_full, forecasted_values_full, 'ro--', label='Forecasted Registrations')

plt.title('Current & Estimated EV Market')
plt.xlabel('Year')
plt.ylabel('Number of EV Registrations')
plt.legend()
plt.grid(True)
plt.show()

