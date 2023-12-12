# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 01:27:06 2023

@author: melvi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings



def worldbank_data(path):
    """
    Read World Bank data from a CSV file.

    Parameters:
    - path (str): The file path of the CSV file containing World Bank data.

    Returns:
    - Tuple: A tuple containing two elements:
        1. DataFrame: Transposed DataFrame with 'Country Name' as index.
        2. DataFrame: Raw DataFrame with 'Country Name' as index (no transposition).
    """
    return pd.read_csv(path, skiprows=4).set_index(['Country Name']).T, pd.read_csv(path, skiprows=4)

years_data, countries_data = worldbank_data("API_19_DS2_en_csv_v2_6183479.csv")
years_data=years_data.head()
print(years_data)
countries_data=countries_data.head()
print(countries_data)
countries_data = worldbank_data("API_19_DS2_en_csv_v2_6183479.csv")[1].iloc[:, :-1].fillna(0)
print(countries_data)

"""
Print the unique values in the 'Indicator Name' column of the World Bank data
"""

print(countries_data['Indicator Name'].unique())

sub_df = countries_data[countries_data['Country Name'].isin(['India', 'United Kingdom', 'Australia', 'United States'])]
print(sub_df.head())
print(sub_df.shape)
print(sub_df.describe())


countries_to_plot = ['India', 'Australia', 'United Kingdom', 'United States']
indicators_to_plot = ['Population growth (annual %)']
filtered_df = sub_df[sub_df['Country Name'].isin(countries_to_plot) & sub_df['Indicator Name'].isin(indicators_to_plot)]
plt.figure(figsize=(10, 5))
sns.lineplot(x='variable', y='value', hue='Country Name', data=pd.melt(filtered_df, id_vars=['Country Name'], value_vars=sub_df.columns[34:]))
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Overall Population Growth by % each year')
plt.legend(title='Country Name')
plt.show()

"""
Bar plot

"""
countries_to_plot = ['India', 'Australia', 'United Kingdom', 'United States']
filtered_data = sub_df[(sub_df['Indicator Name'] == 'Methane emissions (kt of CO2 equivalent)') & (sub_df['Country Name'].isin(countries_to_plot))]
selected_columns = filtered_data.columns[34::10]
filtered_data.set_index('Country Name')[selected_columns].T.plot(kind='bar', figsize=(10, 6))
plt.title('Bar Plot for Methane Gas Emissions')
plt.xlabel('Years')
plt.ylabel('Emissions (kt of CO2 equivalent)')
plt.legend(title='Country Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

countries_to_plot = ['India', 'Australia', 'United Kingdom', 'United States']
filtered_data = sub_df[(sub_df['Indicator Name'] == 'Total greenhouse gas emissions (kt of CO2 equivalent)') & (sub_df['Country Name'].isin(countries_to_plot))]
selected_columns = filtered_data.columns[34::10]
filtered_data.set_index('Country Name')[selected_columns].T.plot(kind='bar', figsize=(10, 6))
plt.title('Bar Plot for Total greenhouse gas emissions')
plt.xlabel('Years')
plt.ylabel('Emissions (kt of CO2 equivalent)')
plt.legend(title='Country Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


countries_to_plot = ['India', 'Australia', 'United Kingdom', 'United States']
filtered_data = sub_df[(sub_df['Indicator Name'] == 'CO2 emissions (kt)') & (sub_df['Country Name'].isin(countries_to_plot))]
plt.figure(figsize=(10, 5))
sns.lineplot(x='variable', y='value', hue='Country Name', data=pd.melt(filtered_data, id_vars=['Country Name'], value_vars=sub_df.columns[34:-2]))
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('emissions (kt)')
plt.title('CO2 emissions (kt)')
plt.legend(title='Country Name')
plt.show()


indicators_to_plot = ['Population growth (annual %)', 'Methane emissions (kt of CO2 equivalent)', 'Total greenhouse gas emissions (kt of CO2 equivalent)',  'CO2 emissions (kt)']
indicator_names_mapping = {'Population growth (annual %)': 'Population Growth %', 'Methane emissions (kt of CO2 equivalent)': 'Methane Emissions',
                           'Total greenhouse gas emissions (kt of CO2 equivalent)': 'Total GHG Emissions', 'CO2 emissions (kt)': 'CO2 Emissions'}

plt.figure(figsize=(6,4))
Ind = sub_df[(sub_df['Country Name'] == 'India') & (sub_df['Indicator Name'].isin(indicators_to_plot))]
Ind['Indicator Name'] = Ind['Indicator Name'].map(indicator_names_mapping)
heatmap = Ind.set_index(['Indicator Name']).iloc[:, 4:].transpose()
sns.heatmap(heatmap.corr(), annot=True)
plt.title('India')
plt.show()

plt.figure(figsize=(6,4))
Ind = sub_df[(sub_df['Country Name'] == 'United Kingdom') & (sub_df['Indicator Name'].isin(indicators_to_plot))]
Ind['Indicator Name'] = Ind['Indicator Name'].map(indicator_names_mapping)
heatmap = Ind.set_index(['Indicator Name']).iloc[:, 4:].transpose()
sns.heatmap(heatmap.corr(), annot=True)
plt.title('United Kingdom')
plt.show()

plt.figure(figsize=(6,4))
Ind = sub_df[(sub_df['Country Name'] == 'United States') & (sub_df['Indicator Name'].isin(indicators_to_plot))]
Ind['Indicator Name'] = Ind['Indicator Name'].map(indicator_names_mapping)
heatmap = Ind.set_index(['Indicator Name']).iloc[:, 4:].transpose()
sns.heatmap(heatmap.corr(), annot=True)
plt.title('United States')
plt.show()

plt.figure(figsize=(6,4))
Ind = sub_df[(sub_df['Country Name'] == 'Australia') & (sub_df['Indicator Name'].isin(indicators_to_plot))]
Ind['Indicator Name'] = Ind['Indicator Name'].map(indicator_names_mapping)
heatmap = Ind.set_index(['Indicator Name']).iloc[:, 4:].transpose()
sns.heatmap(heatmap.corr(), annot=True)
plt.title('Australia')
plt.show()


