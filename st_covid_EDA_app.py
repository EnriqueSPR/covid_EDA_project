# Import necessary libraries

import numpy as np
import streamlit as st
import pandas as pd
import base64
from PIL import Image
import plotly.express as px
from datetime import date, datetime, timedelta
import dateutil.relativedelta
from sklearn.impute import SimpleImputer 
from collections import OrderedDict 


# Layout
st.set_page_config(
    page_title="covid-data-tool",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="auto") 
col1 = st.sidebar


# Logo
image = Image.open('logo.png')
st.image(image, width = 700)

# Title
st.markdown("""## **An App to explore covid data**""")

# Description
st.markdown("""

**Description**: This app was built to study the evolution of the covid pandemia by generating visually appealing plots. 

""")

# About

expander_bar = st.beta_expander("About", expanded=True)
expander_bar.markdown("""
* **Python libraries used:** pandas, streamlit, collections, numpy, base64, PIL, datetime, sklearn and plotly.
* **Data**: [wordometers](https://www.worldometers.info/coronavirus/) and [our-world-in-data](https://covid.ourworldindata.org/data/owid-covid-data.csv).
* **Author**: [Enrique Alcalde](https://enriquespr.github.io/Enrique_portfolio_web/) 🙋🏽‍♂️.
---
""")

#LayoutPLOTS
col2, col3 = st.beta_columns((2,1))

# Load data at side bar

@st.cache(allow_output_mutation=True)
def load_df():
    covid_url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(covid_url, parse_dates=True)
    return df

# Load and prepare df
data_load_state = col2.info('Parsing most recent data...')
df = load_df()
data_load_state.success('Parsing most recent data... done!')
df["perc_death_cases"] = (df["total_deaths"]/df["total_cases"])*100 # Adding an extra param
df["perc_death_cases"] = df["perc_death_cases"].apply(lambda x: round(x,1))
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df.loc[df["location"]=="Northern Cyprus", "continent"] = "Europe" # dealing with some nan continent info missing...
df.loc[df["location"]=="International", "continent"] = "World"
df.loc[df["continent"].isnull(),"continent"] = df["location"]
df.loc[df["continent"]=="European Union", "continent"] = "Europe"
df = df.reindex(sorted(df.columns), axis=1) # reordering columns alphabetically

# Download File if u wish

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="all_data.csv">Download all the data in CSV format</a>'
    return href

def filedownloader(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="plot_data.csv">Download the table in CSV format</a>'
    return href

col2.markdown(filedownload(df), unsafe_allow_html=True)
col2.markdown("""---""")

# Sidebar - Inputs
col1.markdown("""
**Choose your custom covid analysis:** 
""")

countries_list = list(df["location"].unique())
europe_list = df.loc[(df["continent"]=="Europe")]["location"].unique()
asia_list = df.loc[(df["continent"]=="Asia")]["location"].unique()
africa_list = df.loc[(df["continent"]=="Africa")]["location"].unique()
n_america_list = df.loc[(df["continent"]=="North America")]["location"].unique()
s_america_list = df.loc[(df["continent"]=="South America")]["location"].unique()
oceania_list = df.loc[(df["continent"]=="Oceania")]["location"].unique()

countries = col1.beta_container()
all = col1.checkbox("Select all")
europe = col1.checkbox("Select European Countries")
north_america = col1.checkbox("Select North America")
south_america = col1.checkbox("Select South American Countries")
asia = col1.checkbox("Select Asian Countries")
africa = col1.checkbox("Select African Countries")
oceania = col1.checkbox("Select Oceania Countries")
 
if all:
    countries = countries.multiselect('Please, select the countries or continents you want to analyze:', countries_list, countries_list)
elif europe:
    countries = countries.multiselect('Please, select the countries or continents you want to analyze:', countries_list, europe_list)
elif north_america:
    countries = countries.multiselect('Please, select the countries or continents you want to analyze:', countries_list, n_america_list)
elif south_america:
    countries = countries.multiselect('Please, select the countries or continents you want to analyze:', countries_list, s_america_list)
elif asia:
    countries = countries.multiselect('Please, select the countries or continents you want to analyze:', countries_list, asia_list)
elif oceania:
    countries = countries.multiselect('Please, select the countries or continents you want to analyze:', countries_list, oceania_list)
else:
    countries =  countries.multiselect('Please, select the countries or continents you want to analyze:', countries_list, countries_list[27])


parameters_list = list(df.columns)
parameter_x = col1.selectbox('Please, select the parameter you want to incorporate in the X axis:', parameters_list)

parameters_list = list(df.columns)
parameter_y = col1.selectbox('Please, select the parameter you want to incorporate in the Y axis:', parameters_list)

moving_avg = list(range(1,31))
date_cum = col1.select_slider('If you selected date_line, please, specify the number of days moving average:', moving_avg)


# PLOT 1: Date line plot 

# Get date_df based on parameter(s) and countries


def date_line_plot():
    df_date = df.loc[(df["location"].isin(countries)), (["location", "date"]+[parameter_y])].reset_index(drop=True)
    df_date["date"] = pd.to_datetime(df_date["date"], format="%Y-%m-%d")

    param_list = []
    for j in countries:
        for i in range(len(df_date)):
            if df_date.loc[i,"location"]==j:
                param_cum = df_date.loc[(((df_date["date"] >= 
                (df_date.loc[i,"date"] - dateutil.relativedelta.relativedelta(days=date_cum)))
                &(df_date["date"] < df_date.loc[i,"date"]))&(df_date["location"]==j))][parameter_y].sum()
                date = df_date.loc[i,"date"]
                param_list.append((j, date, param_cum))

    df_param = pd.DataFrame(param_list, columns=["country", "date", parameter_y])
    title = 'Date-Line Plot: {} days moving average'.format(date_cum)

    fig = px.line(df_param, x="date", y=parameter_y, color='country', title=title, template="plotly_dark")
 
    col2.plotly_chart(fig, use_container_width=True)
    col3.markdown("""
       &nbsp;
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
       #### Table with the data used for the line plot:
        """)
    col3.table(df_param.head(3))
    col3.markdown(filedownloader(df_param), unsafe_allow_html=True)


if col2.button('Show Data-Line Plot'):
    date_line_plot()

col2.info("A linear graph that shows parameter frequencies along the date. Choose 'date' in the X axis and your parameter of choice followed by the number of days moving average on the left panel.")
col2.markdown("""---""")

# PLOT 2: Scatterplot of two parameters


dict_param = {"total_cases": "max", "new_cases": "median", "new_cases_smoothed": "median", "total_deaths": "max", "total_deaths_per_million":"max", "new_deaths":"median", "new_deaths_smoothed":"median",
                        "total_cases_per_million": "max", "new_cases_per_million": "median", "new_cases_smoothed_per_million": "median",
                        "new_deaths_smoothed_per_million":"median", "reproduction_rate": "median", 
                        "icu_patients": "median", "icu_patients_per_million": "median", "hosp_patients": "median", "hosp_patients_per_million": "median",
                        "weekly_icu_admissions": "median", "weekly_icu_admissions_per_million": "median", "weekly_hosp_admissions": "median", "weekly_hosp_admissions_per_million": "median",
                        "new_tests": "median", "total_tests": "max", "total_tests_per_thousand": "max", "new_tests_per_thousand": "median",
                        "new_tests_smoothed": "median", "new_tests_smoothed_per_thousand": "median", "positive_rate": "median", "tests_per_case": "median", 
                        "total_vaccinations": "max", "people_vaccinated": "median", "people_fully_vaccinated": "median", "new_vaccinations": "median",
                        "new_vaccinations_smoothed": "median", "total_vaccinations_per_hundred": "max", "people_vaccinated_per_hundred": "median", "people_fully_vaccinated_per_hundred": "median",
                        "new_vaccinations_smoothed_per_million": "median", "stringency_index": "mean", "population": "median", "population_density": "median", 
                        "median_age": "mean", "aged_65_older": "median", "aged_70_older": "median", "gdp_per_capita": "median", "extreme_poverty": "median",
                        "cardiovasc_death_rate": "median", "diabetes_prevalence": "mean", "female_smokers": "median", "male_smokers": "median", "handwashing_facilities": "median",
                        "hospital_beds_per_thousand": "median", "life_expectancy": "mean", "human_development_index": "median", "perc_death_cases": "median" }

dict_param = dict(OrderedDict(sorted(dict_param.items()))) # reorder dict alphabetically

list_cols_pivot = list(sorted(dict_param.keys()))

df_pivot = pd.pivot_table(data=df, index=["continent", "location"], 
              values=list_cols_pivot,
              aggfunc = dict_param).reset_index()

df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1) # reordering columns alphabetically

if len(countries) > 205:
            countries = list(df_pivot["location"].unique())


def scatter_plot():
    try:
        df_piv = df_pivot.loc[df_pivot["location"].isin(countries)]
        fig = px.scatter(df_piv,
                        title="Relactionship between {} and {} for each country".format(parameter_x, parameter_y),
                        x=parameter_x, 
                        y=parameter_y, 
                        hover_name = "location",
                        hover_data=["continent", "location", "population", "gdp_per_capita"],
                        color="continent",
                        template="plotly_dark") 
        fig.update_traces(marker=dict(size=15,
                                  line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
                        # width=700, height=500)

        col2.plotly_chart(fig, use_container_width=True)
        col3.markdown("""
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        #### Table with the data used for the scatter plot:
        """)
        col3.table(df_piv.head(3))
        col3.markdown(filedownloader(df_pivot), unsafe_allow_html=True)

    except Exception as e:
        col2.warning("This parameter combination was not valid or a selected country is not present in the pivot table (no data) ⚠.")

def scatter_plot_size():
    try:
        df_piv = df_pivot.loc[df_pivot["location"].isin(countries)]
        fig = px.scatter(df_piv,
        title="Relactionship between {} and {} for each country".format(parameter_x, parameter_y),
        x=parameter_x, 
        y=parameter_y, 
        size=df_piv[size_param],
        hover_name = "location",
        hover_data=["continent", "location", "population", "gdp_per_capita"],
        color="continent",
        template="plotly_dark") 
        # width=700, height=500)

        col2.plotly_chart(fig, use_container_width=True)
        col3.markdown("""
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        #### Table with the data used for the scatter plot:
        """)
        col3.table(df_piv.head(3))
        col3.markdown(filedownloader(df_piv), unsafe_allow_html=True)

    except Exception as e:
        col2.warning("This parameter combination was not valid or a selected country is not present in the pivot table (no data) ⚠.")


def impute_with_median(df):
    """Iterate through columns of Pandas DataFrame.
    Where NaNs exist replace with median"""
    
    # Get list of DataFrame column names
    cols = list(df)
    # Loop through columns
    for column in cols:
        # Transfer column to independent series
        col_data = df[column]
        # Look to see if there is any missing numerical data
        missing_data = sum(col_data.isna())
        if missing_data > 0:
            # Get median and replace missing numerical data with median
            col_median = col_data.median()
            col_data.fillna(col_median, inplace=True)
            df[column] = col_data
    return df  

if col2.button('Show Scatter Plot'):
    scatter_plot()

col2.info("Plot to study the relactionship between parameters. Choose a parameter for the X and Y axis and the list of countries to be included in the scatter plot.")

size = col2.checkbox("Note: Click here if you want to include a size option fot the scatter plot.")
if size:
    size_param = col2.selectbox('Please, select the parameter you want to use for the size:', parameters_list)
    df_pivot = impute_with_median(df_pivot)
    scatter_plot_size()

col2.markdown("""---""")


# PLOT 3: Bar Plot for a custom parameter (most recent data)


df_pivot_bar = pd.pivot_table(data=df, index=["continent", "location", "date"], 
              values=list_cols_pivot,
              aggfunc = dict_param).reset_index()

df_pivot_bar = df_pivot_bar.reindex(sorted(df_pivot_bar.columns), axis=1) # reordering columns alphabetically

dict_param.update(date = "max")
dict_param = dict(OrderedDict(sorted(dict_param.items()))) # order alph
list_cols_pivot = list(sorted(dict_param.keys()))
# shape (69363, 55)

# next i will get the earliest data shape (193, 55)
df_pivot_bar = df_pivot_bar.loc[(df_pivot_bar["date"] == (df_pivot_bar["date"].max()))].reset_index(drop=True)

def bar_plot():
    try:
        df_pivot_bar.sort_values(by=parameter_y, inplace=True)
        df_plot = df_pivot_bar.loc[df_pivot_bar["location"].isin(countries)].fillna(0)
        fig = px.bar(df_plot, x=df_plot["location"].unique(), y=parameter_y,
                hover_data=['continent', 'perc_death_cases'], color=parameter_y, template="plotly_dark",
                color_continuous_scale=px.colors.sequential.Viridis)

        col2.plotly_chart(fig, use_container_width=True)
        col3.markdown("""
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `   
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        #### Table with the data used for the bar plot:
        """)
        col3.table(df_pivot_bar.head(3))
        col3.markdown(filedownloader(df_pivot_bar), unsafe_allow_html=True)

    except Exception as e:
        col2.warning("This parameter combination was not valid or a selected country is not present in the pivot table (no data) ⚠.")

if col2.button('Show Bar Plot'):
    bar_plot()

col2.info("This plot displays the most recent data of a parameter (Y axis) of choice for each country (X_axis).")


def bar_plot_incidence():
    #try:
        inci = []
        for j in countries:
            df_c = df.loc[(df["location"]==j)]
            for i in range(len(df_c)):
                df_c = df_c.loc[(df_c["date"] >= (df_c["date"].max() - dateutil.relativedelta.relativedelta(days=incidence_d)))]
                param_cum = df_c["new_cases"].sum()
                popul = df_c["population"].mean()
                inci.append((j, (param_cum/popul)*100000))

        df_incidencia = pd.DataFrame(inci, columns = ["location", "incidence rate"])
        df_incidencia = pd.pivot_table(data=df_incidencia, index=["location"], 
              values=["location", "incidence rate"]).reset_index()
        df_incidencia = df_incidencia.sort_values(by="incidence rate").reset_index(drop=True)
        df_incidencia["incidence rate"] = round(df_incidencia["incidence rate"])

        fig = px.bar(df_incidencia, x=df_incidencia["location"].unique(), y="incidence rate",
                hover_data=['location'], color="location", template="plotly_dark",
                color_continuous_scale=px.colors.sequential.YlOrRd)

        col2.plotly_chart(fig, use_container_width=True)
        col3.markdown("""
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `   
        ` `  
        ` `  
        ` `   
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `   
        ` `  
        ` `  
        ` `
        ` `  
        ` `  
        ` `   
        ` `  
        ` `  
        ` `
        #### Table with the data used for the bar plot:
        """)
        col3.table(df_incidencia.head(3))
        col3.markdown(filedownloader(df_incidencia), unsafe_allow_html=True)

    #except Exception as e:
        #col2.warning("This parameter combination was not valid or a selected country is not present in the pivot table (no data) ⚠.")

incidence = col2.checkbox("Note, one of the most widely used epidemiological parameter is the indicence rate\
(new detected cases in a perdiod of time per 100.000 inhabitants). Click here to calculate the incidence rate for the desired countries \
 of choice.")

if incidence:
    incidence_days = list(range(1,31))
    incidence_d = col2.select_slider('Please, specify the number of days for calculating the incidence (Typically 14 days):', incidence_days, 14)
    bar_plot_incidence()

col2.markdown("""---""")


# 4 Map Plot
df_map = df.loc[~(df["location"].isin(["World", "International"]))].reset_index(drop=True)
df_map.sort_values(by=["date","location"], inplace=True)
df_map["date"] = df_map["date"].astype("str")

def map_plot():
    try:
        df_plot_map = df_map.loc[df_map["location"].isin(countries)]
        fig = px.choropleth(data_frame = df_plot_map,
                        locations= "iso_code",
                        locationmode = "ISO-3",
                        color = parameter_y, 
                        hover_name = "location",
                        color_continuous_scale= "plasma",
                        range_color = [0, df_plot_map[parameter_y].max()/2],
                        animation_frame= "date",
                       title='{}'.format(parameter_y))
        col2.plotly_chart(fig, use_container_width=True)
        col3.markdown("""
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        #### Table with the data used for the map plot:
        """)
        col3.table(df_pivot_bar.head(3))
        col3.markdown(filedownloader(df_pivot_bar), unsafe_allow_html=True)

    except Exception as e:
        col2.warning("This parameter combination was not valid or a selected country is not present in the pivot table (no data) ⚠.")

if col2.button('Show Map Plot'):
    map_plot()

col2.info("This plot displays the evolution overtime of a parameter (Y axis) on a map plot of the selected countries.")
col2.markdown("""---""")

# 5  Bubble Plot

df_bubble = df[["continent", "location","date","total_cases","total_deaths","total_cases_per_million", "total_deaths_per_million", "life_expectancy", "perc_death_cases", "total_vaccinations_per_hundred"]]
df_bubble.loc[df_bubble["continent"].isnull(),"continent"] = df_bubble["location"]
populated_countries_list = list(df.loc[df["population"]>4000000, "location"].unique())
df_bubble = df_bubble.loc[df_bubble["location"].isin(populated_countries_list)].reset_index(drop=True)
df_bubble['total_vaccinations_per_hundred'].fillna((df_bubble['total_vaccinations_per_hundred'].mean()), inplace=True)
df_bubble = impute_with_median(df_bubble)
df_bubble["date"] = pd.to_datetime(df_bubble["date"], format="%Y-%m-%d")
df_bubble.sort_values(by=["continent","date"], inplace=True) # IMPORTANT TO SORT CONTINENT ALSO
df_bubble = df_bubble.loc[~(df_bubble["date"] <= "2020-02-05")].reset_index(drop=True) # ealier dates not sorted. 😐?
df_bubble["date"] = df_bubble["date"].astype("str")
xmin, xmax = 1, max(df_bubble["total_cases_per_million"])
ymin, ymax = 1, max(df_bubble["total_deaths_per_million"])

def bubble_plot():
    try:
        fig = px.scatter(df_bubble, x="total_cases_per_million", y="total_deaths_per_million", animation_frame="date", animation_group="location",
        color="continent", hover_name="location", width=800, height=600, size = "total_vaccinations_per_hundred", size_max=50,
        range_x=[xmin,xmax], range_y=[ymin,ymax], template= "plotly_dark")

        fig.update_traces(marker=dict(size=10,
                                          line=dict(width=2, color='white')), selector=dict(mode='markers'))

        col2.plotly_chart(fig, use_container_width=True)
        col3.markdown("""
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        ` `  
        #### Table with the data used for the bubble plot:
        """)
        col3.table(df_bubble.head(3))
        col3.markdown(filedownloader(df_bubble), unsafe_allow_html=True)

    except Exception as e:
        col2.warning("This parameter combination was not valid or a selected country is not present in the pivot table (no data) ⚠.")

if col2.button('Show Bubble Plot'):
    bubble_plot()

col2.info("This is a nice plot to observe the evolution of the pandemia. \
It plots the total_cases_per_million (X axis) vs the total_deaths_per_million overtime for each country. \
Note: To make this plot only countries with a population higher than more 4 million were included.")
col2.markdown("""---""")

# 6 Bubble Plot By Continent

def bubble_plot_continent():
    try:
        fig = px.scatter(df_bubble, x="total_cases_per_million", y="total_deaths_per_million", animation_frame="date", animation_group="location",
        color="continent", hover_name="location", width=1400, height=400, size = "total_vaccinations_per_hundred", size_max=50, facet_col="continent",
        range_x=[xmin,xmax], range_y=[ymin,ymax], template= "plotly_dark")

        fig.update_traces(marker=dict(size=10,
                                          line=dict(width=2, color='white')), selector=dict(mode='markers'))

        st.plotly_chart(fig, use_container_width=True)
 
    except Exception as e:
        col2.warning("This parameter combination was not valid or a selected country is not present in the pivot table (no data) ⚠.")

if col2.button('Show Bubble Plot by Continent'):
    bubble_plot_continent()

col2.info("In this case the bubble plot is showed individually for each continent for better comparisson.")
col2.markdown("""---""")