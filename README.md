# Project Introduction

Final Project for CAPP 30254 - Machine  Learning for Public Policy. Completed by Dominic Teo, Eunjung Choi and Ya-Han Cheng.

Our project is titled **“Predicting whether a violent crime is likely to occur in Chicago based on past reported crime data, socio-economic indicators and weather data”**. This repository will contain both the relevant datasets that we use as well as the Jupyter Notebooks that we use to clean the data and also to create the various Machine Learning models that we use. 

# Datasets 
There are 4 different datasets that we use but only 3 datasets can be found in our repository, under the ‘Datasets’ folder, with the 4th one being downloaded in our Jupyter Notebook and they are titled as follows:

1. Reported crimes dataset (source: City of Chicago data portal): The file is too big to push into our repo. However, it can be easily downloaded at https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2. We utilized the 2015 data to present. 
2. Chicago boundaries neighbourhood dataset: "Boundaries-Neighborhoods.geojson"
3. Weather dataset (source: NOAA): “2162082.csv”

# Data Cleaning
The data cleaning steps that are necessary in order to create the csv dataset to be used in creating and testing the different machine learning models. The Jupyter Notebook file for this is titled “data_cleaning.ipynb”. 

However, we also provided the cleaned csv files that can be used directly without having to run through the data cleaning process. The full csv (around 1.2 million rows) is unfortunately too big to be pushed to our repository so we instead created a randomly sampled csv (10% of ‘crime.csv’), titled “crime_reduced.csv”. These csv files can then be used to run the other Jupyter Notebooks meant to create and evaluate the different machine learning models. 


# How to run codes 
All our codes for building, training and testing models are in the Jupyter Notebook files. The codes for running decision tree classifier, random forest classifier, and ada-boost classifier are in the file “dt_rf_ab.ipynb”, and the codes for running logistic regression model are in “logistic_regression.py”.  
