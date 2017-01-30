"""
This program uses linear regression to predict life expectancy in a
particular country based on BMI from birth

In this version, we are predicting the life expectancy for Laos from 
a BMI of 21.07931. Data for other countries can be found in 
bmi_and_life_expectancy.csv
"""

import pandas as pd 
from sklearn.linear_model import LinearRegression

bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

bmi_life_model = LinearRegression()
bmi_life_model.fit(x_values, y_values)

laos_life_exp = bmi_life_model.predict(21.07931)
