"""
 This program uses the Boston house-prices dataset, which consists of 13 features 
 of 506 houses and their median value in $1000's, and fits a model
 on these 13 features to predict a sample house price.
 """

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']

model = LinearRegression()
model.fit(x,y)

sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
                
prediction = model.predict(sample_house)
