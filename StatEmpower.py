import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV # import GridSearchCV for hyperparameter tuning

# Load the dataset
df = pd.read_csv("retail_sales.csv")

# Perform exploratory data analysis
print("\nData shape: ", df.shape) # print the number of rows and columns in the dataset
print("\nFeatures: \n", df.columns) # print the column names of the dataset
print("\nMissing values: \n", df.isnull().sum()) # print the number of missing values in each column
print("\nDescriptive statistics: \n", df.describe()) # generate summary statistics for the numerical features in the dataset
sns.pairplot(df) # create a pairplot to visualize the relationships between features
plt.show()

# Build a linear regression model
X = df[["day_of_week", "promo", "school_holiday"]].values
y = df["sales"].values

# Hyperparameter tuning for linear regression
param_grid = {'fit_intercept': [True, False], # try different values for the fit_intercept parameter
              'normalize': [True, False]} # try different values for the normalize parameter

reg = GridSearchCV(LinearRegression(), param_grid, cv=5) # create a grid search object with 5-fold cross-validation
reg.fit(X, y) # fit the grid search object to the data

y_pred = reg.predict(X)

print("\nLinear Regression Results:")
print("Best parameters:", reg.best_params_) # print the best hyperparameters found by the grid search
print("Coefficients: \n", reg.best_estimator_.coef_) # print the coefficients of the linear regression model
print("Intercept: \n", reg.best_estimator_.intercept_) # print the intercept of the linear regression model
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred)) # calculate and print the mean squared error between the true sales values and the predicted values
print("R2 score: %.2f" % r2_score(y, y_pred)) # calculate and print the R^2 score between the true sales values and the predicted values

# Build a decision tree regression model
# Hyperparameter tuning for decision tree regression
param_grid_dtr = {'criterion': ['mse', 'friedman_mse'], # try different values for the criterion parameter
                  'splitter': ['best', 'random'], # try different values for the splitter parameter
                  'max_depth': [3, 5, 7, None]} # try different values for the max_depth parameter

dtr = GridSearchCV(DecisionTreeRegressor(), param_grid_dtr, cv=5) # create a grid search object with 5-fold cross-valid

