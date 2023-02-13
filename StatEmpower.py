import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

print("\nLinear Regression Results:")
print("Coefficients: \n", reg.coef_) # print the coefficients of the linear regression model
print("Intercept: \n", reg.intercept_) # print the intercept of the linear regression model
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred)) # calculate and print the mean squared error between the true sales values and the predicted values
print("R2 score: %.2f" % r2_score(y, y_pred)) # calculate and print the R^2 score between the true sales values and the predicted values

# Build a decision tree regression model
dtr = DecisionTreeRegressor().fit(X, y)
y_pred_dtr = dtr.predict(X)

print("\nDecision Tree Regression Results:")
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred_dtr)) # calculate and print the mean squared error between the true sales values and the predicted values
print("R2 score: %.2f" % r2_score(y, y_pred_dtr)) # calculate and print the R^2 score between the true sales values and the predicted values

# Make predictions
X_new = np.array([[3, 1, 0]]) # Predict sales for a promo day during the week with no school holiday
print("\nPredicted sales (linear regression): %.2f" % reg.predict(X_new))
print("Predicted sales (decision tree regression): %.2f" % dtr.predict(X_new))
