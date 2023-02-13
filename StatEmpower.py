import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("retail_sales.csv")

# Perform exploratory data analysis
print("\nData shape: ", df.shape)
print("\nFeatures: \n", df.columns)
print("\nMissing values: \n", df.isnull().sum())
print("\nDescriptive statistics: \n", df.describe())
sns.pairplot(df)
plt.show()

# Build a predictive model
X = df[["day_of_week", "promo", "school_holiday"]].values
y = df["sales"].values

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

print("\nCoefficients: \n", reg.coef_)
print("\nIntercept: \n", reg.intercept_)
print("\nMean squared error: %.2f" % mean_squared_error(y, y_pred))
print("\nR2 score: %.2f" % r2_score(y, y_pred))

# Make predictions
X_new = np.array([[3, 1, 0]]) # Predict sales for a promo day during the week with no school holiday
print("\nPredicted sales: %.2f" % reg.predict(X_new))