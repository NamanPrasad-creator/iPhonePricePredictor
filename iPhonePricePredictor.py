# Import necessary libraries
import pandas                      # Library for data manipulation
import matplotlib.pyplot as plt    # Library for data visualization
from sklearn.linear_model import LinearRegression  # Linear regression model for prediction

# Load the dataset containing iPhone version and price information
data = pandas.read_csv('iphone_price.csv')

# Plot the data to visualize the relationship between iPhone version and price
plt.scatter(data['version'], data['price'])
plt.xlabel("iPhone Version")
plt.ylabel("Price")
plt.title("iPhone Price vs Version")
plt.show()

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the data, using 'version' as the feature and 'price' as the target
model.fit(data[['version']], data[['price']])

# Predict the price for iPhone version 30
print(model.predict([[30]]))   # Output predicted price for iPhone version 30
