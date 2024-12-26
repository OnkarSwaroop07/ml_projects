import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Creating dataset
hours = np.array([10, 7, 5, 11, 14, 3, 12, 8, 9]).reshape(-1, 1)
marks = np.array([90, 85, 75, 95, 99, 65, 96, 87, 89])

# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(hours, marks, train_size=0.5, random_state=16)

# Model Training
model = LinearRegression()
model.fit(x_train, y_train)

# Prediction for test data
y_pred_test = model.predict(x_test)

# Calculate MSE for the test set
mse = mean_squared_error(y_test, y_pred_test)
print(f"MSE for test data: {mse:.4f}")

# Prediction for a new data point (Desired hours)
i = int(input("Enter your desired hours: "))
y_pred_single = model.predict([[i]])
print(f"Predicted marks for {i} hours of study: {y_pred_single[0]:.2f}")

# Plotting the regression line
plt.scatter(hours, marks, color='blue', label='Actual data')  # Data points
plt.plot(hours, model.predict(hours), color='red', label='Regression line')  # Regression line
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Marks vs Hours Studied')
plt.legend()

# Save the plot to a file
plt.savefig('marks_vs_hours.png')  # Saves the file in the current working directory
plt.show()