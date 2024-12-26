import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x = np.array([10,30,45,60,75,100]).reshape(-1,1)
y = np.array([1299,3699,5499,7299,9099,12099])

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 32)

# Training
model = LinearRegression()
model.fit(x_train,y_train)

# Predicting
y_pred = model.predict(x_test)

# Evaluating
mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error: {mse:.2f}%")

# Testing the Algorithm
new_feature = np.array([[140]])
predicted_price = model.predict(new_feature)
print(predicted_price)