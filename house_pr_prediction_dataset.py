import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

house_size = np.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750]).reshape(-1, 1)
house_price = np.array([50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000])

# Splitting
x_train, x_test, y_train, y_test = train_test_split(house_size, house_price, test_size=0.2, random_state=58)

# Training
model = LinearRegression()
model.fit(x_train, y_train)

# Testing
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

# Visualization
plt.scatter(house_size, house_price, color = 'blue', label = "Actual Prices")
plt.plot(house_size, model.predict(house_size), color = 'red', label = "Regression Line")
plt.title("House Price vs Size")
plt.xlabel("Size(sq. ft.)")
plt.ylabel("Price")
plt.legend()
plt.show()