from math import inf
from data_loader import csv_loader
import matplotlib.pyplot as plt

# Import the data
loaded_data = csv_loader('Mouse_Weight_and_Height_Dataset.csv') 
headers = loaded_data[0]
data = [(float(x), float(y)) for x, y in loaded_data[1]]


# Plot the data
plot_data = False
if plot_data:
    x, y = zip(*data)
    plt.scatter(x,y, marker='o', color='b')
    plt.xlabel("Mouse Weight")
    plt.ylabel("Mouse Height")
    plt.title(" ")
    plt.show()

# Problem statement:
# Determine if mouse weight is a predictor of mouse height.
# -- Find the analytical solution via minimizing the residual sum of squares.
# -- The parameters B_0 and B_1 can be found with the following equations, found by
# -- B_1 = sum((X_i - X_bar) * (Y_i - Y_bar))
# --       ________________________________
# --              sum((X_i - X_bar)^2)
# -- 
# --
# -- B_0 = Y_bar - (B_1 * X_bar) 

X_bar = 0
Y_bar = 0

for x, y in data:
    X_bar += x
    Y_bar += y

X_bar /= len(data)
Y_bar /= len(data)

print(X_bar, Y_bar)