from math import inf
from data_loader import csv_loader
import matplotlib.pyplot as plt

# Import the data
loaded_data = csv_loader('Mouse_Weight_and_Height_Dataset.csv') 
headers = loaded_data[0]
data = [(float(x), float(y)) for x, y in loaded_data[1]]

# Problem statement:
# Determine if mouse weight is a predictor of mouse height.
# -- Find the parameters for the line of best fit with equation y = B_0 + (B_1 * x)
# -- Find the analytical solution via minimizing the residual sum of squares.
# -- The parameters B_0 and B_1 can be found with the following equations:
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

B_1_equation_numerator = 0
B_1_equation_denominator = 0
for X_i, Y_i in data:
    B_1_equation_numerator += (X_i - X_bar) * (Y_i - Y_bar)
    B_1_equation_denominator += (X_i - X_bar)**2
B_1 = B_1_equation_numerator / B_1_equation_denominator
B_0 = (Y_bar) - (B_1 * X_bar)

# Plot the data
plot_data = True
if plot_data:
    X, Y = zip(*data)
    fitted_line_points = []
    for X_i in X:
        fitted_line_points.append((B_1 * X_i) + B_0)
    plt.scatter(X,Y, marker='o', color='b')
    plt.plot(X,fitted_line_points, color='red')
    plt.xlabel("Mouse Weight")
    plt.ylabel("Mouse Height")
    plt.title(" ")
    plt.show()