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

# Problem statement
# We want to see if weight is a predictor of height, so weight is our independent variable.
# We use R^2 to determine blah blah blah
# R^2:  SSR(mean) - SSR(fitted line)
#       ____________________________
#                SSR(mean)
#
# SSR = (observed_i - predicted_i)^2
#
# 1. Calculate SSR(fitted line):
# * Calculate maximum and minimum Y coordinates, this will inform us when to stop in training.
# ** Take the absolute highest or lowest Y coordinate and set the y_intercept to that
# ** Training will be limited to [Y_lowest, Y_highest]


fitted_y_intercept = 0
fited_slope = 1
mean_intercept = 0
MEAN_SLOPE = 0
data_y_lowest = inf
data_y_highest = -inf

for _, y in data:
    data_y_lowest = min(data_y_lowest, y)
    data_y_highest = max(data_y_highest, y)

fitted_y_intercept = data_y_lowest if abs(data_y_lowest) > abs(data_y_highest) else data_y_highest


# * Minimize RSS w.r.t. fitted_y_intercept
for x, y in data:
    pass





# 2. Calculate SSR(mean):
