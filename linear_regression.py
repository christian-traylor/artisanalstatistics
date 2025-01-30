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
fitted_slope = 1
mean_intercept = 0
MEAN_SLOPE = 0
data_y_lowest = inf
data_y_highest = -inf

for _, y in data:
    data_y_lowest = min(data_y_lowest, y)
    data_y_highest = max(data_y_highest, y)


fitted_y_intercept = None
direction = None
if abs(data_y_lowest) > abs(data_y_highest):
    fitted_y_intercept = data_y_lowest
    direction = "UP"
else:
    fitted_y_intercept = data_y_highest
    direction = "DOWN"


# * Minimize RSS w.r.t. fitted_y_intercept using the equation:
# * y = mx + b
rss_y_intercept_loss_best_value = inf
rss_y_intercept_loss_best_value_number = None

# We should maybe just sort the data, so we can adjust move to the next data point. Set fitted_y_intercept to next data point

for x, _ in data:
    if direction == "DOWN":
        fitted_y_intercept = None # we need variation 
    for x_i, observed_i in data:
        predicted_i = (x_i * fitted_slope) + fitted_y_intercept
        rss_y_intercept_loss = (observed_i - predicted_i)**2
        if rss_y_intercept_loss < rss_y_intercept_loss_best_value:
            rss_y_intercept_loss_best_value_number = fitted_y_intercept
            rss_y_intercept_loss_best_value = rss_y_intercept_loss

print(rss_y_intercept_loss_best_value_number)

# * Minimize RSS w.r.t. fitted_slope using the equation:
# * y = mx + b
# rss_slope_loss_best_value = -1
# rss_slope_loss_best_value_number = None

# for x, _ in data: 
#     for x_i, observed_i in data:
#         predicted_i = (x_i * fitted_slope) + fitted_y_intercept
#         rss_slope_loss = (observed_i - predicted_i)**2
#         if rss_y_intercept_loss < rss_y_intercept_loss_best_value:
#             rss_slope_loss_best_value_number = fitted_y_intercept
#             rss_slope_loss_best_value = rss_y_intercept_loss

# 2. Calculate SSR(mean):
