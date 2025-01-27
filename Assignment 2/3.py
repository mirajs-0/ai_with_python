import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading the CSV File using pandas
file = pd.read_csv("weight-height.csv")

# Fetching weights and heights from the CSV File
weight_in_pound = file["weight"]
height_in_inch = file["height"]

# Converting weight and height
weight_in_kg = weight_in_pound * 2.54
height_in_cm = height_in_inch * 0.454

# Calculating Mean Weight and Height
mean_weight = np.mean(weight_in_kg)
mean_height = np.mean(height_in_cm)

# Printing values of CSV File and Mean Value
print(file)

print(f"Mean Weight (kg): {mean_weight}")
print(f"Mean Height (cm): {mean_height}")

# Plotting the heights in histograms
plt.hist(height_in_cm, bins=20, color='blue', edgecolor='black')
plt.title("Histogram of Heights (in cm)")
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.show()