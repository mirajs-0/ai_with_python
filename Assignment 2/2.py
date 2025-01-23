import numpy as np
import matplotlib.pyplot as plt

# Creating numpy vector using array
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

# Using scatter for drawing scattering pattern
plt.scatter(x, y, marker="+")
plt.title("Scattering points for x and y")
plt.xlabel("x")
plt.ylabel("y")

# Displaying the plot in the graph
plt.show()