import numpy as np
import matplotlib.pyplot as plt
# Defining values of x manually using range
x = np.array(range(1, 13))

# Defining equations for y
y1 = np.array(2*x+1)
y2 = np.array(2*x+2)
y3 = np.array(2*x+3)

# Using subplot for different graphs for different equations

plt.subplot(1, 3, 1)
plt.plot(x, y1, color="black", linestyle="-", label="y = 2x+1")
plt.title("y = 2x + 1")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 3, 2)
plt.plot(x, y2, color="red", linestyle=":", label="y=2x+2")
plt.title("y = 2x + 2")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 3, 3)
plt.plot(x, y3, color="blue", linestyle="--", label="y=2x+3")
plt.title("y=2x+3")
plt.xlabel("x")
plt.ylabel("y")

# Main Title for the graph
plt.suptitle("Graphs of straight lines using equation")

# Preventing overlaps between labels
plt.tight_layout()

plt.show()
