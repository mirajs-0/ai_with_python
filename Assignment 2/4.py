import numpy as np

# Defining Matrix A
A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])

# Calculating inverse of the Matrix
A_inverse = np.linalg.inv(A)

# Compute the product A * A_inv and A_inv * A
product_A_A_inverse = A@A_inverse
product_A_inverse_A = A_inverse@A

# Print the results
print("Matrix A:")
print(A)

print("\nInverse of Matrix A (A_inv):")
print(A_inverse)

print("\nProduct of A and A_inv (A * A_inv):")
print(product_A_A_inverse)

print("\nProduct of A_inv and A (A_inv * A):")
print(product_A_inverse_A)
