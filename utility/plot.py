import numpy as np
import pickle
import matplotlib.pyplot as plt

Omg = np.array([[ 1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.73], [ 0.00,  1.00,  0.00,  0.79,  0.83,  0.96,  0.00], [ 0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00], [ 0.00,  0.79,  0.00,  1.62,  0.65,  0.76,  0.00], [ 0.00,  0.83,  0.00,  0.65,  1.69,  0.79,  0.00], [ 0.00,  0.96,  0.00,  0.76,  0.79,  1.92,  0.00], [ 0.73,  0.00,  0.00,  0.00,  0.00,  0.00,  1.54]])

Omg_hat = np.array([[ 1.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.65], [ 0.00,  0.94,  0.00,  0.61,  0.64,  0.87,  0.00], [ 0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00], [ 0.00,  0.61,  0.00,  1.25,  0.45,  0.58,  0.00], [ 0.00,  0.64,  0.00,  0.45,  1.30,  0.75,  0.00], [ 0.00,  0.87,  0.00,  0.58,  0.75,  1.50,  0.00], [ 0.65,  0.00,  0.00,  0.00,  0.00,  0.00 , 1.34]])

# Loop over data dimensions and create text annotations.
pMat = Omg.copy()
pMat[Omg_hat != 0] = 1
fig, ax = plt.subplots()
im = ax.imshow(pMat)
ax.set_title("Constrained Mask")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(Omg_hat)
for i in range(7):
    for j in range(7):
        text = ax.text(j, i, Omg_hat[i, j], ha="center", va="center", color="w")
ax.set_title("Estimated Omega")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(Omg)
for i in range(7):
    for j in range(7):
        text = ax.text(j, i, Omg[i, j], ha="center", va="center", color="w")
ax.set_title("Groundtruth Omega")
fig.tight_layout()
plt.show()