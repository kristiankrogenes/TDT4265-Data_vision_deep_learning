import numpy as np

inds = np.array([0.4, 0.7, 0.6, 0.9]) >= 0.9

arr = np.array([[0., 0., 1., 1.],
    [0., 0., 1.5, 1.5],
    [3., 3., 4., 4.],
    [5., 5., 8., 8.]])

print(any(inds))
print("1", inds)
print("2", arr)

print("3", arr[inds])