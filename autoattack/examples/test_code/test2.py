import numpy as np

svd90 = [5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6]
svd70 = [10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2]
svd50 = [23.1, 23.0, 23.6, 23.1, 22.9, 23.0, 23.0, 23.4, 23.1, 23.3]
svd30 = [47.4, 47.2, 47.3, 47.1, 47.4, 47.2, 47.5, 47.6, 47.4, 47.4]
svd20 = [60.3, 60.2, 60.0, 59.8, 60.2, 59.8, 59.8, 59.9, 59.8, 59.9]
svd10 = [77.5, 77.6, 77.2, 77.5, 77.1, 77.8, 76.8, 77.2, 77.5, 77.3]

print(np.mean(svd90))
print(np.mean(svd70))
print(np.mean(svd50))
print(np.mean(svd30))
print(np.mean(svd20))
print(np.mean(svd10))
print(np.std(svd90))
print(np.std(svd70))
print(np.std(svd50))
print(np.std(svd30))
print(np.std(svd20))
print(np.std(svd10))