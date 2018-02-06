a = np.array([[2, 3, 4]])
b = np.ones(a.shape)
C = a.T @ b
C + np.eye(3)
