import  numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])

b = a[1:3,2:3]

k = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])


print(a[::-1].T[::-1])

g = np.flip(np.flip(a, 0), 1)

print()
print(g)
print(np.flip(a,0))