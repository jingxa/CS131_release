import numpy as np


if __name__ == "__main__":
    a = np.array([1])
    b = np.array([[7,7,7,7,7],[1,2,3,4,5],[0,6,6,6,6]])

    c = b[:,0:4]
    f = c[:,: -1]
    print(c)
    print('----')
    print(f)