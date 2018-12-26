import numpy as np
import scipy
import scipy.linalg


class PCA(object):
    """Class implementing Principal component analysis (PCA).

    Steps to perform PCA on a matrix of features X:
        1. Fit the training data using method `fit` (either with eigen decomposition of SVD)
        2. Project X into a lower dimensional space using method `transform`
        3. Optionally reconstruct the original X (as best as possible) using method `reconstruct`
    """

    def __init__(self):
        self.W_pca = None
        self.mean = None

    def fit(self, X, method='svd'):
        """Fit the training data X using the chosen method.

        Will store the projection matrix in self.W_pca and the mean of the data in self.mean

        Args:
            X: numpy array of shape (N, D). Each of the N rows represent a data point.
               Each data point contains D features.
            method: Method to solve PCA. Must be one of 'svd' or 'eigen'.
        """
        _, D = X.shape
        self.mean = None   # empirical mean, has shape (D,)
        X_centered = None  # zero-centered data

        # YOUR CODE HERE
        # 1. Compute the mean and store it in self.mean
        self.mean = np.mean(X, axis=0)      # 均值
        X_centered = X - self.mean
        # 2. Apply either method to `X_centered`
        if method == "svd":
            vecs, vals = self._svd(X_centered)
        elif method == "eigen":
            vecs, vals = self._eigen_decomp(X_centered)
        else:
            print("ERROR: Unknown method!")
            return

        # data:(N, D) * matrix:(D,k) = new_data:(N,k)
        self.W_pca = vecs
        # END YOUR CODE

        # Make sure that X_centered has mean zero
        assert np.allclose(X_centered.mean(), 0.0)

        # Make sure that self.mean is set and has the right shape
        assert self.mean is not None and self.mean.shape == (D,)

        # Make sure that self.W_pca is set and has the right shape
        assert self.W_pca is not None and self.W_pca.shape == (D, D)

        # Each column of `self.W_pca` should have norm 1 (each one is an eigenvector)
        for i in range(D):
            assert np.allclose(np.linalg.norm(self.W_pca[:, i]), 1.0)

    def _eigen_decomp(self, X):
        N, D = X.shape
        e_vecs = None
        e_vals = None
        # YOUR CODE HERE
        # Steps: 计算步骤如下
        #     1. compute the covariance matrix of X, of shape (D, D)
        matrix = np.matmul(X.T, X) / (X.shape[0] - 1)        # 协方差矩阵 (D,D)
        #     2. compute the eigenvalues and eigenvectors of the covariance matrix
        e_vals, e_vecs = np.linalg.eig(matrix)      # 计算特征值特征向量
        #     3. Sort both of them in decreasing order (ex: 1.0 > 0.5 > 0.0 > -0.2 > -1.2)
        indices = np.argsort(-e_vals)       # 从达到小排序，序号
        e_vals = np.real(e_vals[indices])   # 返回实数
        e_vecs = np.real(e_vecs[:, indices])
        # END YOUR CODE
        # Check the output shapes
        assert e_vals.shape == (D,)
        assert e_vecs.shape == (D, D)

        return e_vecs, e_vals

    def _svd(self, X):

        vecs = None  # shape (D, D)
        N, D = X.shape
        vals = None  # shape (K,)
        # YOUR CODE HERE
        # Here, compute the SVD of X
        # Make sure to return vecs as the matrix of vectors where each column is a singular vector
        # U: X*X.T , V.T = X.T*X, S:square roots of the eigenvalues of U or V.T
        u, s, vt = np.linalg.svd(X)
        vals = s
        vecs = vt.T  # 转置
        # END YOUR CODE
        assert vecs.shape == (D, D)
        K = min(N, D)
        assert vals.shape == (K,)

        return vecs, vals

    def transform(self, X, n_components):
        N, _ = X.shape
        X_proj = None
        # YOUR CODE HERE
        # We need to modify X in two steps:
        #     1. first substract the mean stored during `fit`
        #     2. then project onto a subspace of dimension `n_components` using `self.W_pca`
        X_proj = X - self.mean
        X_proj = X_proj.dot(self.W_pca[:, :n_components])
        # END YOUR CODE

        assert X_proj.shape == (N, n_components), "X_proj doesn't have the right shape"

        return X_proj

    def reconstruct(self, X_proj):
        N, n_components = X_proj.shape
        X = None

        # YOUR CODE HERE
        # Steps:
        #     1. project back onto the original space of dimension D
        X = np.zeros((N, self.W_pca.shape[1]))      # 原空间
        X[:, :n_components] = X_proj
        #     2. add the mean that we substracted in `transform`
        X = X.dot(np.linalg.inv(self.W_pca)) + self.mean
        pass
        # END YOUR CODE

        return X


class LDA(object):
    """Class implementing Principal component analysis (PCA).

    Steps to perform PCA on a matrix of features X:
        1. Fit the training data using method `fit` (either with eigen decomposition of SVD)
        2. Project X into a lower dimensional space using method `transform`
        3. Optionally reconstruct the original X (as best as possible) using method `reconstruct`
    """

    def __init__(self):
        self.W_lda = None

    def fit(self, X, y):
        N, D = X.shape

        scatter_between = self._between_class_scatter(X, y)
        scatter_within = self._within_class_scatter(X, y)

        e_vecs = None

        # YOUR CODE HERE
        # Solve generalized eigenvalue problem for matrices `scatter_between` and `scatter_within`
        # Use `scipy.linalg.eig` instead of numpy's eigenvalue solver.
        # Don't forget to sort the values and vectors in descending order.
        e_vals, e_vecs = scipy.linalg.eig(np.linalg.inv(scatter_within).dot(scatter_between))
        # END YOUR CODE
        sorting_order = np.argsort(e_vals)[::-1]
        e_vals = e_vals[sorting_order]
        e_vecs = e_vecs[:,sorting_order]        # 从大到小排序

        self.W_lda = e_vecs

        # Check that the shape of `self.W_lda` is correct
        assert self.W_lda.shape == (D, D)

        # Each column of `self.W_lda` should have norm 1 (each one is an eigenvector)
        for i in range(D):
            assert np.allclose(np.linalg.norm(self.W_lda[:, i]), 1.0)

    def _within_class_scatter(self, X, y):
        _, D = X.shape
        assert X.shape[0] == y.shape[0]
        scatter_within = np.zeros((D, D))

        for i in np.unique(y):
            # YOUR CODE HERE
            # Get the covariance matrix for class i, and add it to scatter_within
            X_i = X[y == i]     # 类为i
            X_i_centered = X_i - np.mean(X_i, axis=0)
            S_i = np.matmul(X_i_centered.T, X_i_centered)       # 协方差矩阵
            scatter_within += S_i       # 结果
            # END YOUR CODE

        return scatter_within

    def _between_class_scatter(self, X, y):

        _, D = X.shape
        assert X.shape[0] == y.shape[0]
        scatter_between = np.zeros((D, D))

        mu = X.mean(axis=0)
        X_copy = X.copy()
        for i in np.unique(y):
            # YOUR CODE HERE
            X_i = X[y==i]           # 类别i
            mu_i = np.mean(X_i, axis=0) # 类别i的均值
            # scatter_between += (len(y[y==i])) * np.matmul((mu_i - mu).T, (mu_i - mu))
            #  `scatter_between` is the covariance matrix of X where we replaced every example labeled i with mu_i.
            X_copy[y==i] = mu_i
            # END YOUR CODE
        return scatter_between


    def transform(self, X, n_components):
        N, _ = X.shape
        X_proj = None
        # YOUR CODE HERE
        # project onto a subspace of dimension `n_components` using `self.W_lda`
        X_proj = np.matmul(X, self.W_lda[:, :n_components])
        # END YOUR CODE

        assert X_proj.shape == (N, n_components), "X_proj doesn't have the right shape"

        return X_proj
