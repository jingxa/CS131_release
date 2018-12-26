import numpy as np


def compute_distances(X1, X2):

    M = X1.shape[0]
    N = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]

    dists = np.zeros((M, N))

    # YOUR CODE HERE
    # (x-y)^2 == x^2 + y^2 -2xy
    x_2 = np.sum(X1**2, axis=1)[:, np.newaxis]     # 增加一个维度
    y_2 = np.sum(X2**2, axis=1)
    xy_2 = np.dot(X1, X2.T)
    dists = x_2 + y_2 - 2 * xy_2
    # END YOUR CODE

    assert dists.shape == (M, N), "dists should have shape (M, N), got %s" % dists.shape

    return dists


def predict_labels(dists, y_train, k=1):

    num_test, num_train = dists.shape
    y_pred = np.zeros(num_test, dtype=np.int)

    for i in range(num_test):
        # A list of length k storing the labels of the k nearest neighbors to
        # the ith test point.
        closest_y = []
        # Use the distance matrix to find the k nearest neighbors of the ith
        # testing point, and use self.y_train to find the labels of these
        # neighbors. Store these labels in closest_y.
        # Hint: Look up the function numpy.argsort.

        # Now that you have found the labels of the k nearest neighbors, you
        # need to find the most common label in the list closest_y of labels.
        # Store this label in y_pred[i]. Break ties by choosing the smaller
        # label.

        # YOUR CODE HERE
        indices = np.argsort(dists[i])      # 最近的label,排序
        closest_y = y_train[indices[:k]]    # 选择最近的k个
        # y_train 为重复的label，本例子中有800章图片，共16个分类
        y_pred[i] = np.bincount(closest_y).argmax()     # 选择k个最相似的label中重复最多的
        # END YOUR CODE

    return y_pred


def split_folds(X_train, y_train, num_folds):
    """Split up the training data into `num_folds` folds.

    The goal of the functions is to return training sets (features and labels) along with
    corresponding validation sets. In each fold, the validation set will represent (1/num_folds)
    of the data while the training set represent (num_folds-1)/num_folds.
    If num_folds=5, this corresponds to a 80% / 20% split.

    For instance, if X_train = [0, 1, 2, 3, 4, 5], and we want three folds, the output will be:
        X_trains = [[2, 3, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 2, 3]]
        X_vals = [[0, 1],
                  [2, 3],
                  [4, 5]]

    Args:
        X_train: numpy array of shape (N, D) containing N examples with D features each
        y_train: numpy array of shape (N,) containing the label of each example
        num_folds: number of folds to split the data into

    jeturns:
        X_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds, D)
        y_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds)
        X_vals: numpy array of shape (num_folds, train_size / num_folds, D)
        y_vals: numpy array of shape (num_folds, train_size / num_folds)

    """
    assert X_train.shape[0] == y_train.shape[0]

    validation_size = X_train.shape[0] // num_folds
    training_size = X_train.shape[0] - validation_size

    X_trains = np.zeros((num_folds, training_size, X_train.shape[1]))
    y_trains = np.zeros((num_folds, training_size), dtype=np.int)
    X_vals = np.zeros((num_folds, validation_size, X_train.shape[1]))
    y_vals = np.zeros((num_folds, validation_size), dtype=np.int)

    # YOUR CODE HERE
    # Hint: You can use the numpy array_split function.
    # 例如 [1, 2, 3, 4,5]
    # [1,2,3,4: 5]
    # [2,3,4,5: 1]
    # [3,4,5,1: 2]
    # [4,5,1,2: 3]
    # [5,1,2,3: 4] 这五种组合方式
    X_num_folds = np.array(np.array_split(X_train, num_folds))
    y_num_folds = np.array(np.array_split(y_train,num_folds))

    for i in range(num_folds):
        X_trains[i] = X_num_folds[(np.arange(num_folds) != i)].reshape((-1,X_trains.shape[-1]))
        X_vals[i] = X_num_folds[i]

        y_trains[i] = y_num_folds[(np.arange(num_folds) != i)].reshape((-1, y_trains.shape[-1]))
        y_vals[i] = y_num_folds[i]

    # END YOUR CODE

    return X_trains, y_trains, X_vals, y_vals
