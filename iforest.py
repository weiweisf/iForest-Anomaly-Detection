import math
import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import product
import random
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = math.ceil(math.log2(self.sample_size))
        self.trees = []

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        for i in range(self.n_trees):
            idx = np.random.randint(X.shape[0], size=self.sample_size)
            X_sub = X[idx, :]
            tree = IsolationTree(X_sub, 0, self.height_limit)
            self.trees.append(tree.fit(improved))

        return self

    def c_func(self, x):
        if x > 2:
            return 2 * (np.log(x-1)+0.5772156649) - 2*(x-1)/x
        elif x==2:
            return 1
        else:
            return 0

    def path_length_instance_rec(self, x, T, e):
        if isinstance(T, IsolationTree.exNode):
            return e + self.c_func(T.size)
        a = T.splitAtt
        if x[a] < T.splitValue:
            return self.path_length_instance(x, T.left, e+1)
        else:
            return self.path_length_instance(x, T.right, e+1)

    def path_length_instance(self, x, T, e=0):
        while not isinstance(T, IsolationTree.exNode):
            a = T.splitAtt
            if x[a] < T.splitValue:
                T = T.left
            else:
                T = T.right
            e += 1
        return e + self.c_func(T.size)

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        p = Pool(4)
        # paths = []
        input_list = product(X, self.trees)
        lengths = p.starmap(self.path_length_instance, input_list)
        lengths = np.array(lengths).reshape(len(X), -1)
        paths = lengths.mean(axis=1)
        # for x_i in X:
        #     # lengths = 0
        #
        #     # input_list = [[x_i, T, 0] for T in self.trees]
        #     input_list = list(product([x_i], self.trees))
        #     lengths = p.starmap(self.path_length_instance, input_list)
        #     # for i, T in enumerate(self.trees):
        #     #     lengths += self.path_length_instance(x_i, T, 0)
        #     # paths.append(lengths/(i+1))
        #     paths.append(np.mean(np.array(lengths)))
        return np.array(paths)


    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        Eh = self.path_length(X)
        return 2 ** (-Eh/self.c_func(self.sample_size))


    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return scores>=threshold

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."


class IsolationTree:
    class exNode:
        def __init__(self, X):
            self.size = len(X)
            self.left = None
            self.right = None

    class inNode:
        def __init__(self, left, right, splitAtt, splitValue, n_nodes):
            self.left = left
            self.right = right
            self.splitAtt = splitAtt
            self.splitValue = splitValue
            self.n_nodes = n_nodes

    def count_nodes(self, node):
        if node is None:
            return 0
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

    def __init__(self, X:np.ndarray, current_height, height_limit):
        self.X = X
        self.e = current_height
        self.l = height_limit


    def fit(self, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if not improved:
            X = self.X
            if (self.e >= self.l) or (len(X) <= 1):
                self.root = self.exNode(X)
            else:
                Q = np.arange(X.shape[1])
                q = np.random.choice(Q)
                p = np.random.uniform(min(X[:,q]), max(X[:,q]))
                X_l = X[X[:,q]<p]
                X_r = X[X[:,q]>=p]
                left_tree = IsolationTree(X_l, self.e+1, self.l).fit(improved)
                right_tree = IsolationTree(X_r, self.e+1, self.l).fit(improved)
                n_nodes = self.count_nodes(left_tree) + self.count_nodes(right_tree)
                self.root = self.inNode(left_tree,
                                        right_tree,
                                        splitAtt=q,
                                        splitValue=p,
                                        n_nodes=n_nodes)
        else:
            # use min
            X = self.X
            if (self.e >= self.l) or (len(X) <= 1):
                self.root = self.exNode(X)
            else:
                Q = np.arange(X.shape[1])
                q_list = random.sample(list(Q), 8)
                # bottom_len = 3
                min_len = 4
                for q in q_list:
                    u = np.random.uniform()
                    min_q = min(X[:, q])
                    max_q = max(X[:, q])
                    divi_q = (max_q - min_q) / 2.2
                    if u <= 0.5:
                        p = np.random.uniform(min_q, min_q+divi_q)
                    else:
                        p = np.random.uniform(max_q-divi_q, max_q)
                    X_l_tmp = X[X[:, q] < p]
                    X_r_tmp = X[X[:, q] >= p]
                    if min(len(X_l_tmp), len(X_r_tmp)) <= min_len:
                        break
                X_l = X_l_tmp
                X_r = X_r_tmp

                left_tree = IsolationTree(X_l, self.e + 1, self.l).fit(improved)
                right_tree = IsolationTree(X_r, self.e + 1, self.l).fit(improved)
                n_nodes = self.count_nodes(left_tree) + self.count_nodes(right_tree)
                self.root = self.inNode(left_tree,
                                        right_tree,
                                        splitAtt=q,
                                        splitValue=p,
                                        n_nodes=n_nodes)

        # fit time exceed
        # else:
        #     X = self.X
        #     if (self.e >= self.l) or (len(X) <= 1):
        #         self.root = self.exNode(X)
        #     else:
        #         Q = np.arange(X.shape[1])
        #         kurt = np.array([kurtosis(X[:, i]) for i in range(X.shape[1])])
        #         kurt = kurt - min(kurt)
        #         # print(kurt)
        #         # sys.exit()
        #         kurt_prob = kurt / np.sum(kurt)
        #         # print("done here")
        #         kurt_prob[kurt_prob.argsort()[:5]] = 0
        #         kurt_prob = kurt_prob / np.sum(kurt_prob)
        #         q = np.random.choice(Q, p=kurt_prob)
        #         u = np.random.uniform()
        #         min_q = min(X[:, q])
        #         max_q = max(X[:, q])
        #         divi_q = (max_q - min_q) / 3
        #         if u <= 0.5:
        #             p = np.random.uniform(min_q, min_q+divi_q)
        #         else:
        #             p = np.random.uniform(max_q-divi_q, max_q)
        #         X_l = X[X[:, q] < p]
        #         X_r = X[X[:, q] >= p]
        #         left_tree = IsolationTree(X_l, self.e + 1, self.l).fit(improved)
        #         right_tree = IsolationTree(X_r, self.e + 1, self.l).fit(improved)
        #         n_nodes = self.count_nodes(left_tree) + self.count_nodes(right_tree)
        #         self.root = self.inNode(left_tree,
        #                                 right_tree,
        #                                 splitAtt=q,
        #                                 splitValue=p,
        #                                 n_nodes=n_nodes)
        return self.root


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    return(TP, FP, TN, FN)

def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    for threshhold in (1 - np.arange(0, 1, step=0.01)):
        TP, FP, TN, FN = perf_measure(y.values, (scores>=threshhold))
        tpr = TP/(TP+FN)
        fpr = FP/(FP+TN)
        if tpr >= desired_TPR:
            break
    return threshhold, fpr
