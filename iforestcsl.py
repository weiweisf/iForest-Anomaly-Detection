
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import multiprocessing as mp
import itertools

cpu_cnt = mp.cpu_count()
# cpu_cnt

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=100):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        self.l = np.ceil(np.log2(self.sample_size)) 

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        for _ in range(self.n_trees):
            ix = random.sample(range(X.shape[0]), self.sample_size) 
            self.trees.append(IsolationTree(self.l).fit(X[ix]))
        
        return self 
    
    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        pool = mp.Pool(processes=cpu_cnt)
        
        paramlist = itertools.product(X, self.trees)

        treePathLength = pool.starmap(tree_path_len, paramlist)
        avgPathLength = np.array(treePathLength).reshape(len(X), -1).mean(axis=1)
        
        return avgPathLength

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        scoreLst = []
        avgPathLength = self.path_length(X)
        c = c_value(self.sample_size)
        for i in range(X.shape[0]):
            scoreLst.append(2.0**(-avgPathLength[i] / c))
        
        return np.array(scoreLst)

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores >= threshold).astype(int)

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)

class IsolationTree:
    def __init__(self, height_limit, current_height=0):
        self.height_limit = height_limit
        self.current_height = current_height # number of decision nodes
        self.root = None
        self.n_nodes = 0

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if self.current_height >= self.height_limit or len(X) <= 1:
            self.size = len(X) 
            self.n_nodes += 1
            return LeafNode(size=self.size)
        else:
            q = random.sample(range(X.shape[1]), 1)[0]
            min_val = X[:, q].min()
            max_val = X[:, q].max()
            if min_val == max_val:
                self.n_nodes += 1
                return LeafNode(size=len(X))
            p = float(random.uniform(min_val, max_val))
            
            tree_left = IsolationTree(self.height_limit, self.current_height + 1).fit(X[X[:, q] < p])
            tree_right = IsolationTree(self.height_limit, self.current_height + 1).fit(X[X[:, q] >= p])
            
            self.n_nodes += self.count_nodes(tree_left) + self.count_nodes(tree_right)
            self.root = DecisionNode(size=len(X), splitAttr=q, splitVal=p, n_nodes = self.n_nodes,
                                     left=tree_left, right=tree_right)
        return self.root
    
    def count_nodes(self, tr):
        if tr is None:
            return 0
        else:
            return 1 + self.count_nodes(tr.left) + self.count_nodes(tr.right)
        
class DecisionNode:
    def __init__(self, size, splitAttr, splitVal, left, right, n_nodes, node_type='decision_node'):
        self.left = left
        self.right = right
        self.splitAttr = splitAttr
        self.splitVal = splitVal
        self.size = size
        self.node_type = node_type
        self.n_nodes = n_nodes

class LeafNode:
    def __init__(self, size, left=None, right=None, n_nodes=1, node_type='leaf_node'):
        self.size = size   
        self.node_type = node_type
        self.left = left
        self.right = right
        self.n_nodes = n_nodes
        
def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold, FPR = 1, 0
    for s in np.arange(threshold, 0, -0.01):
        y_pred = (scores >= s).astype(int)
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        if np.abs(TPR - desired_TPR) <= 0.1:
            FPR = FP / (FP + TN)
            break
    return s, FPR

def c_value(n):
    if n > 2:
        return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - (2.0 * (n - 1.0) / (n * 1.0))
    elif n == 2:
        return 1.0
    else:
        return np.finfo(np.float).eps

def tree_path_len(xs, tr, e=0):
    while not isinstance(tr, LeafNode):
        if xs[tr.splitAttr] < tr.splitVal:
            tr = tr.left
        else:
            tr = tr.right
        e += 1
    return e + c_value(tr.size)
