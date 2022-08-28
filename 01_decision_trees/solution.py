# Author: Brynjar Ingi Óðinsson
# Date:  17.08.2022
# Project: 01_decision_trees
# Acknowledgements: Prior function and split_data function taken from lecture notes

from random import sample
from traceback import print_exception
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import pretty_errors

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    samples_count = len(targets)
    class_probs = []
    class_count = 0

    for c in classes:
        for t in targets:
            if t == c:
                class_count += 1
        if samples_count == 0:
            class_probs.append(0)
            class_count = 0
        else:
            class_probs.append(class_count / samples_count)
            class_count = 0
    return (class_probs)


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = features[features[:, split_feature_index] < theta]
    targets_1 = targets[features[:, split_feature_index] < theta]

    features_2 = features[features[:, split_feature_index] >= theta]
    targets_2 = targets[features[:, split_feature_index] >= theta]

    
    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    '''
    Calculate the gini impurity of a single branch e.g. gini_impurity(t_1,classes) -> 0.2517
    '''
    class_probs = prior(targets, classes)
    denom = sum(class_probs)
    return (1 - sum([p**2 for p in class_probs])) / 2



def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    return ((g1 * t1.shape[0] + g2 * t2.shape[0]) / n)


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (t1, t2) = split_data(features, targets, split_feature_index, theta)
    return  (weighted_impurity(t1[1], t2[1], classes))



def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        thetas = np.linspace(np.min(features[:, i]), np.max(features[:, i]), num_tries)
        # iterate thresholds
        for theta in thetas:
            gini = total_gini_impurity(features, targets, classes, i, theta)
            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.6
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    def plot(self):
        plot_tree(self.tree, filled=True)
        plt.show()

    def plot_progress(self):
        # Remove this method if you don't go for independent section.
        '''
        Plot the progress of the training process.
        '''
        n  = [0]*self.train_features.shape[0]

        ticks = []
        count = 0
        for i in range(self.train_features.shape[0]):
            n[i] = self.tree.fit(self.train_features[1:i][:], self.train_targets[1:i])
        plt.plot(ticks, n)
        plt.show()

    def guess(self):        
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        return confusion_matrix(self.test_targets, self.tree.predict(self.test_features))

# MAIN PART 
features, targets, classes = load_iris()
(f_1,t_1), (f_2,t_2) = split_data(features,targets, 2 ,4.65)

dt = IrisTreeTrainer(features,targets,classes=classes)
dt.train()
print(f'The accuracy is: {dt.accuracy()}')
print(f' I Guessed the following: {dt.guess()}')
print(f'The confusion matrix is: {dt.confusion_matrix()}')
dt.plot_progress()