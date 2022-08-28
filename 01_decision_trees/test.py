from traceback import print_exception
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test

def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    samples_count = len(targets)
    class_probs = []
      
    print("in prior!!!")
    for c in classes:
        class_count = 0
        for t in targets:
            if t == c:
                class_count += 1
        print("SAMPLES COUNT", samples_count)
        class_probs.append(class_count / samples_count)

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
    print("in split_data!!!!!")
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
    print("GINI IMPURITY TARGETS: ",targets)
    print(" in gini_impurity!!!!!")
    class_probs = prior(targets, classes)
    denom = sum(class_probs)
    print("DENOM", denom)
    return (1 - sum([p**2 for p in class_probs]) / denom) / 2



def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    print("WEIGHTED IMPURITY TARGET 1: ", t1 )
    print("WEIGHTED IMPURITY TARGET 2: ", t2)
    print("in weighted_impurity!!!!")
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    return ((g1 * t1.shape[0] + g2 * t2.shape[0]) / n)/2


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
    print("in total_gini_impurity!!!!!!")
    (t1, t2) = split_data(features, targets, split_feature_index, theta)
    return  2 *(weighted_impurity(t1[1], t2[1], classes))



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
    print(targets)
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

features, targets, classes = load_iris()
best_split=brute_best_split(features,targets,classes,30)

