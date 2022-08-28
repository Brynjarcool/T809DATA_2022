# Author: Brynjar Ingi Óðinsson
# Date:20.08.2022
# Project: 02_nearest_neighbours
# Acknowledgements: 

import numpy as np
import matplotlib.pyplot as plt
import help
from tools import load_iris, split_train_test, plot_points
import pretty_errors
from sklearn.metrics import confusion_matrix, accuracy_score


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    d = 0

    for i in range(x.shape[0]):
        d += (x[i] - y[i])**2
    return d**0.5


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    return np.argsort(distances)[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    for i in range(len(targets)):
        targets[i] = classes[targets[i]]
    return np.argmax(np.bincount(targets))


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    distances = euclidian_distances(x, points)
    votes = vote(np.argsort(distances)[:k], point_targets)
    return votes


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    return_list = []
    for point in range(len(points)):
        return_list.append(knn(points[point], help.remove_one(points,point),help.remove_one(point_targets,point),classes,k))
    return return_list

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions = knn_predict(points, point_targets, classes, k)
    return np.mean(predictions == point_targets, where = np.isfinite(predictions))


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    return confusion_matrix(knn_predict(points, point_targets, classes, k),point_targets)


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    N = points.shape[0]
    interval = np.arange(1, N - 1)
    accuracies = np.zeros(interval.shape[0])
    for i in range(interval.shape[0]):
        accuracies[i] = knn_accuracy(points, point_targets, classes, interval[i])
    return interval[np.argmax(accuracies)]


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    colors = ["yellow","purple","blue"]
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        predictions = knn_predict(points, point_targets, classes, k)[i]
        if predictions == point_targets[i]:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors="green", linewidths=2)
        else:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors="red", linewidths= 2)
    plt.show()


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    weight_vec = np.zeros(len(targets))
    weighted_dist = np.zeros(len(targets))

    for i in range(len(targets)):
        weight_vec[i] = 1 / distances[targets[i]]
        targets[i] = classes[targets[i]]
        weighted_dist[i] = weight_vec[i] * classes[targets[i]]
    return np.argmax(np.bincount(targets, weights=weighted_dist))



def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    distances = euclidian_distances(x, points)
    votes = weighted_vote(np.argsort(distances)[:k], distances, point_targets)
    return votes


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    return_list = []
    for point in range(len(points)):
        return_list.append(wknn(points[point], help.remove_one(points,point),help.remove_one(point_targets,point),classes,k))
    return return_list

def wknn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    return np.mean(wknn_predict(points, point_targets, classes, k) == point_targets, where = np.isfinite(wknn_predict(points, point_targets, classes, k)))



def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ks = np.arange(1, points.shape[0])
    w_accuracies = np.zeros(ks.shape[0])
    accuracies = np.zeros((ks.shape[0]))
    for i in range(ks.shape[0]):
        w_accuracies[i] = wknn_accuracy(points, targets, classes, ks[i])
        accuracies[i] = knn_accuracy(points, targets, classes, ks[i])
    plt.plot(ks, accuracies)
    print(w_accuracies)
    plt.plot(ks, w_accuracies)
    plt.show()

d,t,classes = load_iris()
x,points = d[0,:],d[1:,:]
x_target, point_targets = t[0], t[1:]
(d_train,t_train), (d_test,t_test) = split_train_test(d,t, train_ratio = 0.8)
# knn_plot_points(d,t,classes,3)")
compare_knns(d_test,t_test,classes)