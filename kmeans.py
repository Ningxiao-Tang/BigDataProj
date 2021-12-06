import numpy as np
from scipy.spatial import distance
from copy import deepcopy
import math


def select_centers(points, k, c):
    N = points.shape[0]
    for i in range(2, k+1):
        # compute shortest distance squared (from both centres)
        distances = []
        for point in points:
            d0 = distance.euclidean(point, c[0])
            minDis = d0*d0
            for center in c:
                d = distance.euclidean(point, center)
                minDis = min(minDis, d*d)
            distances.append(minDis)
        # compute cumulative distnace squared in cumul
        cumul = []
        sum_val = 0
        for dis in distances:
            sum_val += dis
            cumul.append(sum_val)
        # probabilities
        for i in range(len(cumul)):
            cumul[i] = cumul[i] / sum_val
        p = np.random.uniform(0, 1)
        index = 0
        for i in range(len(cumul)):
            if cumul[i] >= p:
                index = i
                break
        c.append(points[index])
    return c


def kmeansPP(points, k):
    # INITIAL MEANS
    c = []
    # RANDOMLY SELECT 1ST MEAN
    N = points.shape[0]
    n = points.shape[1]
    temp = np.random.randint(N)
    c.append(points[temp])
    c = select_centers(points, k, c)
    c = np.array(c, dtype=np.float)
    c_old = np.zeros((k, n))
    clusters = np.zeros(N) # Cluster Lables(0, 1, 2)
    error = np.linalg.norm(c - c_old,
                           axis=1)  # Error variable for euclidean distance between new centroids and old centroids
    # Loop will run till the error becomes zero
    while (error != 0).all():
        # Assigning each value to its closest cluster
        for i in range(N):
            distances = np.linalg.norm(points[i] - c, axis=1)  # computes dist of an object with each cluster centre
            cluster = np.argmin(distances)  # returns the index containing the min dist in "distances"
            clusters[i] = cluster  # assigns cluster label
        # Storing the old centroid values
        c_old = deepcopy(c)
        # Finding the new centroids by taking the average value
        for i in range(k):
            center = [points[j] for j in range(N) if clusters[j] == i]
            c[i] = np.mean(center, axis=0)
        error = np.linalg.norm(c - c_old, axis=1)

    return c
