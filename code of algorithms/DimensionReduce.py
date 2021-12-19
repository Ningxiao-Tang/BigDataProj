import math
import random
import numpy as np


def fastmap(k, distances, col, X):
    N = distances.shape[0]
    if k <= 0:
        return X
    else:
        col += 1
    a, b = select_pivots(distances)
    farthest = distances[a][b]
    if farthest == 0:
        for i in range(N):
            X[i][col] = 0
    currDim = []
    for i in range(N):
        if i == a:
            X[i][col] = 0
            currDim.append(0)
        elif i == b:
            X[i][col] = farthest
            currDim.append(farthest)
        else:
            temp = ((distances[a][i] ** 2) + (farthest ** 2) - (distances[b][i] ** 2)) / float(2 * farthest)
            currDim.append(farthest)
            X[i][col] = temp
    projection = np.zeros((N, N), dtype=np.float)
    if k >= 1:
        for i in range(N):
            for j in range(N):
                tmp = (distances[i][j] ** 2) - ((currDim[i] - currDim[j]) ** 2)
                projection[i][j] = np.sqrt(np.absolute(tmp))
        return fastmap(k-1, projection, col, X)
    else:
        return X



def init_distances(points):
    N = points.shape[0]
    n = points.shape[1]
    distances = np.zeros((N, N), dtype=np.float)
    for i in range(0, N):
        for j in range(0, N):
            dis = 0
            for k in range(0, n):
                dis += (points[i][k]-points[j][k])**2
            distances[i][j] = math.sqrt(dis)
    return distances


def select_pivots(distances):
    N = distances.shape[0]
    obj_b = random.randint(0, N-1)
    while True:
        farthest = max(distances[obj_b])
        obj_a = distances[obj_b].tolist().index(farthest)
        tmp = max(distances[obj_a])
        tmpObj = distances[obj_a].tolist().index(tmp)
        if (tmpObj == obj_b):
            break
        else:
            obj_b = tmpObj
    if obj_a < obj_b:
        return obj_a, obj_b
    else:
        return obj_b, obj_a


def dimRed(points, k):
    # points: dataset of dimension n
    # k: dimensionality of target space
    N = points.shape[0]
    n = points.shape[1]
    X = np.zeros((N, k), dtype=np.float)
    col = int(-1)
    distances = init_distances(points)
    X = fastmap(k, distances, col, X)
    return X

