import DimensionReduce
import kmeans
import pandas as pd
import math


def reducedData(readPath, k, writePath):
    df = pd.read_csv(readPath)
    points = df.to_numpy()[:, 1:]
    points = DimensionReduce.dimRed(points, k)
    N = points.shape[0]
    indexes = []
    fields = []
    for i in range(1, N + 1):
        indexes.append('points' + str(i))
    for i in range(1, k + 1):
        fields.append('dim' + str(i))
    newDf = pd.DataFrame(points, indexes, fields)
    newDf.to_csv(writePath)


def compute_cost(path, k):
    df = pd.read_csv(path)
    points = df.to_numpy()[:, 1:]
    points = points.astype('float')
    c, clusters = kmeans.kmeansPP(points, k)
    N = points.shape[0]
    n = points.shape[1]
    cost = 0
    for i in range(N):
        cluster = int(clusters[i])
        dis = 0
        for j in range(n):
            dis += (points[i][j] - c[cluster][j]) ** 2
        dis = math.sqrt(dis)
        cost += dis
    return cost

