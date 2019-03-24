import math
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats


def euclid_dist(p1, p2):
    out = 0
    for i in range(len(p1)-1):
        out += pow(p2[i] - p1[i], 2)
    return math.sqrt(out)


def knn(data, query, k):
    distances_and_indicies = []
    for index, datapoint in enumerate(data):
        distances_and_indicies.append((euclid_dist(datapoint, query), index, datapoint[2]))

    sorted_neighbors = sorted(distances_and_indicies, key=lambda x: x[0])
    candidates = sorted_neighbors[:k]

    print(candidates)
    candidate_labels = (list(map(lambda y: y[2], candidates)))

    classification = stats.mode(candidate_labels)

    return classification


mock_data = []
for i in range(50):
    x = np.random.random_integers(1, high=20)
    y = np.random.random_integers(1, high=20)

    # Mock labeling scheme: 1 if sum is odd, 0 if even
    if (x+y) % 2 != 0:
        label = 1
    else:
        label = 0
    mock_data.append([x, y, label])

query = [10,2]
print(knn(mock_data, query, 5))


data_x = list(map(lambda point: point[0], mock_data))
data_y = list(map(lambda point: point[1], mock_data))


plt.axis([0, 21, 0, 21])
plt.plot(data_x, data_y, 'b.')
plt.plot(query[0], query[1], 'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.suptitle('KNN Test')
plt.show()

