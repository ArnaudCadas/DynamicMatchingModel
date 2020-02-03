import numpy as np
import matplotlib.pyplot as plt
import time

import MatchingModel as Model
import Policies as Policies

np.random.seed(42)
W = Model.MatchingGraph([(1, 1), (2, 1), (2, 2), (3, 2)], 3, 2)
epsilon = 0.1
alpha = np.array([(1./2.)-epsilon, (1./4.)+epsilon, (1./4.)])
beta = np.array([1./2., 1./2.])
arrival_dist = Model.NodesData.items(alpha, beta, W)
costs = Model.NodesData(np.array([10., 10., 1., 1., 1000.]), W)
P = [Policies.P14T23(thresholds=np.array([11, 0])),
     Policies.P13T24D2(thresholds=np.array([5, 14]))]
# P = [Policies.P14T23(thresholds=np.array([14, 0])),
#      Policies.OptimalW(thresholds=np.array([14])),
#      Policies.OptimalWBis(thresholds=np.array([14])),
#      Policies.P13T24D2(thresholds=np.array([5, 14]))]
# P = [Policies.P13T24D2(thresholds=np.array([5, 14])),
#      Policies.P13T24D2(thresholds=np.array([5, 15])),
#      Policies.OptimalW(thresholds=np.array([14])),
#      Policies.OptimalW(thresholds=np.array([15]))]
x0 = Model.State.zeros(W)
test_model = Model.Model(W, arrival_dist, costs, x0)

plt.ion()

res = test_model.run(nb_iter=1000, policies=P, plot=True)

t = time.time()
N = 1000000
c, x = test_model.average_cost(N, P, plot=True)
print(time.time()-t)
for i in range(len(P)):
    print(P[i], ": ", c[i][N])

plt.ioff()
plt.show()

