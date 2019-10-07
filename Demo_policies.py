import numpy as np
import matplotlib.pyplot as plt
import time

import MatchingModel as Model
import Policies as Policies


W = Model.MatchingGraph([(1, 1), (2, 1), (2, 2), (3, 1)], 3, 2)
alpha = np.array([2./6., 2./6., 2./6.])
beta = np.array([1./2., 1./2.])
arrival_dist = Model.NodesData.items(alpha, beta, W)
costs = Model.NodesData(np.array([1., 1000., 1., 10., 10.]), W)
P = [Policies.ThresholdsWithPriorities(matching_order=Model.EdgeData([0, 3, 1, 2], W),
                                       thresholds=Model.EdgeData([0, 5, 5, 0], W)),
     Policies.MaxWeight_policy(costs)]
x0 = Model.State.zeros(W)
test_model = Model.Model(W, arrival_dist, costs, x0)

plt.ion()

res = test_model.run(1000, P, plot=True)

t = time.time()
N = 100000
c = test_model.average_cost(N, P, plot=True)
print(time.time()-t)

plt.ioff()
plt.show()

