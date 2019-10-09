import numpy as np
import matplotlib.pyplot as plt
import time

import MatchingModel as Model
import Policies as Policies


W = Model.MatchingGraph([(1, 1), (2, 1), (2, 2), (3, 2)], 3, 2)
alpha = np.array([2./6., 2./6., 2./6.])
beta = np.array([1./2., 1./2.])
arrival_dist = Model.NodesData.items(alpha, beta, W)
costs = Model.NodesData(np.array([100., 1., 100., 10., 10.]), W)
P = [Policies.P14T23(thresholds=np.array([5, 5])),
     Policies.MaxWeight_policy(costs)]
x0 = Model.State.zeros(W)
test_model = Model.Model(W, arrival_dist, costs, x0)

plt.ion()

res = test_model.run(nb_iter=1000, policies=P, plot=True)

t = time.time()
N = 100000
c = test_model.average_cost(N, P, plot=True)
print(time.time()-t)

plt.ioff()
plt.show()

