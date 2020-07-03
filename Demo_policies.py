import numpy as np
import matplotlib.pyplot as plt
import time

import MatchingModel as Model
import Policies as Policies

def demo_W():
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
    N = 10000
    c, x = test_model.average_cost(N, P, plot=True)
    print(time.time()-t)
    for i in range(len(P)):
        print(P[i], ": ", c[i][N])

    plt.ioff()
    plt.show()


def demo_N():
    np.random.seed(42)
    N_graph = Model.MatchingGraph([(1, 1), (1, 2), (2, 2)], 2, 2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = Model.NodesData.items(alpha, beta, N_graph)
    costs = Model.NodesData(np.array([10., 1., 1., 10.]), N_graph)
    P = [Policies.Threshold_N(threshold=3.),
         Policies.MaxWeight(costs=costs)]
    x0 = Model.State.zeros(N_graph)
    test_model = Model.Model(N_graph, arrival_dist, costs, x0)

    plt.ion()

    res = test_model.run(nb_iter=1000, policies=P, plot=True)

    t = time.time()
    N = 10000
    c, x = test_model.average_cost(N, P, plot=True)
    print(time.time() - t)
    for i in range(len(P)):
        print(P[i], ": ", c[i][N])

    plt.ioff()
    plt.show()


def demo_N_with_capacity():
    np.random.seed(42)
    N_graph = Model.MatchingGraph([(1, 1), (1, 2), (2, 2)], 2, 2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = Model.NodesData.items(alpha, beta, N_graph)
    costs = Model.NodesData(np.array([10., 1., 1., 10.]), N_graph)
    P = [Policies.Threshold_N(threshold=3.),
         Policies.MaxWeight(costs=costs),
         Policies.NoMatchings()]
    x0 = Model.State.zeros(N_graph, 10.)
    test_model = Model.Model(N_graph, arrival_dist, costs, x0, 10., 100.)

    plt.ion()

    res = test_model.run(nb_iter=1000, policies=P, plot=True)

    t = time.time()
    N = 10000
    c, x = test_model.average_cost(N, P, plot=True)
    print(time.time() - t)
    for i in range(len(P)):
        print(P[i], ": ", c[i][N])

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    demo_N_with_capacity()
    # demo_N()
    # demo_W()
