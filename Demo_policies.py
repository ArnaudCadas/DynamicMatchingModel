import numpy as np
import matplotlib.pyplot as plt
import time

import MatchingModel as Model
import Policies as Policies
import ReinforcementLearning as RL

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
    N_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = Model.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = Model.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    P = [Policies.Threshold_N(threshold=0.),
         Policies.Threshold_N(threshold=1.),
         Policies.MaxWeight(costs=costs),
         Policies.NoMatchings()]
    capacity = 5.
    x0 = Model.State.zeros(matching_graph=N_graph, capacity=capacity)
    test_model = Model.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0,
                             capacity=capacity, penalty=100., state_space="state_with_arrival")

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


def demo_N_with_capacity_discounted():
    np.random.seed(42)
    N_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = Model.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = Model.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    P = [Policies.Threshold_N(threshold=0.),
         Policies.Threshold_N(threshold=1.),
         Policies.MaxWeight(costs=costs)]
    capacity = 5.
    x0 = Model.State.zeros(matching_graph=N_graph, capacity=capacity)
    test_model = Model.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0,
                             capacity=capacity, penalty=100., state_space="state_with_arrival", discount=0.9)

    plt.ion()

    res = test_model.run(nb_iter=1000, policies=P, plot=True)

    t = time.time()
    N = 10000
    c, x = test_model.discounted_cost(N, P, plot=True)
    print(time.time() - t)
    for i in range(len(P)):
        print(P[i], ": ", c[i][N])

    plt.ioff()
    plt.show()


def demo_N_value_iteration():
    N_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = Model.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = Model.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    capacity = 5.
    x0 = Model.State.zeros(matching_graph=N_graph, capacity=capacity)
    N_model = Model.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0,
                          capacity=capacity, penalty=100., state_space="state_with_arrival", discount=0.9)
    value_iteration = RL.ValueIteration(model=N_model)
    print(value_iteration.V.values)

    print("Create policy based on value iteration...")
    t = time.time()
    p = Policies.ValueIterationOptimal(model=N_model, nb_iterations=100)
    print("End of creation, runtime: {}".format(time.time() - t))
    print("For debug")

    # t = time.time()
    # nb_iterations = 100
    # value_iteration.run(nb_iterations=1)
    # for i in np.arange(1, nb_iterations):
    #     print("\n##### ITERATION {} #####".format(i))
    #     print("time from start: {}".format(time.time() - t))
    #     print("V[((0., 0., 0., 0.), (1., 0., 1., 0.))]: {}".format(
    #         value_iteration.V.values[((0., 0., 0., 0.), (1., 0., 1., 0.))]))
    #     print(value_iteration.is_optimal())
    #     value_iteration.iterate()


if __name__ == "__main__":
    demo_N_value_iteration()
    # demo_N_with_capacity()
    # demo_N_with_capacity_discounted()
    # demo_N()
    # demo_W()
