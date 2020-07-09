import numpy as np
import matplotlib.pyplot as plt
import time

import MatchingModel as mm
import Policies as po
import ReinforcementLearning as rl


def demo_W():
    np.random.seed(42)
    W = mm.MatchingGraph([(1, 1), (2, 1), (2, 2), (3, 2)], 3, 2)
    epsilon = 0.1
    alpha = np.array([(1./2.)-epsilon, (1./4.)+epsilon, (1./4.)])
    beta = np.array([1./2., 1./2.])
    arrival_dist = mm.NodesData.items(alpha, beta, W)
    costs = mm.NodesData(np.array([10., 10., 1., 1., 1000.]), W)
    P = [po.P14T23(thresholds=np.array([11, 0])),
         po.P13T24D2(thresholds=np.array([5, 14]))]
    # P = [po.P14T23(thresholds=np.array([14, 0])),
    #      po.OptimalW(thresholds=np.array([14])),
    #      po.OptimalWBis(thresholds=np.array([14])),
    #      po.P13T24D2(thresholds=np.array([5, 14]))]
    # P = [po.P13T24D2(thresholds=np.array([5, 14])),
    #      po.P13T24D2(thresholds=np.array([5, 15])),
    #      po.OptimalW(thresholds=np.array([14])),
    #      po.OptimalW(thresholds=np.array([15]))]
    x0 = mm.State.zeros(W)
    test_model = mm.Model(W, arrival_dist, costs, x0)

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
    N_graph = mm.MatchingGraph([(1, 1), (1, 2), (2, 2)], 2, 2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(alpha, beta, N_graph)
    costs = mm.NodesData(np.array([10., 1., 1., 10.]), N_graph)
    P = [po.Threshold_N(threshold=3.),
         po.MaxWeight(costs=costs)]
    x0 = mm.State.zeros(N_graph)
    test_model = mm.Model(N_graph, arrival_dist, costs, x0)

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
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    P = [po.Threshold_N(threshold=1, state_space="state_and_arrival"),
         po.Threshold_N(threshold=2, state_space="state_and_arrival"),
         po.Threshold_N(threshold=3, state_space="state_and_arrival"),
         po.Threshold_N(threshold=4, state_space="state_and_arrival")]
    capacity = 5.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    test_model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0,
                          capacity=capacity, penalty=100., state_space="state_with_arrival")

    plt.ion()

    res = test_model.run(nb_iter=1000, policies=P, plot=True)

    t = time.time()
    N = 1000000
    c, x = test_model.average_cost(N, P, plot=True)
    print(time.time() - t)
    for i in range(len(P)):
        print(P[i], ": ", c[i][N])

    plt.ioff()
    plt.show()


def demo_N_with_capacity_discounted():
    np.random.seed(42)
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    P = [po.Threshold_N(threshold=0.),
         po.Threshold_N(threshold=1.),
         po.MaxWeight(costs=costs)]
    capacity = 5.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    test_model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0,
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


def demo_N_value_iteration(do_value_iteration=False, nb_value_iterations=100):
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    capacity = 5.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    init_arrival = mm.State(values=np.array([0., 1., 1., 0.]), matching_graph=N_graph, capacity=capacity)
    N_model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0,
                       capacity=capacity, penalty=100., state_space="state_with_arrival", discount=0.9,
                       init_arrival=init_arrival)
    value_iteration_result_file = "value_iteration_N_graph.p"

    if do_value_iteration:
        print("Start of value iteration...")
        t = time.time()
        value_iteration = rl.ValueIteration(model=N_model)
        value_iteration.run(nb_iterations=nb_value_iterations, save_file=value_iteration_result_file)
        print("End of value iteration, runtime: {}".format(time.time() - t))

    P = [po.ValueIterationOptimal(state_space="state_and_arrival", model=N_model,
                                  load_file=value_iteration_result_file),
         po.MaxWeight(state_space="state_and_arrival", costs=costs)]
    print("For debug")

    plt.ion()

    res = N_model.run(nb_iter=1000, policies=P, plot=True)

    t = time.time()
    N = 10000
    c, x = N_model.discounted_cost(N, P, plot=True)
    print(time.time() - t)
    for i in range(len(P)):
        print(P[i], ": ", c[i][N])

    plt.ioff()
    plt.show()

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


def demo_N_salmut():
    np.random.seed(42)
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    capacity = 5.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    N_model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0, capacity=capacity,
                       penalty=100., state_space="state_with_arrival")

    fast_time_scale = rl.BorkarFastTimeScale(power=0.6, shift=2., scale=100.)
    slow_time_scale = rl.ClassicTimeScale(power=1., scalar=10.)
    N_salmut = rl.Salmut(model=N_model, fast_time_scale=fast_time_scale, slow_time_scale=slow_time_scale)

    ti = time.time()
    final_threshold = N_salmut.run(nb_iterations=1000000, plot=True)
    print("Salmut has ended, runtime: {}, final threshold: {}".format(time.time() - ti, final_threshold))
    plt.show()


if __name__ == "__main__":
    demo_N_salmut()
    # demo_N_value_iteration(do_value_iteration=False)
    # demo_N_with_capacity()
    # demo_N_with_capacity_discounted()
    # demo_N()
    # demo_W()
