import numpy as np
import matplotlib.pyplot as plt
import time

import MatchingModel as mm
import Policies as po
import ReinforcementLearning as rl
import utils as utils


class Model_N_a:
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    capacity = 5.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0, capacity=capacity,
                     penalty=100., state_space="state_with_arrival")
    # Optimal threshold = 2


class Model_N_b:
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 50., 50., 1.]), matching_graph=N_graph)
    capacity = 5.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0, capacity=capacity,
                     penalty=100., state_space="state_with_arrival")
    # Optimal threshold = 4


class Model_N_a_discounted:
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    capacity = 5.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    init_arrival = mm.State(values=np.array([1., 0., 1., 0.]), matching_graph=N_graph,
                            capacity=capacity)
    model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0,
                     init_arrival=init_arrival, capacity=capacity,
                     penalty=100., state_space="state_with_arrival", discount=0.9)
    # Optimal threshold = 1


class Model_N_b_discounted:
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 20., 20., 1.]), matching_graph=N_graph)
    capacity = 5.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    init_arrival = mm.State(values=np.array([1., 0., 1., 0.]), matching_graph=N_graph,
                            capacity=capacity)
    model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0,
                     init_arrival=init_arrival, capacity=capacity,
                     penalty=100., state_space="state_with_arrival", discount=0.9)
    # Optimal threshold = 2


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
    np.random.seed(0)
    P = [po.Threshold_N(threshold=0, state_space="state_and_arrival"),
         po.Threshold_N(threshold=1, state_space="state_and_arrival"),
         po.Threshold_N(threshold=2, state_space="state_and_arrival"),
         po.Threshold_N(threshold=3, state_space="state_and_arrival"),
         po.Threshold_N(threshold=4, state_space="state_and_arrival"),
         po.Threshold_N(threshold=5, state_space="state_and_arrival")]
    test_model = Model_N_b.model

    plt.ion()

    res = test_model.run(nb_iter=1000, policies=P, plot=True)

    t = time.time()
    N = 100000
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
    np.random.seed(1)
    N_model = Model_N_b_discounted.model
    costs = N_model.costs
    value_iteration_result_file = "value_iteration_N_a_discounted_100.p"

    if do_value_iteration:
        print("Start of value iteration...")
        t = time.time()
        value_iteration = rl.ValueIteration(model=N_model)
        value_iteration.run(nb_iterations=nb_value_iterations, save_file=value_iteration_result_file, plot=True)
        print("End of value iteration, runtime: {}".format(time.time() - t))
        # For debug
        actor_l3_state_list = [mm.State(values=np.array([float(k), 0., 0., float(k)]),
                                        matching_graph=N_model.matching_graph, capacity=N_model.capacity)
                               for k in np.arange(N_model.capacity + 1)]
        actor_l4_state_list = [mm.State(values=np.array([0., float(k), float(k), 0.]),
                                        matching_graph=N_model.matching_graph, capacity=N_model.capacity)
                               for k in np.arange(N_model.capacity + 1)]
        actor_state_list = actor_l3_state_list + actor_l4_state_list[1:]
        arrival_list = [mm.State(values=np.array([1., 0., 1., 0.]), matching_graph=N_model.matching_graph,
                                 capacity=N_model.capacity),
                        mm.State(values=np.array([0., 1., 0., 1.]), matching_graph=N_model.matching_graph,
                                 capacity=N_model.capacity),
                        mm.State(values=np.array([1., 0., 0., 1.]), matching_graph=N_model.matching_graph,
                                 capacity=N_model.capacity),
                        mm.State(values=np.array([0., 1., 1., 0.]), matching_graph=N_model.matching_graph,
                                 capacity=N_model.capacity)]
        value_function_test = np.array([[value_iteration.V[state, arrivals] for state in actor_l3_state_list]
                                        for arrivals in arrival_list])

    P = [po.ValueIterationOptimal(state_space="state_and_arrival", model=N_model,
                                  load_file=value_iteration_result_file),
         po.MaxWeight(state_space="state_and_arrival", costs=costs)]

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


def demo_N_relative_value_iteration(do_value_iteration=False, nb_value_iterations=100):
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
                       capacity=capacity, penalty=100., state_space="state_with_arrival", init_arrival=init_arrival)
    relative_value_iteration_result_file = "relative_value_iteration_N_graph.p"

    if do_value_iteration:
        print("Start of relative value iteration...")
        t = time.time()
        relative_value_iteration = rl.RelativeValueIteration(model=N_model)
        relative_value_iteration.run(nb_iterations=nb_value_iterations, save_file=relative_value_iteration_result_file)
        print("End of relative value iteration, runtime: {}".format(time.time() - t))
        print("Is the solution optimal ? {}".format(relative_value_iteration.is_optimal()))

    P = [po.RelativeValueIterationOptimal(state_space="state_and_arrival", model=N_model,
                                          load_file=relative_value_iteration_result_file),
         po.MaxWeight(state_space="state_and_arrival", costs=costs)]
    print("For debug")

    plt.ion()

    res = N_model.run(nb_iter=1000, policies=P, plot=True)

    t = time.time()
    N = 10000
    c, x = N_model.average_cost(N, P, plot=True)
    print(time.time() - t)
    for i in range(len(P)):
        print(P[i], ": ", c[i][N])

    plt.ioff()
    plt.show()


def demo_N_relative_value_iteration_randomized_policy(nb_value_iterations=100):
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
                       capacity=capacity, penalty=100., state_space="state_with_arrival", init_arrival=init_arrival)
    policy = po.Threshold_N_norm_dist_all(state_space="state_and_arrival", threshold=2.5)
    relative_value_iteration_result_file = "relative_value_iteration_N_graph_randomized_policy.p"

    print("Start of relative value iteration...")
    t = time.time()
    relative_value_iteration = rl.RelativeValueIteration(model=N_model, policy=policy)
    relative_value_iteration.run(nb_iterations=nb_value_iterations, plot=True)
    print("End of relative value iteration, runtime: {}".format(time.time() - t))
    print("Is the solution optimal ? {}".format(relative_value_iteration.is_optimal()))


def demo_N_salmut():
    np.random.seed(1)
    N_model = Model_N_b.model

    # fast_time_scale = rl.BorkarFastTimeScale(power=0.8, shift=2., scale=100.)
    fast_time_scale = rl.ClassicTimeScale(power=0.75, scalar=1., shift=2.) # 0.9 in converged graphic
    # fast_time_scale = rl.ClassicTimeScale(power=0.9, scalar=1., shift=2.)
    slow_time_scale = rl.ClassicTimeScale(power=0.95, scalar=1.) # 0.7 in converged graphic
    approximation_time_scale = rl.LinearTimeScale(start=1., step=0.000005) # criticRVI LinearApproxSwitch ???K
    # approximation_time_scale = rl.LinearTimeScale(start=1., step=0.0000075) # criticRVI LinearApproxSwitch 150K
    # approximation_time_scale = rl.LinearTimeScale(start=1., step=0.) # NoApproxSwitch
    # approximation_time_scale = rl.LinearTimeScale(start=1., step=0.00001) # criticRVI LinearApproxSwitch 100K
    sigma = 0.5
    # N_salmut = rl.SalmutDBWithoutOccurrences(model=N_model, fast_time_scale=fast_time_scale,
    #                        slow_time_scale=slow_time_scale, approximation_time_scale=approximation_time_scale,
    #                        sigma=sigma)
    # N_salmut = rl.SalmutDBcriticRVI(model=N_model, fast_time_scale=fast_time_scale,
    #                                          slow_time_scale=slow_time_scale,
    #                                          approximation_time_scale=approximation_time_scale,
    #                                          sigma=sigma)
    N_salmut = rl.SalmutDBactorQ(model=N_model, fast_time_scale=fast_time_scale,
                                    slow_time_scale=slow_time_scale,
                                    approximation_time_scale=approximation_time_scale,
                                    sigma=sigma)


    ti = time.time()
    final_threshold, threshold_traj = N_salmut.run(nb_iterations=200000, plot=True, verbose=True)
    np.savetxt("Demo_N_b_seed1_SALMUT_DBactorQ_CorrectStepSizes075095scalar1_LinearApproxSwitch200K_200K.csv", threshold_traj, delimiter=",")
    print("Salmut has ended, runtime: {}, final threshold: {}".format(time.time() - ti, final_threshold))
    plt.show()


def demo_N_salmut_discounted():
    np.random.seed(1)
    N_model = Model_N_b_discounted.model

    # fast_time_scale = rl.BorkarFastTimeScale(power=0.8, shift=2., scale=100.)
    fast_time_scale = rl.ClassicTimeScale(power=0.6, scalar=1., shift=2.) # 0.8
    # fast_time_scale = rl.ClassicTimeScale(power=0.9, scalar=1., shift=2.)
    slow_time_scale = rl.ClassicTimeScale(power=0.7, scalar=1.) # 0.9
    approximation_time_scale = rl.LinearTimeScale(start=1.01, step=0.000005) # 0.000001
    sigma = 0.5
    N_salmut = rl.SalmutDBactorQDiscounted(
        model=N_model, fast_time_scale=fast_time_scale, slow_time_scale=slow_time_scale,
        approximation_time_scale=approximation_time_scale, sigma=sigma)

    ti = time.time()
    final_threshold, threshold_traj = N_salmut.run(nb_iterations=200000, plot=True, verbose=True)
    np.savetxt("Demo_N_b_discounted_seed1_SALMUT_DBactorQ_CorrectStepSizes0607scalar1_LinearApprox200K_200K.csv", threshold_traj, delimiter=",")
    print("Salmut has ended, runtime: {}, final threshold: {}".format(time.time() - ti, final_threshold))
    plt.show()


def demo_N_salmut_capacity3():
    np.random.seed(0)
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    capacity = 3.
    x0 = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    N_model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0, capacity=capacity,
                       penalty=100., state_space="state_with_arrival")

    # fast_time_scale = rl.BorkarFastTimeScale(power=0.8, shift=2., scale=100.)
    fast_time_scale = rl.ClassicTimeScale(power=0.8, scalar=1., shift=2.)
    slow_time_scale = rl.ClassicTimeScale(power=1., scalar=10.)
    N_salmut = rl.SalmutDBWithoutActor(model=N_model, fast_time_scale=fast_time_scale, slow_time_scale=slow_time_scale)

    ti = time.time()
    final_threshold = N_salmut.run(nb_iterations=100000, plot=True, verbose=True)
    print("Salmut has ended, runtime: {}, final threshold: {}".format(time.time() - ti, final_threshold))
    plt.show()


def demo_N_compute_optimal_threshold():
    N_model = Model_N_b.model

    optimal_threshold = utils.compute_optimal_threshold(model=N_model)
    print("Optimal threshold: {}".format(optimal_threshold))


def demo_N_randomized_policy_average_cost():
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    capacity = 5.
    init_state = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    init_arrival = mm.State(values=np.array([1., 0., 1., 0.]), matching_graph=N_graph, capacity=capacity)
    N_model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=init_state,
                       init_arrival=init_arrival, capacity=capacity, penalty=100., state_space="state_with_arrival")

    average_cost_array = utils.compute_optimal_threshold(model=N_model, return_array=True)

    threshold_list = np.linspace(1., 3., 20)
    sigma = 0.25
    policies_list = [po.Threshold_N_norm_dist_all(state_space="state_and_arrival", threshold=threshold, sigma=sigma)
                     for threshold in threshold_list]
    full_state = mm.State(values=np.array([5., 0., 0., 5.]), matching_graph=N_graph, capacity=capacity)
    empty_arrival = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    policies_average_cost_list = [np.dot(np.array([threshold_prob
                                  for _, threshold_prob in policy.distribution(state=full_state,
                                                                               arrivals=empty_arrival)]),
                                         average_cost_array.T)
                                  for policy in policies_list]
    print("for debug")


if __name__ == "__main__":
    # demo_N_randomized_policy_average_cost()
    # demo_N_compute_optimal_threshold()
    demo_N_salmut_discounted()
    # demo_N_salmut()
    # demo_N_salmut_capacity3()
    # demo_N_relative_value_iteration(do_value_iteration=False)
    # demo_N_relative_value_iteration_randomized_policy()
    # demo_N_value_iteration(do_value_iteration=True)
    # demo_N_with_capacity()
    # demo_N_with_capacity_discounted()
    # demo_N()
    # demo_W()
