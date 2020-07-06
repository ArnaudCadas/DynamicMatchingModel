import pytest
import numpy as np
import itertools

import Policies as Policies
import MatchingModel as Model
import ReinforcementLearning as RL


class TestValueFunction:
    N_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = Model.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = Model.NodesData(data=np.array([10., 1., 1., 10.]), matching_graph=N_graph)
    capacity = 2.
    x0 = Model.State.zeros(matching_graph=N_graph, capacity=capacity)
    N_model = Model.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0, capacity=capacity,
                          penalty=100., state_space="state_with_arrival")

    def test_init(self):
        N_value_function_a = RL.ValueFunction(model=TestValueFunction.N_model)

        assert N_value_function_a.model is TestValueFunction.N_model
        assert np.all(N_value_function_a.complete_arrival_graph_edges_list == [(1, 1), (1, 2), (2, 1), (2, 2)])

        x_0_b = Model.State.zeros(matching_graph=TestValueFunction.N_graph, capacity=np.inf)
        N_model_b = Model.Model(matching_graph=TestValueIteration.N_graph,
                                arrival_dist=TestValueIteration.arrival_dist, costs=TestValueIteration.costs,
                                init_state=x_0_b, capacity=np.inf,
                                penalty=100., state_space="state_with_arrival")
        with pytest.raises(AssertionError):
            RL.ValueFunction(model=N_model_b)

    def test_build_state_space(self):
        N_value_function_a = RL.ValueFunction(model=TestValueFunction.N_model)
        state_space_theory_state_tuples = [(0., 0., 0., 0.), (0., 1., 0., 1.), (0., 2., 0., 2.), (0., 1., 1., 0.),
                                           (0., 2., 1., 1.), (0., 2., 2., 0.), (1., 0., 0., 1.), (1., 1., 0., 2.),
                                           (1., 1., 1., 1.), (1., 2., 1., 2.), (1., 2., 2., 1.), (2., 0., 0., 2.),
                                           (2., 1., 1., 2.), (2., 2., 2., 2.), (1., 0., 1., 0.), (1., 1., 2., 0.),
                                           (2., 0., 1., 1.), (2., 1., 2., 1.), (2., 0., 2., 0.)]
        state_space_theory_arrivals_tuples = [(1., 0., 1., 0.), (1., 0., 0., 1.), (0., 1., 1., 0.), (0., 1., 0., 1.)]
        state_space_theory = [(Model.State(values=np.array(state_tuple), matching_graph=TestValueFunction.N_graph,
                                           capacity=TestValueFunction.capacity),
                               Model.State(values=np.array(arrivals_tuple), matching_graph=TestValueFunction.N_graph,
                                           capacity=TestValueFunction.capacity))
                              for state_tuple, arrivals_tuple in itertools.product(state_space_theory_state_tuples,
                                                                                   state_space_theory_arrivals_tuples)]

        assert np.all(N_value_function_a.state_space == state_space_theory)

    def test_initialise_values(self):
        N_value_function_a = RL.ValueFunction(model=TestValueFunction.N_model)

        for state in N_value_function_a.state_space:
            assert N_value_function_a[state] == 0.

    def test_get_item(self):
        # TODO
        pass

    def test_set_item(self):
        # TODO
        pass

    def test_copy(self):
        # TODO
        pass


class TestValueIteration:
    N_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = Model.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = Model.NodesData(data=np.array([10., 1., 1., 10.]), matching_graph=N_graph)
    x0 = Model.State.zeros(matching_graph=N_graph, capacity=10.)
    N_model = Model.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=x0, capacity=10.,
                          penalty=100., state_space="state_with_arrival")

    def test_init(self):
        N_value_iteration = RL.ValueIteration(model=TestValueIteration.N_model)

        assert N_value_iteration.model is TestValueIteration.N_model

    def test_bellman_operator_with_matching(self):
        # TODO
        pass

    def test_bellman_operator(self):
        # TODO
        pass

    def test_is_optimal(self):
        # TODO
        pass

    def test_iterate(self):
        # TODO
        pass

    def test_run(self):
        # TODO
        pass

