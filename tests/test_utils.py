import pytest
import numpy as np

import utils as utils
import Policies as po
import MatchingModel as mm


class TestTransitionMatrix:
    N_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    epsilon = 0.1
    alpha = np.array([(1. / 2.) + epsilon, (1. / 2.) - epsilon])
    beta = np.array([(1. / 2.) - epsilon, (1. / 2.) + epsilon])
    arrival_dist = mm.NodesData.items(demand_items=alpha, supply_items=beta, matching_graph=N_graph)
    costs = mm.NodesData(data=np.array([1., 10., 10., 1.]), matching_graph=N_graph)
    capacity = 3.
    init_state = mm.State.zeros(matching_graph=N_graph, capacity=capacity)
    init_arrival = mm.State(values=np.array([1., 0., 1., 0.]), matching_graph=N_graph, capacity=capacity)
    N_model = mm.Model(matching_graph=N_graph, arrival_dist=arrival_dist, costs=costs, init_state=init_state,
                       init_arrival=init_arrival, capacity=capacity, penalty=100., state_space="state_with_arrival")

    def test_build_values(self):
        threshold = 0
        policy = po.Threshold_N(state_space="state_and_arrival", threshold=threshold)
        transition_matrix = utils.TransitionMatrix(model=TestTransitionMatrix.N_model, policy=policy)

        assert np.all(transition_matrix.values >= 0.)
        assert np.all(transition_matrix.values <= 1.)
        assert np.all(np.sum(transition_matrix.values, axis=1) == 1.)

