import pytest
import numpy as np

import MatchingModel as Model
import Policies as Policies


def test_thresholds_with_priorities_init():
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_order = Model.EdgeData(data=np.array([0, 2, 1]), matching_graph=matching_graph)
    thresholds = Model.EdgeData(data=np.array([3., 0., 0.]), matching_graph=matching_graph)
    policy = Policies.ThresholdsWithPriorities(matching_order=matching_order, thresholds=thresholds)

    assert policy.matching_order is matching_order
    assert policy.thresholds is policy.thresholds


def test_thresholds_with_priorities_init_different_matching_graph():
    matching_graph_o = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_graph_t = Model.MatchingGraph(edges=[(2, 1), (1, 1)], nb_demand_classes=2, nb_supply_classes=1)
    matching_order = Model.EdgeData(data=np.array([0, 2, 1]), matching_graph=matching_graph_o)
    thresholds = Model.EdgeData(data=np.array([3., 0.]), matching_graph=matching_graph_t)
    with pytest.raises(AssertionError):
        _ = Policies.ThresholdsWithPriorities(matching_order=matching_order, thresholds=thresholds)


def test_thresholds_with_priorities_match_n_end_priority():
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_order = Model.EdgeData(data=np.array([0, 2, 1]), matching_graph=matching_graph)
    thresholds = Model.EdgeData(data=np.array([3., 0., 0.]), matching_graph=matching_graph)
    policy = Policies.ThresholdsWithPriorities(matching_order=matching_order, thresholds=thresholds)

    x = Model.State(values=np.array([5., 5., 5., 5.]), matchingGraph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([2., 5., 2., 5.])
    assert np.all(u.data == u_theory)

    x = Model.State(values=np.array([0., 5., 5., 0.]), matchingGraph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([0., 0., 0., 0.])
    assert np.all(u.data == u_theory)

    x = Model.State(values=np.array([5., 0., 5., 0.]), matchingGraph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([2., 0., 2., 0.])
    assert np.all(u.data == u_theory)

    x = Model.State(values=np.array([0., 5., 0., 5.]), matchingGraph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([0., 5., 0., 5.])
    assert np.all(u.data == u_theory)

    x = Model.State(values=np.array([5., 0., 0., 5.]), matchingGraph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([5., 0., 0., 5.])
    assert np.all(u.data == u_theory)

    x = Model.State(values=np.array([2., 5., 2., 5.]), matchingGraph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([0., 5., 0., 5.])
    assert np.all(u.data == u_theory)

    x = Model.State(values=np.array([3., 4., 2., 5.]), matchingGraph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([1., 4., 0., 5.])
    assert np.all(u.data == u_theory)


def test_thresholds_with_priorities_match_n_middle_priority():
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_order = Model.EdgeData(data=np.array([2, 0, 1]), matching_graph=matching_graph)
    thresholds = Model.EdgeData(data=np.array([0., 2., 1.]), matching_graph=matching_graph)
    policy = Policies.ThresholdsWithPriorities(matching_order=matching_order, thresholds=thresholds)

    x = Model.State(values=np.array([5., 5., 5., 5.]), matchingGraph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([5., 1., 2., 4.])
    assert np.all(u.data == u_theory)


def test_thresholds_with_priorities_str():
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_order = Model.EdgeData(data=np.array([0, 2, 1]), matching_graph=matching_graph)
    thresholds = Model.EdgeData(data=np.array([3., 0., 0.]), matching_graph=matching_graph)
    policy = Policies.ThresholdsWithPriorities(matching_order=matching_order, thresholds=thresholds)

    assert str(policy) == 'TwP p={} t={}'.format(matching_order, thresholds)


def test_maxweight_init():
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    costs = Model.NodesData(A=np.array([3., 1., 2., 3.]), matchingGraph=matching_graph)
    policy = Policies.MaxWeight(costs=costs)

    assert policy.costs is costs


def test_maxweight_match_graph_n():
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    costs = Model.NodesData(A=np.array([3., 1., 2., 3.]), matchingGraph=matching_graph)
    policy = Policies.MaxWeight(costs=costs)

    x = Model.State(values=np.array([1., 0., 0., 1.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([1., 0., 0., 1.]))

    x = Model.State(values=np.array([1., 0., 1., 0.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([1., 0., 1., 0.]))

    x = Model.State(values=np.array([0., 1., 0., 1.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([0., 1., 0., 1.]))

    x = Model.State(values=np.array([1., 1., 1., 1.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([1., 1., 1., 1.]))

    x = Model.State(values=np.array([10., 5., 3., 12.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([10., 5., 3., 12.]))

    x = Model.State(values=np.array([5., 10., 12., 3.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([5., 3., 5., 3.]))


def test_maxweight_match_graph_w():
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (2, 1), (2, 2), (3, 2)], nb_demand_classes=3,
                                         nb_supply_classes=2)
    costs = Model.NodesData(A=np.array([5., 2., 1., 4., 2.]), matchingGraph=matching_graph)
    policy = Policies.MaxWeight(costs=costs)

    x = Model.State(values=np.array([1., 0., 0., 1., 0.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([1., 0., 0., 1., 0.]))

    x = Model.State(values=np.array([0., 1., 0., 1., 0.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([0., 1., 0., 1., 0.]))

    x = Model.State(values=np.array([0., 1., 0., 0., 1.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([0., 1., 0., 0., 1.]))

    x = Model.State(values=np.array([0., 0., 1., 0., 1.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([0., 0., 1., 0., 1.]))

    x = Model.State(values=np.array([3., 5., 4., 5., 7.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([3., 5., 4., 5., 7.]))

    x = Model.State(values=np.array([3., 5., 4., 2., 10.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([2., 5., 4., 2., 9.]))

    x = Model.State(values=np.array([2., 0., 1., 0., 3.]), matchingGraph=matching_graph)
    u = policy.match(x=x)
    assert np.all(u.data == np.array([0., 0., 1., 0., 1.]))
