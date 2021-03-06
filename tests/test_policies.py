import pytest
import numpy as np

import Policies as po
import MatchingModel as mm


def test_thresholds_with_priorities_init():
    matching_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_order = mm.EdgesData(data=np.array([0, 2, 1]), matching_graph=matching_graph)
    thresholds = mm.EdgesData(data=np.array([3., 0., 0.]), matching_graph=matching_graph)
    policy = po.ThresholdsWithPriorities(state_space="state", matching_order=matching_order, thresholds=thresholds)

    assert policy.matching_order is matching_order
    assert policy.thresholds is policy.thresholds


def test_thresholds_with_priorities_init_different_matching_graph():
    matching_graph_o = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_graph_t = mm.MatchingGraph(edges=[(2, 1), (1, 1)], nb_demand_classes=2, nb_supply_classes=1)
    matching_order = mm.EdgesData(data=np.array([0, 2, 1]), matching_graph=matching_graph_o)
    thresholds = mm.EdgesData(data=np.array([3., 0.]), matching_graph=matching_graph_t)
    with pytest.raises(AssertionError):
        _ = po.ThresholdsWithPriorities(state_space="state", matching_order=matching_order, thresholds=thresholds)


def test_thresholds_with_priorities_match_n_end_priority():
    matching_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_order = mm.EdgesData(data=np.array([0, 2, 1]), matching_graph=matching_graph)
    thresholds = mm.EdgesData(data=np.array([3., 0., 0.]), matching_graph=matching_graph)
    policy = po.ThresholdsWithPriorities(state_space="state", matching_order=matching_order, thresholds=thresholds)

    x = mm.State(values=np.array([5., 5., 5., 5.]), matching_graph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([2., 5., 2., 5.])
    assert np.all(u.to_nodesdata() == u_theory)

    x = mm.State(values=np.array([0., 5., 5., 0.]), matching_graph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([0., 0., 0., 0.])
    assert np.all(u.to_nodesdata() == u_theory)

    x = mm.State(values=np.array([5., 0., 5., 0.]), matching_graph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([2., 0., 2., 0.])
    assert np.all(u.to_nodesdata() == u_theory)

    x = mm.State(values=np.array([0., 5., 0., 5.]), matching_graph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([0., 5., 0., 5.])
    assert np.all(u.to_nodesdata() == u_theory)

    x = mm.State(values=np.array([5., 0., 0., 5.]), matching_graph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([5., 0., 0., 5.])
    assert np.all(u.to_nodesdata() == u_theory)

    x = mm.State(values=np.array([2., 5., 2., 5.]), matching_graph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([0., 5., 0., 5.])
    assert np.all(u.to_nodesdata() == u_theory)

    x = mm.State(values=np.array([3., 4., 2., 5.]), matching_graph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([1., 4., 0., 5.])
    assert np.all(u.to_nodesdata() == u_theory)


@pytest.mark.skip(reason="The policy ThresholdWithPriorities is WIP")
def test_thresholds_with_priorities_match_n_middle_priority():
    matching_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_order = mm.EdgesData(data=np.array([2, 0, 1]), matching_graph=matching_graph)
    thresholds = mm.EdgesData(data=np.array([0., 2., 1.]), matching_graph=matching_graph)
    policy = po.ThresholdsWithPriorities(state_space="state", matching_order=matching_order, thresholds=thresholds)

    x = mm.State(values=np.array([5., 5., 5., 5.]), matching_graph=matching_graph)
    u = policy.match(x)
    u_theory = np.array([5., 1., 2., 4.])
    assert np.all(u.data == u_theory)


def test_thresholds_with_priorities_str():
    matching_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    matching_order = mm.EdgesData(data=np.array([0, 2, 1]), matching_graph=matching_graph)
    thresholds = mm.EdgesData(data=np.array([3., 0., 0.]), matching_graph=matching_graph)
    policy = po.ThresholdsWithPriorities(state_space="state", matching_order=matching_order, thresholds=thresholds)

    assert str(policy) == 'TwP p={} t={}'.format(matching_order, thresholds)


class TestNoMatchings:
    policy = po.NoMatchings(state_space="state")

    def test_match(self):
        matching_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)

        x = mm.State(values=np.array([1., 1., 1., 1.]), matching_graph=matching_graph)
        u = TestNoMatchings.policy.match(state=x)
        assert u == mm.Matching(state=x, values=np.array([0., 0., 0.]))

        x = mm.State(values=np.array([4., 0., 2., 2.]), matching_graph=matching_graph)
        u = TestNoMatchings.policy.match(state=x)
        assert u == mm.Matching(state=x, values=np.array([0., 0., 0.]))

        x = mm.State(values=np.array([4., 2., 0., 6.]), matching_graph=matching_graph)
        u = TestNoMatchings.policy.match(state=x)
        assert u == mm.Matching(state=x, values=np.array([0., 0., 0.]))


class TestThresholdN:
    policy = po.Threshold_N(state_space="state", threshold=3.)

    def test_init(self):
        threshold = 3.
        policy = po.Threshold_N(state_space="state", threshold=threshold)

        assert policy.threshold == threshold

    def test_match(self):
        matching_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)

        x = mm.State(values=np.array([1., 1., 1., 1.]), matching_graph=matching_graph)
        u = TestThresholdN.policy.match(state=x)
        assert u == mm.Matching(state=x, values=np.array([1., 0., 1.]))

        x = mm.State(values=np.array([4., 0., 2., 2.]), matching_graph=matching_graph)
        u = TestThresholdN.policy.match(state=x)
        assert u == mm.Matching(state=x, values=np.array([2., 0., 0.]))

        x = mm.State(values=np.array([4., 2., 0., 6.]), matching_graph=matching_graph)
        u = TestThresholdN.policy.match(state=x)
        assert u == mm.Matching(state=x, values=np.array([0., 1., 2.]))


class TestMaxWeight:

    def test_init(self):
        matching_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        costs = mm.NodesData(data=np.array([3., 1., 2., 3.]), matching_graph=matching_graph)
        policy = po.MaxWeight(state_space="state", costs=costs)

        assert policy.costs == costs

    def test_match_graph_n(self):
        matching_graph = mm.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        costs = mm.NodesData(data=np.array([3., 1., 2., 3.]), matching_graph=matching_graph)
        policy = po.MaxWeight(state_space="state", costs=costs)

        x = mm.State(values=np.array([1., 0., 0., 1.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([1., 0., 0., 1.]))

        x = mm.State(values=np.array([1., 0., 1., 0.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([1., 0., 1., 0.]))

        x = mm.State(values=np.array([0., 1., 0., 1.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([0., 1., 0., 1.]))

        x = mm.State(values=np.array([1., 1., 1., 1.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([1., 1., 1., 1.]))

        x = mm.State(values=np.array([10., 5., 3., 12.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([10., 5., 3., 12.]))

        x = mm.State(values=np.array([5., 10., 12., 3.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([5., 3., 5., 3.]))

    def test_match_graph_w(self):
        matching_graph = mm.MatchingGraph(edges=[(1, 1), (2, 1), (2, 2), (3, 2)], nb_demand_classes=3,
                                          nb_supply_classes=2)
        costs = mm.NodesData(data=np.array([5., 2., 1., 4., 2.]), matching_graph=matching_graph)
        policy = po.MaxWeight(state_space="state", costs=costs)

        x = mm.State(values=np.array([1., 0., 0., 1., 0.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([1., 0., 0., 1., 0.]))

        x = mm.State(values=np.array([0., 1., 0., 1., 0.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([0., 1., 0., 1., 0.]))

        x = mm.State(values=np.array([0., 1., 0., 0., 1.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([0., 1., 0., 0., 1.]))

        x = mm.State(values=np.array([0., 0., 1., 0., 1.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([0., 0., 1., 0., 1.]))

        x = mm.State(values=np.array([3., 5., 4., 5., 7.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([3., 5., 4., 5., 7.]))

        x = mm.State(values=np.array([3., 5., 4., 2., 10.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([2., 5., 4., 2., 9.]))

        x = mm.State(values=np.array([2., 0., 1., 0., 3.]), matching_graph=matching_graph)
        u = policy.match(state=x)
        assert np.all(u.to_nodesdata() == np.array([0., 0., 1., 0., 1.]))
