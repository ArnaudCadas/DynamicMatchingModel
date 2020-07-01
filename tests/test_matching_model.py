import pytest
import numpy as np

import MatchingModel as Model


class TestMatchingGraph:
    N_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    # TODO: add another example such as the W-shaped graph

    def test_init(self):
        edges = [(1, 1), (1, 2), (2, 2)]
        nb_demand_classes = 2
        nb_supply_classes = 2
        N_graph = Model.MatchingGraph(edges=edges, nb_demand_classes=nb_demand_classes,
                                          nb_supply_classes=nb_supply_classes)
        demand_class_set = np.array([1, 2])
        demand_class_subsets = [(1,), (2,), (1, 2)]
        supply_class_set = np.array([1, 2])
        supply_class_subsets = [(1,), (2,), (1, 2)]

        assert N_graph.edges == edges
        assert N_graph.nb_demand_classes == nb_demand_classes
        assert N_graph.nb_supply_classes == nb_supply_classes
        assert np.all(N_graph.demand_class_set == demand_class_set)
        assert np.all(N_graph.demand_class_subsets == demand_class_subsets)
        assert np.all(N_graph.supply_class_set == supply_class_set)
        assert np.all(N_graph.supply_class_subsets == supply_class_subsets)

    def test_n(self):
        assert TestMatchingGraph.N_graph.n == 4

    def test_nb_edges(self):
        assert TestMatchingGraph.N_graph.nb_edges == 3

    def test_nodes(self):
        assert np.all(TestMatchingGraph.N_graph.nodes == ['d1', 'd2', 's1', 's2'])

    def test_build_demandToSupply(self):
        assert TestMatchingGraph.N_graph.demandToSupply == {(1,): [1, 2], (2,): [2], (1, 2): [1, 2]}

    def test_build_supplyToDemand(self):
        assert TestMatchingGraph.N_graph.supplyToDemand == {(1,): [1], (2,): [1, 2], (1, 2): [1, 2]}

    def test_isEdge(self):
        assert TestMatchingGraph.N_graph.isEdge((1, 1))
        assert TestMatchingGraph.N_graph.isEdge((1, 2))
        assert TestMatchingGraph.N_graph.isEdge((2, 2))
        assert not TestMatchingGraph.N_graph.isEdge((2, 1))
        assert not TestMatchingGraph.N_graph.isEdge((3, 5))

    def test_Dcomplement(self):
        assert TestMatchingGraph.N_graph.Dcomplement((1,)) == (2,)
        assert TestMatchingGraph.N_graph.Dcomplement((2,)) == (1,)
        assert not TestMatchingGraph.N_graph.Dcomplement((1, 2)) == (1,)

    def test_Scomplement(self):
        assert TestMatchingGraph.N_graph.Scomplement((1,)) == (2,)
        assert TestMatchingGraph.N_graph.Scomplement((2,)) == (1,)
        assert not TestMatchingGraph.N_graph.Scomplement((1, 2)) == (1,)

    def test_edgeIndex(self):
        assert TestMatchingGraph.N_graph.edgeIndex((1, 1)) == 0
        assert TestMatchingGraph.N_graph.edgeIndex((1, 2)) == 1
        assert TestMatchingGraph.N_graph.edgeIndex((2, 2)) == 2
        with pytest.raises(ValueError):
            TestMatchingGraph.N_graph.edgeIndex((2, 1))
        with pytest.raises(ValueError):
            TestMatchingGraph.N_graph.edgeIndex((3, 5))

    def test_degree(self):
        assert TestMatchingGraph.N_graph.degree() == Model.NodesData(np.array([2, 1, 1, 2]), TestMatchingGraph.N_graph)

    def test_maximal_matchings(self):
        # TODO
        pass

    def test_eq(self):
        edges = [(1, 1), (1, 2), (2, 2)]
        nb_demand_classes = 2
        nb_supply_classes = 2
        N_graph_bis = Model.MatchingGraph(edges=edges, nb_demand_classes=nb_demand_classes,
                                          nb_supply_classes=nb_supply_classes)

        assert TestMatchingGraph.N_graph == N_graph_bis


class TestNodesData:
    N_nodes = Model.NodesData(data=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)

    def test_init(self):
        data = np.array([5, 3, 3, 5])
        matchingGraph = TestMatchingGraph.N_graph
        N_nodes = Model.NodesData(data=data, matchingGraph=matchingGraph)

        assert np.all(N_nodes.data == data)
        assert N_nodes.matchingGraph == matchingGraph

    def test_fromDict(self):
        data_dict = {'d1': 5, 'd2': 3, 's1': 3, 's2': 5}
        data = np.array([5, 3, 3, 5])
        matchingGraph = TestMatchingGraph.N_graph
        N_nodes = Model.NodesData.fromDict(data=data_dict, matchingGraph=matchingGraph)

        assert np.all(N_nodes.data == data)
        assert N_nodes.matchingGraph == matchingGraph

    def test_zeros(self):
        data = np.zeros(4)
        matchingGraph = TestMatchingGraph.N_graph
        N_nodes = Model.NodesData.zeros(matchingGraph=matchingGraph)

        assert np.all(N_nodes.data == data)
        assert N_nodes.matchingGraph == matchingGraph

    def test_items(self):
        demand_items = np.array([5, 3])
        supply_items = np.array([3, 5])
        data = np.hstack((demand_items, supply_items))
        matchingGraph = TestMatchingGraph.N_graph
        N_nodes = Model.NodesData.items(demand_items=demand_items, supply_items=supply_items,
                                        matchingGraph=matchingGraph)

        assert np.all(N_nodes.data == data)
        assert N_nodes.matchingGraph == matchingGraph

    def test_demand(self):
        assert TestNodesData.N_nodes.demand(np.array([1])) == np.array([5])
        assert TestNodesData.N_nodes.demand(np.array([2])) == np.array([3])
        assert np.all(TestNodesData.N_nodes.demand(np.array([1, 2])) == np.array([5, 3]))

    def test_supply(self):
        assert TestNodesData.N_nodes.supply(np.array([1])) == np.array([3])
        assert TestNodesData.N_nodes.supply(np.array([2])) == np.array([5])
        assert np.all(TestNodesData.N_nodes.supply(np.array([1, 2])) == np.array([3, 5]))

    def test_getitem(self):
        assert np.all(TestNodesData.N_nodes[1, 1] == np.array([5, 3]))
        assert np.all(TestNodesData.N_nodes[1, 2] == np.array([5, 5]))
        assert np.all(TestNodesData.N_nodes[2, 2] == np.array([3, 5]))

    def test_setitem(self):
        N_nodes_copy = TestNodesData.N_nodes.copy()
        N_nodes_copy[1, 1] = np.array([-3, 18])
        assert np.all(N_nodes_copy.data == np.array([-3, 3, 18, 5]))
        N_nodes_copy[1, 2] = np.array([-7, 24])
        assert np.all(N_nodes_copy.data == np.array([-7, 3, 18, 24]))
        N_nodes_copy[2, 2] = np.array([4, -9])
        assert np.all(N_nodes_copy.data == np.array([-7, 4, 18, -9]))

    def test_add(self):
        N_nodes_a = Model.NodesData(data=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)
        N_nodes_b = Model.NodesData(data=np.array([-5, 4, -8, -1]), matchingGraph=TestMatchingGraph.N_graph)
        N_nodes_sum = Model.NodesData(data=np.array([0, 7, -5, 4]), matchingGraph=TestMatchingGraph.N_graph)

        assert N_nodes_sum == N_nodes_a + N_nodes_b

    def test_iadd(self):
        N_nodes_a = Model.NodesData(data=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)
        N_nodes_b = Model.NodesData(data=np.array([-5, 4, -8, -1]), matchingGraph=TestMatchingGraph.N_graph)
        N_nodes_sum = Model.NodesData(data=np.array([0, 7, -5, 4]), matchingGraph=TestMatchingGraph.N_graph)

        N_nodes_a += N_nodes_b
        assert N_nodes_sum == N_nodes_a

    def test_eq(self):
        N_nodes_a = Model.NodesData(data=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)

        assert TestNodesData.N_nodes == N_nodes_a

    def test_copy(self):
        N_nodes_copy = TestNodesData.N_nodes.copy()

        assert TestNodesData.N_nodes == N_nodes_copy
        assert TestNodesData.N_nodes.data is not N_nodes_copy.data


class TestState:
    N_state = Model.State(values=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)

    def test_init(self):
        values_a = np.array([5, 3, 3, 5])
        matchingGraph = TestMatchingGraph.N_graph
        N_state_a = Model.State(values=values_a, matchingGraph=matchingGraph)

        assert np.all(N_state_a.data == values_a)
        assert N_state_a.matchingGraph == matchingGraph

        values_b = np.array([-2, 3, 3, 5])
        with pytest.raises(ValueError):
            _ = Model.State(values=values_b, matchingGraph=matchingGraph)

        values_c = np.array([2, 3, 3, 5])
        with pytest.raises(ValueError):
            _ = Model.State(values=values_c, matchingGraph=matchingGraph)

    def test_matchings_available(self):
        matchings = [(1, 1), (1, 2), (2, 2)]
        assert np.all(TestState.N_state.matchings_available() == matchings)

        N_state_a = Model.State(values=np.array([5, 0, 5, 0]), matchingGraph=TestMatchingGraph.N_graph)
        matchings_a = [(1, 1)]
        assert np.all(N_state_a.matchings_available() == matchings_a)

        N_state_b = Model.State(values=np.array([5, 0, 0, 5]), matchingGraph=TestMatchingGraph.N_graph)
        matchings_b = [(1, 2)]
        assert np.all(N_state_b.matchings_available() == matchings_b)

        N_state_c = Model.State(values=np.array([2, 3, 0, 5]), matchingGraph=TestMatchingGraph.N_graph)
        matchings_c = [(1, 2), (2, 2)]
        assert np.all(N_state_c.matchings_available() == matchings_c)

    def test_matchings_available_subgraph(self):
        matchings_subgraph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2,
                                                 nb_supply_classes=2)
        assert TestState.N_state.matchings_available_subgraph() == matchings_subgraph

        N_state_a = Model.State(values=np.array([5, 0, 5, 0]), matchingGraph=TestMatchingGraph.N_graph)
        matchings_subgraph_a = Model.MatchingGraph(edges=[(1, 1)], nb_demand_classes=2, nb_supply_classes=2)
        assert np.all(N_state_a.matchings_available_subgraph() == matchings_subgraph_a)

        N_state_b = Model.State(values=np.array([5, 0, 0, 5]), matchingGraph=TestMatchingGraph.N_graph)
        matchings_subgraph_b = Model.MatchingGraph(edges=[(1, 2)], nb_demand_classes=2, nb_supply_classes=2)
        assert np.all(N_state_b.matchings_available_subgraph() == matchings_subgraph_b)

        N_state_c = Model.State(values=np.array([2, 3, 0, 5]), matchingGraph=TestMatchingGraph.N_graph)
        matchings_subgraph_c = Model.MatchingGraph(edges=[(1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        assert np.all(N_state_c.matchings_available_subgraph() == matchings_subgraph_c)

    def test_add(self):
        N_state_a = Model.State(values=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)
        N_state_b = Model.State(values=np.array([1, 2, 2, 1]), matchingGraph=TestMatchingGraph.N_graph)
        N_state_sum = Model.State(values=np.array([6, 5, 5, 6]), matchingGraph=TestMatchingGraph.N_graph)

        assert N_state_sum == N_state_a + N_state_b

        N_nodes_b = Model.NodesData(data=np.array([1, 2, 2, 1]), matchingGraph=TestMatchingGraph.N_graph)
        with pytest.raises(TypeError):
            N_state_a + N_nodes_b
        with pytest.raises(TypeError):
            N_nodes_b + N_state_a

    def test_iadd(self):
        N_state_a = Model.State(values=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)
        N_state_b = Model.State(values=np.array([1, 2, 2, 1]), matchingGraph=TestMatchingGraph.N_graph)
        N_state_sum = Model.State(values=np.array([6, 5, 5, 6]), matchingGraph=TestMatchingGraph.N_graph)

        N_state_a += N_state_b
        assert N_state_sum == N_state_a

        N_nodes_b = Model.NodesData(data=np.array([1, 2, 2, 1]), matchingGraph=TestMatchingGraph.N_graph)
        with pytest.raises(TypeError):
            N_state_a += N_nodes_b

    def test_sub(self):
        N_state_copy = TestState.N_state.copy()
        N_matching = Model.Matching(x=N_state_copy, values=np.array([2, 1, 1, 2]))
        N_state_diff = Model.State(values=np.array([3, 2, 2, 3]), matchingGraph=TestMatchingGraph.N_graph)

        assert N_state_diff == N_state_copy - N_matching

        N_nodes = Model.NodesData(data=np.array([2, 1, 1, 2]), matchingGraph=TestMatchingGraph.N_graph)
        with pytest.raises(TypeError):
            N_state_copy - N_nodes
        N_state_b = Model.State(values=np.array([2, 1, 1, 2]), matchingGraph=TestMatchingGraph.N_graph)
        with pytest.raises(TypeError):
            N_state_copy - N_state_b

    def test_isub(self):
        N_state_copy = TestState.N_state.copy()
        N_matching = Model.Matching(x=N_state_copy, values=np.array([2, 1, 1, 2]))
        N_state_diff = Model.State(values=np.array([3, 2, 2, 3]), matchingGraph=TestMatchingGraph.N_graph)

        N_state_copy -= N_matching
        assert N_state_diff == N_state_copy

        N_nodes = Model.NodesData(data=np.array([2, 1, 1, 2]), matchingGraph=TestMatchingGraph.N_graph)
        with pytest.raises(TypeError):
            N_state_copy -= N_nodes
        N_state_b = Model.State(values=np.array([2, 1, 1, 2]), matchingGraph=TestMatchingGraph.N_graph)
        with pytest.raises(TypeError):
            N_state_copy -= N_state_b

    def test_eq(self):
        N_state_a = Model.State(values=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)
        N_nodes_a = Model.NodesData(data=np.array([5, 3, 3, 5]), matchingGraph=TestMatchingGraph.N_graph)

        assert TestState.N_state == N_state_a
        assert not TestState.N_state == N_nodes_a


class TestMatching:
    # TODO
    pass


class TestEdgeData:
    def test_edgedata_init(self):
        data = np.array([2., 0., -1.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData(data=data, matching_graph=matching_graph)

        assert ed.data is data
        assert ed.matching_graph is matching_graph

    def test_edgedata_init_not_enough_data(self):
        data = np.array([2., 0.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        with pytest.raises(AssertionError):
            _ = Model.EdgeData(data=data, matching_graph=matching_graph)

    def test_edgedata_from_dict(self):
        data = {(1, 1): 2., (1, 2): 0., (2, 2): -1.}
        data_array = np.array([data[(1, 1)], data[(1, 2)], data[(2, 2)]])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData.from_dict(data=data, matching_graph=matching_graph)

        assert np.all(ed.data == data_array)
        assert ed.matching_graph is matching_graph

    def test_edgedata_from_dict_with_default(self):
        data = {(1, 1): 2., (1, 2): 0.}
        data_array = np.array([data[(1, 1)], data[(1, 2)], 0.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData.from_dict(data=data, matching_graph=matching_graph)

        assert np.all(ed.data == data_array)
        assert ed.matching_graph is matching_graph

    def test_edgedata_from_dict_not_enough_data(self):
        data = {(3, 2): 2.}
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        with pytest.raises(ValueError):
            _ = Model.EdgeData.from_dict(data=data, matching_graph=matching_graph)

    def test_edgedata_zeros(self):
        data = np.zeros(3)
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData.zeros(matching_graph=matching_graph)

        assert np.all(ed.data == data)
        assert ed.matching_graph is matching_graph

    def test_edgedata_getitem(self):
        data = np.array([2., 0., -1.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData(data=data, matching_graph=matching_graph)

        assert ed[(1, 1)] == 2.
        assert ed[(1, 2)] == 0.
        assert ed[(2, 2)] == -1.

    def test_edgedata_getitem_fail(self):
        data = np.array([2., 0., -1.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData(data=data, matching_graph=matching_graph)

        with pytest.raises(ValueError):
            ed[(3, 2)]

    def test_edgedata_setitem(self):
        data = np.array([2., 0., -1.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData(data=data, matching_graph=matching_graph)

        ed[(1, 1)] = 5.
        ed[(1, 2)] = 4.
        ed[(2, 2)] = -3.
        assert ed[(1, 1)] == 5.
        assert ed[(1, 2)] == 4.
        assert ed[(2, 2)] == -3.

    def test_edgedata_setitem_fail(self):
        data = np.array([2., 0., -1.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData(data=data, matching_graph=matching_graph)

        with pytest.raises(ValueError):
            ed[(3, 2)] = 9.

    def test_edgedata_copy(self):
        data = np.array([2., 0., -1.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData(data=data, matching_graph=matching_graph)
        ed_copy = ed.copy()

        assert ed.matching_graph is ed_copy.matching_graph
        assert ed.data is not ed_copy.data
        assert np.all(ed.data == ed_copy.data)

    def test_edgedata_str(self):
        data = np.array([2., 0., -1.])
        matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
        ed = Model.EdgeData(data=data, matching_graph=matching_graph)

        assert str(ed) == str(data)
