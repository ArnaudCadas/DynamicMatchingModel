import pytest
import numpy as np

import MatchingModel as Model


def test_edgedata_init():
    data = np.array([2., 0., -1.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData(data=data, matching_graph=matching_graph)

    assert ed.data is data
    assert ed.matching_graph is matching_graph


def test_edgedata_init_not_enough_data():
    data = np.array([2., 0.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    with pytest.raises(AssertionError):
        _ = Model.EdgeData(data=data, matching_graph=matching_graph)


def test_edgedata_from_dict():
    data = {(1, 1): 2., (1, 2): 0., (2, 2): -1.}
    data_array = np.array([data[(1, 1)], data[(1, 2)], data[(2, 2)]])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData.from_dict(data=data, matching_graph=matching_graph)

    assert np.all(ed.data == data_array)
    assert ed.matching_graph is matching_graph


def test_edgedata_from_dict_with_default():
    data = {(1, 1): 2., (1, 2): 0.}
    data_array = np.array([data[(1, 1)], data[(1, 2)], 0.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData.from_dict(data=data, matching_graph=matching_graph)

    assert np.all(ed.data == data_array)
    assert ed.matching_graph is matching_graph


def test_edgedata_from_dict_not_enough_data():
    data = {(3, 2): 2.}
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    with pytest.raises(ValueError):
        _ = Model.EdgeData.from_dict(data=data, matching_graph=matching_graph)


def test_edgedata_zeros():
    data = np.zeros(3)
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData.zeros(matching_graph=matching_graph)

    assert np.all(ed.data == data)
    assert ed.matching_graph is matching_graph


def test_edgedata_getitem():
    data = np.array([2., 0., -1.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData(data=data, matching_graph=matching_graph)

    assert ed[(1, 1)] == 2.
    assert ed[(1, 2)] == 0.
    assert ed[(2, 2)] == -1.


def test_edgedata_getitem_fail():
    data = np.array([2., 0., -1.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData(data=data, matching_graph=matching_graph)

    with pytest.raises(ValueError):
        ed[(3, 2)]


def test_edgedata_setitem():
    data = np.array([2., 0., -1.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData(data=data, matching_graph=matching_graph)

    ed[(1, 1)] = 5.
    ed[(1, 2)] = 4.
    ed[(2, 2)] = -3.
    assert ed[(1, 1)] == 5.
    assert ed[(1, 2)] == 4.
    assert ed[(2, 2)] == -3.


def test_edgedata_setitem_fail():
    data = np.array([2., 0., -1.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData(data=data, matching_graph=matching_graph)

    with pytest.raises(ValueError):
        ed[(3, 2)] = 9.


def test_edgedata_copy():
    data = np.array([2., 0., -1.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData(data=data, matching_graph=matching_graph)
    ed_copy = ed.copy()

    assert ed.matching_graph is ed_copy.matching_graph
    assert ed.data is not ed_copy.data
    assert np.all(ed.data == ed_copy.data)


def test_edgedata_str():
    data = np.array([2., 0., -1.])
    matching_graph = Model.MatchingGraph(edges=[(1, 1), (1, 2), (2, 2)], nb_demand_classes=2, nb_supply_classes=2)
    ed = Model.EdgeData(data=data, matching_graph=matching_graph)

    assert str(ed) == str(data)
