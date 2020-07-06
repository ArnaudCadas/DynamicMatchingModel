import numpy as np
import itertools
import json
from typing import Tuple, List

from MatchingModel import *


class ValueFunction:

    def __init__(self, model: Model):
        self.model = model
        assert np.isfinite(self.model.capacity)

        self.complete_arrival_graph_edges_list = [(demand_class, supply_class)
                                                  for demand_class in self.model.matching_graph.demand_class_set
                                                  for supply_class in self.model.matching_graph.supply_class_set]

        # We build the state space which is a list of all possible states
        self.state_space = []
        self.build_state_space()
        # We initialise the value function to 0 for each state in the state space.
        self.values = {}
        self.initialise_values()

    def build_state_space(self):
        # for arrival_numbers in itertools.combinations_with_replacement(range(int(self.model.capacity) + 1),
        #                                                                r=len(self.complete_arrival_graph_edges_list)):
        tuples_list = []
        for arrival_numbers in itertools.product(range(int(self.model.capacity) + 1),
                                                 repeat=len(self.complete_arrival_graph_edges_list)):
            state = State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
            try:
                for i, arrival_edge in enumerate(self.complete_arrival_graph_edges_list):
                    state[arrival_edge] += arrival_numbers[i]
            except ValueError:
                pass
            else:
                state_tuple = tuple(state.data)
                if state_tuple not in tuples_list:
                    for arrival_pair in self.complete_arrival_graph_edges_list:
                        arrivals = State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
                        arrivals[arrival_pair] += 1.
                        self.state_space.append((state, arrivals))
                        tuples_list.append(tuple(state.data))

    def initialise_values(self):
        for state, arrivals in self.state_space:
            self[state, arrivals] = 0.

    def __getitem__(self, item: Tuple[State, State]):
        state, arrivals = item
        if type(state) == State and type(arrivals) == State:
            return self.values[(tuple(state.data), tuple(arrivals.data))]
        else:
            return NotImplemented

    def __setitem__(self, item: Tuple[State, State], value):
        state, arrivals = item
        if type(state) == State and type(arrivals) == State:
            self.values[(tuple(state.data), tuple(arrivals.data))] = value
        else:
            return NotImplemented

    def copy(self):
        new_value_function = ValueFunction(model=self.model)
        for state, arrivals in self.state_space:
            new_value_function[state, arrivals] = self[state, arrivals]
        return new_value_function


class ValueIteration:

    def __init__(self, model: Model):
        self.model = model
        self.V = ValueFunction(model=self.model)

    def bellman_operator_with_matching(self, state: State, arrivals: State, matching: Matching):
        res = 0.
        if np.any(state.data + arrivals.data > self.model.capacity):
            new_state = state.copy()
            res += self.model.penalty
        else:
            new_state = state + arrivals
        res += np.dot(self.model.costs.data, new_state.data)
        for arrival_edge in self.V.complete_arrival_graph_edges_list:
            arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
            arrival = State.zeros(self.model.matching_graph, self.model.capacity)
            arrival[arrival_edge] += 1.
            res += self.model.discount * self.V[new_state - matching, arrival] * arrival_probability
        return res

    def bellman_operator(self, state: State, arrivals: State):
        res_for_all_matchings = []
        if np.any(state.data + arrivals.data > self.model.capacity):
            new_state = state.copy()
        else:
            new_state = state + arrivals
        for matching in new_state.complete_matchings_available():
            res_for_all_matchings.append(self.bellman_operator_with_matching(state=state, arrivals=arrivals,
                                                                             matching=matching))
        return np.min(res_for_all_matchings)

    def is_optimal(self, atol=1e-6):
        for state, arrivals in self.V.state_space:
            if not np.isclose(self.V[state, arrivals], self.bellman_operator(state=state, arrivals=arrivals),
                              atol=atol):
                return False
        return True

    def iterate(self):
        next_V = self.V.copy()
        for state, arrivals in self.V.state_space:
            next_V[state, arrivals] = self.bellman_operator(state=state, arrivals=arrivals)
        self.V = next_V.copy()

    def run(self, nb_iterations=None, save=False):
        self.V.initialise_values()
        if nb_iterations is None:
            while not self.is_optimal():
                self.iterate()
        else:
            for _ in np.arange(nb_iterations):
                self.iterate()
        if save:
            with open('value_iteration_result.json', 'w') as json_file:
                json.dump(self.V.values, json_file)
        return self.V


