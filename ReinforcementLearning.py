import numpy as np
import itertools
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
                    self.state_space.append(state)
                    tuples_list.append(tuple(state.data))

    def initialise_values(self):
        for state in self.state_space:
            self[state] = 0.

    def __getitem__(self, state: State):
        if type(state) == State:
            return self.values[tuple(state.data)]
        else:
            return NotImplemented

    def __setitem__(self, state: State, value):
        if type(state) == State:
            self.values[tuple(state.data)] = value
        else:
            return NotImplemented

    def copy(self):
        new_value_function = ValueFunction(model=self.model)
        for state in self.state_space:
            new_value_function[state] = self[state]
        return new_value_function


class ValueIteration:

    def __init__(self, model: Model):
        self.model = model
        self.V = ValueFunction(model=self.model)

    def bellman_operator_with_matching(self, state: State, matching: Matching):
        res = np.dot(self.model.costs.data, state.data)
        for arrival_edge in self.V.complete_arrival_graph_edges_list:
            arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
            arrival = State.zeros(self.model.matching_graph, self.model.capacity)
            arrival[arrival_edge] += 1.
            res += self.model.discount * self.V[state - matching + arrival] * arrival_probability
        return res

    def bellman_operator(self, state: State):
        res_for_all_matchings = []
        for matching in state.complete_matchings_available():
            res_for_all_matchings.append(self.bellman_operator_with_matching(state=state, matching=matching))
        return np.max(res_for_all_matchings)

    def is_optimal(self, atol=1e-6):
        for state in self.V.state_space:
            if not np.isclose(self.V[state], self.bellman_operator(state=state), atol=atol):
                return False
        return True

    def iterate(self):
        next_V = self.V.copy()
        for state in self.V.state_space:
            next_V[state] = self.bellman_operator(state=state)
        self.V = next_V.copy()

    def run(self, nb_iterations=None):
        self.V.initialise_values()
        if nb_iterations is None:
            while not self.is_optimal():
                self.iterate()
        else:
            for _ in np.arange(nb_iterations):
                self.iterate()
        return self.V


