import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import Tuple, List

import MatchingModel as mm
import Policies as po


class ValueFunction:

    def __init__(self, model: mm.Model):
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
            state = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
            try:
                for i, arrival_edge in enumerate(self.complete_arrival_graph_edges_list):
                    state[arrival_edge] += arrival_numbers[i]
            except ValueError:
                pass
            else:
                state_tuple = tuple(state.data)
                if state_tuple not in tuples_list:
                    for arrival_pair in self.complete_arrival_graph_edges_list:
                        arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
                        arrivals[arrival_pair] += 1.
                        self.state_space.append((state, arrivals))
                        tuples_list.append(tuple(state.data))

    def initialise_values(self):
        for state, arrivals in self.state_space:
            self[state, arrivals] = 0.

    def __getitem__(self, item: Tuple[mm.State, mm.State]):
        state, arrivals = item
        if type(state) == mm.State and type(arrivals) == mm.State:
            return self.values[(tuple(state.data), tuple(arrivals.data))]
        else:
            return NotImplemented

    def __setitem__(self, item: Tuple[mm.State, mm.State], value):
        state, arrivals = item
        if type(state) == mm.State and type(arrivals) == mm.State:
            self.values[(tuple(state.data), tuple(arrivals.data))] = value
        else:
            return NotImplemented

    def copy(self):
        new_value_function = ValueFunction(model=self.model)
        for state, arrivals in self.state_space:
            new_value_function[state, arrivals] = self[state, arrivals]
        return new_value_function


class ValueIteration:

    def __init__(self, model: mm.Model):
        self.model = model
        self.V = ValueFunction(model=self.model)

    def bellman_operator_with_matching(self, state: mm.State, arrivals: mm.State, matching: mm.Matching):
        res = 0.
        if np.any(state.data + arrivals.data > self.model.capacity):
            new_state = state.copy()
            res += self.model.penalty
        else:
            new_state = state + arrivals
        res += np.dot(self.model.costs.data, new_state.data)
        for arrival_edge in self.V.complete_arrival_graph_edges_list:
            arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
            arrival = mm.State.zeros(self.model.matching_graph, self.model.capacity)
            arrival[arrival_edge] += 1.
            res += self.model.discount * self.V[new_state - matching, arrival] * arrival_probability
        return res

    def bellman_operator(self, state: mm.State, arrivals: mm.State):
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

    def run(self, nb_iterations=None, save_file=None):
        self.V.initialise_values()
        if nb_iterations is None:
            while not self.is_optimal():
                self.iterate()
        else:
            for _ in np.arange(nb_iterations):
                self.iterate()
        if save_file is not None:
            with open(save_file, 'wb') as pickle_file:
                pickle.dump(self, pickle_file)
        return self.V


class TimeScale:

    def __getitem__(self, n):
        raise NotImplementedError


class ClassicTimeScale(TimeScale):

    def __init__(self, power=1., scalar=1.):
        self.power = power
        self.scalar = scalar

    def __getitem__(self, n):
        return self.scalar / np.power(n, self.power)


class BorkarFastTimeScale(TimeScale):

    def __init__(self, power=1., shift=1., scale=1.):
        self.power = power
        self.shift = shift
        self.scale = scale

    def __getitem__(self, n):
        return 1 / np.power(np.floor(n / self.scale) + self.shift, self.power)


class Salmut:

    def __init__(self, model: mm.Model, fast_time_scale: TimeScale, slow_time_scale: TimeScale):
        self.model = model
        self.fast_time_scale = fast_time_scale
        self.slow_time_scale = slow_time_scale
        self.occurrence_count = ValueFunction(model=self.model)
        # We initialise the critic
        self.V = ValueFunction(model=self.model)
        self.state_ast, self.arrivals_ast = self.V.state_space[0]
        # We initialise the actor
        self.threshold = 0
        self.threshold_policy = po.Threshold_N(state_space="state_and_arrival", threshold=self.threshold)
        self.no_threshold_policy = po.Priority_N(state_space="state_and_arrival")

    def iterate_critic(self, current_state: mm.State, current_arrivals: mm.State):
        # We sample the next state and arrivals and get the costs of the current state and arrivals
        next_state_list, next_arrivals, costs_list = self.model.iterate_state_with_arrival(
            states_list=[current_state.copy()], arrivals=current_arrivals.copy(), policies=[self.threshold_policy])
        next_state = next_state_list[0]
        current_costs = costs_list[0]
        current_occurence_count = self.occurrence_count[current_state, current_arrivals]
        # We update the critic through Relative Value Iteration Algorithm
        self.V[current_state, current_arrivals] = (1. - self.fast_time_scale[current_occurence_count]) \
            * self.V[current_state, current_arrivals] + self.fast_time_scale[current_occurence_count] \
            * (current_costs + self.V[next_state, next_arrivals] - self.V[self.state_ast, self.arrivals_ast])
        return next_state, next_arrivals

    def grad_action_smoother(self, nb_possible_match_l3: float):
        # We compute the gradient
        grad = - np.exp(nb_possible_match_l3 - self.threshold) \
               / np.power(1. + np.exp(nb_possible_match_l3 - self.threshold), 2.)
        return grad

    def iterate_actor(self, current_state: mm.State, current_arrivals: mm.State, iteration: float):
        # We compute the current total state
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        current_total_state = current_state + current_arrivals
        # We compute the possible number of matchings that can be done on the edge where there is a threshold
        nb_possible_match_l3 = current_total_state.demand(1) - np.min(current_total_state[1, 1])
        # We sample the action chosen (match the threshold edge or not) and compute the new state
        no_matching_prob = np.random.rand()
        if np.random.rand() <= no_matching_prob or nb_possible_match_l3 <= self.threshold_policy.threshold:
            actor_matching = self.no_threshold_policy.match(state=current_state, arrivals=current_arrivals)
        else:
            actor_matching = self.threshold_policy.match(state=current_state, arrivals=current_arrivals)
        actor_state = current_total_state - actor_matching
        actor_arrival = self.model.sample_arrivals()
        # We update the actor through gradient based method
        new_threshold = self.threshold - self.slow_time_scale[iteration] \
            * self.grad_action_smoother(nb_possible_match_l3=nb_possible_match_l3) \
            * np.power(complex(-1., 0.), no_matching_prob).real * self.V[actor_state, actor_arrival]
        self.threshold = np.minimum(np.maximum(0., new_threshold), self.model.capacity)
        self.threshold_policy.threshold = self.threshold

    def iterate(self, current_state: mm.State, current_arrivals: mm.State, iteration):
        # We update the critic
        next_state, next_arrivals = self.iterate_critic(current_state=current_state, current_arrivals=current_arrivals)
        # We update the actor
        self.iterate_actor(current_state=current_state, current_arrivals=current_arrivals, iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state, next_arrivals] += 1.
        return next_state, next_arrivals

    def is_optimal(self, atol=0.05):
        pass

    def run(self, nb_iterations=None, plot=False):
        self.occurrence_count.initialise_values()
        self.V.initialise_values()
        self.threshold = 0
        self.threshold_policy.threshold = self.threshold
        current_state = self.model.init_state.copy()
        current_arrivals = self.model.init_arrival.copy()
        if plot:
            if nb_iterations is None:
                threshold_traj = []
                threshold_traj.append(self.threshold)
                iteration = 1.
                while not self.is_optimal():
                    current_state, current_arrivals = self.iterate(current_state=current_state,
                                                                   current_arrivals=current_arrivals, iteration=iteration)
                    threshold_traj.append(self.threshold)
                    iteration += 1.
                threshold_traj = np.array(threshold_traj)
            else:
                threshold_traj = np.zeros(nb_iterations + 1)
                threshold_traj[0] = self.threshold
                for iteration in np.arange(1, nb_iterations + 1):
                    current_state, current_arrivals = self.iterate(current_state=current_state,
                                                                   current_arrivals=current_arrivals, iteration=iteration)
                    threshold_traj[iteration] = self.threshold
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.plot(threshold_traj, label="threshold")
            ax.legend(loc='best')
            ax.set_title("Threshold trajectory")
            fig.canvas.draw()
        else:
            if nb_iterations is None:
                iteration = 1.
                while not self.is_optimal():
                    current_state, current_arrivals = self.iterate(current_state=current_state,
                                                                   current_arrivals=current_arrivals,
                                                                   iteration=iteration)
                    iteration += 1.
            else:
                for iteration in np.arange(1, nb_iterations + 1):
                    current_state, current_arrivals = self.iterate(current_state=current_state,
                                                                   current_arrivals=current_arrivals,
                                                                   iteration=iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
        return self.threshold


