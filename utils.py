import numpy as np
import sympy as sp
import itertools
from typing import Tuple
import sys, time

import MatchingModel as mm
import Policies as po


class TransitionMatrix:

    def __init__(self, model: mm.Model, policy):
        self.model = model
        assert np.isfinite(self.model.capacity)
        self.policy = policy

        self.complete_arrival_graph_edges_list = [(demand_class, supply_class)
                                                  for demand_class in self.model.matching_graph.demand_class_set
                                                  for supply_class in self.model.matching_graph.supply_class_set]

        # We build the state space which is a list of all possible states
        self.state_space = []
        self.build_state_space()
        self.nb_states = len(self.state_space)
        # We initialise the transition matrix to 0 for each (state, other_state) in the state space.
        self.values = np.zeros((self.nb_states, self.nb_states))
        self.build_values()

    def build_state_space(self):
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

    def build_values(self):
        for i in np.arange(self.nb_states):
            for j in np.arange(self.nb_states):
                state, arrivals = self.state_space[i]
                next_state, next_arrivals = self.state_space[j]

                if np.any(state.data + arrivals.data > self.model.capacity):
                    arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
                if next_state == (state + arrivals) - self.policy.match(state=state, arrivals=arrivals):
                    demand_next_arrival = np.where(
                        next_arrivals.demand(self.model.matching_graph.demand_class_set) == 1.)[0].item() + 1
                    supply_next_arrival = np.where(
                        next_arrivals.supply(self.model.matching_graph.supply_class_set) == 1.)[0].item() + 1
                    self.values[i, j] = np.prod(self.model.arrival_dist[demand_next_arrival, supply_next_arrival])

    # def __getitem__(self, item: Tuple[Tuple[mm.State, mm.State], Tuple[mm.State, mm.State]]):
    #     (state, arrivals), (next_state, next_arrivals) = item
    #     if type(state) == mm.State and type(arrivals) == mm.State and type(next_state) == mm.State and \
    #             type(next_arrivals) == mm.State:
    #         return self.values[(tuple(state.data), tuple(arrivals.data))]
    #     else:
    #         return NotImplemented
    #
    # def __setitem__(self, item: Tuple[mm.State, mm.State], value):
    #     state, arrivals = item
    #     if type(state) == mm.State and type(arrivals) == mm.State:
    #         self.values[(tuple(state.data), tuple(arrivals.data))] = value
    #     else:
    #         return NotImplemented
    #
    # def copy(self):
    #     new_value_function = ValueFunction(model=self.model)
    #     for state, arrivals in self.state_space:
    #         new_value_function[state, arrivals] = self[state, arrivals]
    #     return new_value_function


def compute_optimal_threshold(model: mm.Model):
    average_cost_array = np.zeros(int(model.capacity + 1))
    for threshold in np.arange(int(model.capacity) + 1):
        # We compute the stationary distribution
        policy = po.Threshold_N(state_space="state_and_arrival", threshold=threshold)
        transition_matrix = TransitionMatrix(model=model, policy=policy)
        # nb_states = transition_matrix.nb_states
        # distribution_constraint = np.ones((1, nb_states))
        # linear_system_coef = np.vstack((transition_matrix.values - np.eye(nb_states), distribution_constraint))
        # linear_system_coef = transition_matrix.values - np.eye(nb_states)
        # linear_system_ordinate = np.vstack((np.zeros((nb_states, 1)), np.ones((1, 1))))
        # linear_system_ordinate = np.zeros((nb_states, 1))
        # stationary_dist = np.linalg.solve(a=linear_system_coef, b=linear_system_ordinate)
        # solution = np.linalg.solve(a=linear_system_coef, b=linear_system_ordinate)
        # linear_system_coef_sp = sp.Matrix(linear_system_coef)
        # linear_system_ordinate_sp = sp.Matrix(linear_system_ordinate)
        # solution = sp.solve_linear_system((linear_system_coef_sp, linear_system_ordinate_sp))
        # solution = np.linalg.lstsq(a=linear_system_coef, b=linear_system_ordinate)
        # stationary_dist = solution[0]
        # stationary_dist = solution / np.sum(solution)

        # eig vector of eig value 1

        eigen_values, eigen_vectors = np.linalg.eig(transition_matrix.values.T)
        eigen_vector_one = np.array(eigen_vectors[:, np.where(np.abs(eigen_values - 1.) < 1e-8)[0][0]].flat)
        stationary_dist = eigen_vector_one.real / np.sum(eigen_vector_one.real)
        # We compute the average cost
        average_cost = 0.
        for i, (state, arrivals) in enumerate(transition_matrix.state_space):
            costs = 0.
            if np.any(state.data + arrivals.data > model.capacity):
                # If we do, we set the arrivals to zero and induce a penalty
                arrivals = mm.State.zeros(matching_graph=model.matching_graph, capacity=model.capacity)
                costs += model.penalty
            # We add the arrivals
            total_state = state + arrivals
            # We compute the costs
            costs += np.dot(total_state.data, model.costs.data)
            average_cost += costs * stationary_dist[i]
        average_cost_array[threshold] = average_cost
    return np.argmin(average_cost_array)


def sec2str(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return '%02d:%02d:%02d' % (hours, minutes, int(seconds))
    elif minutes > 0:
        return '%02d:%02d' % (minutes, int(seconds))
    else:
        return '%0.2f' % seconds


def progprint_old(iterator, total=None, periteration=1, perline=25, show_times=True):
    # time.clock() is cpu time of current process
    # time.time() is wall time
    times = []
    idx = 0
    numdigits = None
    if total is not None:
        numdigits = len('%d' % total)
    for thing in iterator:
        prev_time = time.time()
        yield thing
        times.append(time.time() - prev_time)
        if (idx + 1) % periteration == 0:
            sys.stdout.write('.')
            if (idx+1) % perline == 0:
                if show_times:
                    avgtime = np.mean(times).item()
                    if total is not None:
                        eta = sec2str(avgtime * (total - (idx + 1)))
                        sys.stdout.write((
                            '  [ %%%dd/%%%dd, %%7.2fsec avg, ETA %%s ]\n' % (numdigits, numdigits))
                                         % (idx+1, total, avgtime, eta))
                    else:
                        sys.stdout.write('  [ %d done, %7.2fsec avg ]\n' % (idx+1, avgtime))
                else:
                    if total is not None:
                        sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits, numdigits)) % (idx+1, total))
                    else:
                        sys.stdout.write('  [ %d ]\n' % (idx+1))
        idx += 1
        sys.stdout.flush()
    print('')
    if show_times and len(times) > 0:
        total = sec2str(seconds=np.sum(times))
        print('%7.2fsec avg, %s total\n' % (np.mean(times).item(), total))


def progprint(iterator, total, periteration=1, perline=25, show_times=True):
    # time.clock() is cpu time of current process
    # time.time() is wall time
    assert total % periteration == 0
    times = []
    numdigits = len('%d' % total)
    for dot_idx in np.arange(int(total / periteration)):
        prev_time = time.time()
        for _ in np.arange(periteration):
            yield next(iterator)
        times.append(time.time() - prev_time)
        sys.stdout.write('.')
        if (dot_idx + 1) % perline == 0:
            idx = (dot_idx + 1) * periteration
            if show_times:
                avgtime = np.mean(times).item()
                eta = sec2str(avgtime * (total - idx) / periteration)
                sys.stdout.write((
                    '  [ %%%dd/%%%dd, %%7.2fsec avg, ETA %%s ]\n' % (numdigits, numdigits))
                                 % (idx, total, avgtime, eta))
            else:
                sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits, numdigits)) % (idx, total))
        sys.stdout.flush()
    print('')
    if show_times and len(times) > 0:
        total = sec2str(seconds=np.sum(times))
        print('%7.2fsec avg, %s total\n' % (np.mean(times).item(), total))
