import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
import pickle
from typing import Tuple, List

import MatchingModel as mm
import Policies as po
import utils as utils


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

    def __init__(self, model: mm.Model, policy=None):
        self.model = model
        self.V = ValueFunction(model=self.model)
        self.policy = policy
        if self.policy is not None:
            if isinstance(self.policy, po.RandomizedPolicy):
                self.iterate = self.iterate_with_randomized_policy
            else:
                self.iterate = self.iterate_with_policy
        else:
            self.iterate = self.iterate_without_policy

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

    def bellman_operator_with_randomized_matching(self, state: mm.State, arrivals: mm.State):
        res = 0.
        if np.any(state.data + arrivals.data > self.model.capacity):
            new_state = state.copy()
            res += self.model.penalty
        else:
            new_state = state + arrivals
        res += np.dot(self.model.costs.data, new_state.data)
        for matching, matching_probability in self.policy.distribution(state=state, arrivals=arrivals):
            for arrival_edge in self.V.complete_arrival_graph_edges_list:
                arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
                arrival = mm.State.zeros(self.model.matching_graph, self.model.capacity)
                arrival[arrival_edge] += 1.
                res += self.model.discount * self.V[new_state - matching, arrival] * arrival_probability \
                    * matching_probability
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

    def iterate_without_policy(self):
        next_V = self.V.copy()
        for state, arrivals in self.V.state_space:
            next_V[state, arrivals] = self.bellman_operator(state=state, arrivals=arrivals)
        self.V = next_V.copy()

    def iterate_with_policy(self):
        next_V = self.V.copy()
        for state, arrivals in self.V.state_space:
            matching = self.policy.match(state=state, arrivals=arrivals)
            next_V[state, arrivals] = self.bellman_operator_with_matching(state=state, arrivals=arrivals,
                                                                          matching=matching)
        self.V = next_V.copy()

    def iterate_with_randomized_policy(self):
        next_V = self.V.copy()
        for state, arrivals in self.V.state_space:
            next_V[state, arrivals] = self.bellman_operator_with_randomized_matching(state=state, arrivals=arrivals)
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


class RelativeValueIteration:

    def __init__(self, model: mm.Model, policy=None):
        self.model = model
        self.V = ValueFunction(model=self.model)
        self.state_ast, self.arrivals_ast = self.V.state_space[0]
        self.policy = policy
        if self.policy is not None:
            if isinstance(self.policy, po.RandomizedPolicy):
                self.bellman_operator = self.bellman_operator_with_randomized_policy
            else:
                self.bellman_operator = self.bellman_operator_with_policy
        else:
            self.bellman_operator = self.bellman_operator_classic

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
            res += self.V[new_state - matching, arrival] * arrival_probability
        res -= self.V[self.state_ast, self.arrivals_ast]
        return res

    def bellman_operator_with_policy(self, state: mm.State, arrivals: mm.State):
        matching = self.policy.match(state=state, arrivals=arrivals)
        return self.bellman_operator_with_matching(state=state, arrivals=arrivals, matching=matching)

    def bellman_operator_with_randomized_policy(self, state: mm.State, arrivals: mm.State):
        res = 0.
        if np.any(state.data + arrivals.data > self.model.capacity):
            arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
            res += self.model.penalty
        total_state = state + arrivals
        res += np.dot(self.model.costs.data, total_state.data)
        for matching, matching_probability in self.policy.distribution(state=state, arrivals=arrivals):
            for arrival_edge in self.V.complete_arrival_graph_edges_list:
                arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
                arrival = mm.State.zeros(self.model.matching_graph, self.model.capacity)
                arrival[arrival_edge] += 1.
                res += self.V[total_state - matching, arrival] * arrival_probability * matching_probability
        res -= self.V[self.state_ast, self.arrivals_ast]
        return res

    def bellman_operator_classic(self, state: mm.State, arrivals: mm.State):
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

    def evaluate_value_function(self):
        fixed_point_eq_diff = np.zeros(len(self.V.state_space))
        for i, (state, arrivals) in enumerate(self.V.state_space):
            # We compute the difference in the fixed point Bellman equation
            fixed_point_eq_diff[i] = self.V[state, arrivals] - self.bellman_operator(state=state, arrivals=arrivals)
            return np.linalg.norm(fixed_point_eq_diff)

    def run(self, nb_iterations=None, save_file=None, plot=False):
        self.V.initialise_values()
        if plot:
            if nb_iterations is None:
                value_function_perf = []
                while not self.is_optimal():
                    self.iterate()
                    value_function_perf.append(self.evaluate_value_function())
            else:
                value_function_perf = np.zeros(nb_iterations)
                for i in np.arange(nb_iterations):
                    self.iterate()
                    value_function_perf[i] = self.evaluate_value_function()
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.plot(value_function_perf, label="fixed point diff")
            ax.legend(loc='best')
            ax.set_title("Value function evaluation")
            fig.canvas.draw()
            plt.show()
        else:
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

    def __init__(self, power=1., scalar=1., shift=0.):
        self.power = power
        self.scalar = scalar
        self.shift = shift

    def __getitem__(self, n):
        return self.scalar / np.power(float(n) + self.shift, self.power)


class BorkarFastTimeScale(TimeScale):

    def __init__(self, power=1., shift=1., scale=1.):
        self.power = power
        self.shift = shift
        self.scale = scale

    def __getitem__(self, n):
        return 1. / np.power(np.floor(float(n) / self.scale) + self.shift, self.power)


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

    def evaluate_critic(self):
        fixed_point_diff = np.zeros(len(self.V.state_space))
        for i, (state, arrivals) in enumerate(self.V.state_space):
            # We compute the Bellman operator for this state and arrivals under the current threshold
            bellman_operator = 0.
            if np.any(state.data + arrivals.data > self.model.capacity):
                new_state = state.copy()
                bellman_operator += self.model.penalty
            else:
                new_state = state + arrivals
            bellman_operator += np.dot(self.model.costs.data, new_state.data)
            matching = self.threshold_policy.match(state=new_state, arrivals=mm.State.zeros(
                matching_graph=self.model.matching_graph, capacity=self.model.capacity))
            for arrival_edge in self.V.complete_arrival_graph_edges_list:
                arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
                arrival = mm.State.zeros(self.model.matching_graph, self.model.capacity)
                arrival[arrival_edge] += 1.
                bellman_operator += self.V[new_state - matching, arrival] * arrival_probability
            bellman_operator -= self.V[self.state_ast, self.arrivals_ast]
            # We compute the difference in the fixed point Bellman equation
            fixed_point_diff[i] = self.V[state, arrivals] - bellman_operator
            return np.linalg.norm(fixed_point_diff)

    def run(self, nb_iterations=None, plot=False, verbose=False):
        self.occurrence_count.initialise_values()
        self.V.initialise_values()
        self.threshold = 0
        self.threshold_policy.threshold = self.threshold
        current_state = self.model.init_state.copy()
        current_arrivals = self.model.init_arrival.copy()
        if verbose:
            plot = True
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
                if verbose:
                    critic_evaluation_traj = np.zeros(nb_iterations + 1)
                    critic_evaluation_traj[0] = self.evaluate_critic()
                    critic_breakpoints = []
                threshold_traj = np.zeros(nb_iterations + 1)
                threshold_traj[0] = self.threshold
                for iteration in np.arange(1, nb_iterations + 1):
                    current_state, current_arrivals = self.iterate(current_state=current_state,
                                                                   current_arrivals=current_arrivals, iteration=iteration)
                    threshold_traj[iteration] = self.threshold
                    if verbose:
                        critic_evaluation_traj[iteration] = self.evaluate_critic()
                        if np.round(self.threshold) != np.round(threshold_traj[iteration - 1]):
                            critic_breakpoints.append(iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            axes[0].plot(threshold_traj, label="threshold")
            axes[0].legend(loc='best')
            axes[0].set_title("Threshold trajectory")
            if verbose:
                for critic_breakpoint in critic_breakpoints:
                    axes[1].axvline(x=critic_breakpoint, color="red", linestyle=":", alpha=0.5)
                axes[1].plot(critic_evaluation_traj, label="fixed point diff")
                axes[1].legend(loc='best')
                axes[1].set_title("Critic evaluation")
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


class SalmutB:
    # We change the threshold policy used in critic iteration to choose whether to match or not with probability f(s,t)
    # instead of rounding the threshold. We also use a better definition of P_{s,s^\prime}(t).
    # We remove the (-1)^\alpha

    def __init__(self, model: mm.Model, fast_time_scale: TimeScale, slow_time_scale: TimeScale):
        self.model = model
        self.fast_time_scale = fast_time_scale
        self.slow_time_scale = slow_time_scale
        self.occurrence_count = ValueFunction(model=self.model)
        # We initialise the critic
        self.V = ValueFunction(model=self.model)
        self.state_ast, self.arrivals_ast = self.V.state_space[0]
        # We initialise the actor
        self.threshold_init = 0.01
        self.threshold = self.threshold_init
        self.threshold_policy = po.Threshold_N_continuous(state_space="state_and_arrival", threshold=self.threshold)

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

    def iterate_actor(self, current_state: mm.State, current_arrivals: mm.State, iteration: float):
        # We compute the current total state
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        current_total_state = current_state + current_arrivals
        # We sample the action chosen (match the threshold edge or not) and compute the new state
        actor_matching = mm.Matching.zeros(current_total_state)
        actor_matching[1, 1] += current_total_state[1, 1].min()
        actor_matching[2, 2] += current_total_state[2, 2].min()
        if np.ceil(self.threshold) == np.floor(self.threshold):
            matching_threshold = self.threshold
            grad_smoother_action = 0.
        else:
            ceil_threshold_prob = np.random.rand()
            if np.random.rand() <= ceil_threshold_prob:
                matching_threshold = np.ceil(self.threshold)
                grad_smoother_action = 1.
            else:
                matching_threshold = np.floor(self.threshold)
                grad_smoother_action = -1.
        remaining_items = current_total_state - actor_matching
        l3_matchings = np.maximum(remaining_items[1, 2].min() - matching_threshold, 0.)
        actor_matching[1, 2] += l3_matchings
        # We compute the new state and arrivals
        actor_state = current_total_state - actor_matching
        actor_arrival = self.model.sample_arrivals()
        # We update the actor through gradient based method
        new_threshold = self.threshold - self.slow_time_scale[iteration] \
            * grad_smoother_action * self.V[actor_state, actor_arrival]
        self.threshold = np.minimum(np.maximum(0.01, new_threshold), self.model.capacity - 0.01)
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

    def evaluate_critic(self):
        fixed_point_diff = np.zeros(len(self.V.state_space))
        for i, (state, arrivals) in enumerate(self.V.state_space):
            # We compute the Bellman operator for this state and arrivals under the current threshold
            bellman_operator = 0.
            if np.any(state.data + arrivals.data > self.model.capacity):
                new_state = state.copy()
                bellman_operator += self.model.penalty
            else:
                new_state = state + arrivals
            bellman_operator += np.dot(self.model.costs.data, new_state.data)
            matching = self.threshold_policy.match(state=new_state, arrivals=mm.State.zeros(
                matching_graph=self.model.matching_graph, capacity=self.model.capacity))
            for arrival_edge in self.V.complete_arrival_graph_edges_list:
                arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
                arrival = mm.State.zeros(self.model.matching_graph, self.model.capacity)
                arrival[arrival_edge] += 1.
                bellman_operator += self.V[new_state - matching, arrival] * arrival_probability
            bellman_operator -= self.V[self.state_ast, self.arrivals_ast]
            # We compute the difference in the fixed point Bellman equation
            fixed_point_diff[i] = self.V[state, arrivals] - bellman_operator
            return np.linalg.norm(fixed_point_diff)

    def run(self, nb_iterations=None, plot=False, verbose=False):
        self.occurrence_count.initialise_values()
        self.V.initialise_values()
        self.threshold = self.threshold_init
        self.threshold_policy.threshold = self.threshold
        current_state = self.model.init_state.copy()
        current_arrivals = self.model.init_arrival.copy()
        if verbose:
            plot = True
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
                if verbose:
                    critic_evaluation_traj = np.zeros(nb_iterations + 1)
                    critic_evaluation_traj[0] = self.evaluate_critic()
                    critic_breakpoints = []
                threshold_traj = np.zeros(nb_iterations + 1)
                threshold_traj[0] = self.threshold
                for iteration in np.arange(1, nb_iterations + 1):
                    current_state, current_arrivals = self.iterate(current_state=current_state,
                                                                   current_arrivals=current_arrivals, iteration=iteration)
                    threshold_traj[iteration] = self.threshold
                    if verbose:
                        critic_evaluation_traj[iteration] = self.evaluate_critic()
                        if np.round(self.threshold) != np.round(threshold_traj[iteration - 1]):
                            critic_breakpoints.append(iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            axes[0].plot(threshold_traj, label="threshold")
            axes[0].legend(loc='best')
            axes[0].set_title("Threshold trajectory")
            if verbose:
                for critic_breakpoint in critic_breakpoints:
                    axes[1].axvline(x=critic_breakpoint, color="red", linestyle=":", alpha=0.5)
                axes[1].plot(critic_evaluation_traj, label="fixed point diff")
                axes[1].legend(loc='best')
                axes[1].set_title("Critic evaluation")
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


class SalmutBWithoutActor(SalmutB):

    # def __init__(self, model: mm.Model, fast_time_scale: TimeScale, slow_time_scale: TimeScale):
    #     super(SalmutBWithoutActor, self).__init__(model=model, fast_time_scale=fast_time_scale,
    #                                               slow_time_scale=slow_time_scale)
    #     self.threshold_init = 1
    #     self.threshold = 1
    #     self.threshold_policy = po.Threshold_N(state_space="state_and_arrival", threshold=self.threshold)

    def iterate(self, current_state: mm.State, current_arrivals: mm.State, iteration):
        # We update the critic
        next_state, next_arrivals = self.iterate_critic(current_state=current_state, current_arrivals=current_arrivals)
        # We update the actor
        # self.iterate_actor(current_state=current_state, current_arrivals=current_arrivals, iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state, next_arrivals] += 1.
        return next_state, next_arrivals


class SalmutBA(SalmutB):
    # Compared to SalmutB, we sample the next state in actor iteration according to the threshold policy instead of
    # uniformly

    def iterate_actor(self, current_state: mm.State, current_arrivals: mm.State, iteration: float):
        # We compute the current total state
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        current_total_state = current_state + current_arrivals
        # We sample the action chosen (match the threshold edge or not) and compute the new state
        actor_matching = mm.Matching.zeros(current_total_state)
        actor_matching[1, 1] += current_total_state[1, 1].min()
        actor_matching[2, 2] += current_total_state[2, 2].min()
        if np.ceil(self.threshold) == np.floor(self.threshold):
            matching_threshold = self.threshold
            grad_smoother_action = 0.
        else:
            if np.random.rand() <= self.threshold_policy._threshold_probability:
                matching_threshold = np.ceil(self.threshold)
                grad_smoother_action = 1.
            else:
                matching_threshold = np.floor(self.threshold)
                grad_smoother_action = -1.
        remaining_items = current_total_state - actor_matching
        l3_matchings = np.maximum(remaining_items[1, 2].min() - matching_threshold, 0.)
        actor_matching[1, 2] += l3_matchings
        # We compute the new state and arrivals
        actor_state = current_total_state - actor_matching
        actor_arrival = self.model.sample_arrivals()
        # We update the actor through gradient based method
        new_threshold = self.threshold - self.slow_time_scale[iteration] \
            * grad_smoother_action * self.V[actor_state, actor_arrival]
        self.threshold = np.minimum(np.maximum(0.01, new_threshold), self.model.capacity - 0.01)
        self.threshold_policy.threshold = self.threshold


class SalmutBB(SalmutB):
    # Compared to SalmutB, we dont sample the next state in actor iteration and instead compute exactly the gradient,
    # i.e the diff between the two value functions

    def iterate_actor(self, current_state: mm.State, current_arrivals: mm.State, iteration: float):
        # We compute the current total state
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        current_total_state = current_state + current_arrivals
        # We sample the action chosen (match the threshold edge or not) and compute the new state
        actor_matching = mm.Matching.zeros(current_total_state)
        actor_matching[1, 1] += current_total_state[1, 1].min()
        actor_matching[2, 2] += current_total_state[2, 2].min()

        remaining_items = current_total_state - actor_matching
        actor_matching_ceil = actor_matching.copy()
        actor_matching_ceil[1, 2] += np.maximum(remaining_items[1, 2].min() - np.ceil(self.threshold), 0.)
        actor_matching_floor = actor_matching.copy()
        actor_matching_floor[1, 2] += np.maximum(remaining_items[1, 2].min() - np.floor(self.threshold), 0.)
        # We compute the new state and arrivals
        actor_state_ceil = current_total_state - actor_matching_ceil
        actor_state_floor = current_total_state - actor_matching_floor
        actor_arrival = self.model.sample_arrivals()
        # We update the actor through gradient based method
        new_threshold = self.threshold - self.slow_time_scale[iteration] \
            * (self.V[actor_state_ceil, actor_arrival] - self.V[actor_state_floor, actor_arrival])
        self.threshold = np.minimum(np.maximum(0.01, new_threshold), self.model.capacity - 0.01)
        self.threshold_policy.threshold = self.threshold


class SalmutC:
    # We change the threshold policy used in critic iteration to choose whether to match or not with probability f(s,t)
    # instead of rounding the threshold. We also use a better definition of P_{s,s^\prime}(t).
    # We remove the (-1)^\alpha

    def __init__(self, model: mm.Model, fast_time_scale: TimeScale, slow_time_scale: TimeScale):
        self.model = model
        self.fast_time_scale = fast_time_scale
        self.slow_time_scale = slow_time_scale
        self.occurrence_count = ValueFunction(model=self.model)
        # We initialise the critic
        self.V = ValueFunction(model=self.model)
        self.state_ast, self.arrivals_ast = self.V.state_space[0]
        # We initialise the actor
        self.threshold_init = 0.
        self.threshold = self.threshold_init
        self.threshold_policy = po.Threshold_N_norm_dist(state_space="state_and_arrival", threshold=self.threshold)

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

    def iterate_actor(self, current_state: mm.State, current_arrivals: mm.State, iteration: float):
        # We compute the current total state
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        current_total_state = current_state + current_arrivals
        # We sample the action chosen (match the threshold edge or not) and compute the new state
        actor_matching = mm.Matching.zeros(current_total_state)
        actor_matching[1, 1] += current_total_state[1, 1].min()
        actor_matching[2, 2] += current_total_state[2, 2].min()
        remaining_items = current_total_state - actor_matching
        nb_l3_items = np.min(remaining_items[1, 2])
        matching_threshold = np.random.randint(nb_l3_items + 1)
        sigma = 3. / (2. * np.square(2. * np.log(2.)))
        grad_smoother_action = stats.norm.pdf(self.threshold, loc=matching_threshold, scale=sigma) \
            * (matching_threshold - self.threshold) / np.power(sigma, 2.)
        actor_matching[1, 2] += np.maximum(nb_l3_items - matching_threshold, 0.)
        # We compute the new state and arrivals
        actor_state = current_total_state - actor_matching
        actor_arrival = self.model.sample_arrivals()
        # We update the actor through gradient based method
        new_threshold = self.threshold - self.slow_time_scale[iteration] \
            * grad_smoother_action * self.V[actor_state, actor_arrival]
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

    def evaluate_critic(self):
        fixed_point_diff = np.zeros(len(self.V.state_space))
        for i, (state, arrivals) in enumerate(self.V.state_space):
            # We compute the Bellman operator for this state and arrivals under the current threshold
            bellman_operator = 0.
            if np.any(state.data + arrivals.data > self.model.capacity):
                new_state = state.copy()
                bellman_operator += self.model.penalty
            else:
                new_state = state + arrivals
            bellman_operator += np.dot(self.model.costs.data, new_state.data)
            matching = self.threshold_policy.match(state=new_state, arrivals=mm.State.zeros(
                matching_graph=self.model.matching_graph, capacity=self.model.capacity))
            for arrival_edge in self.V.complete_arrival_graph_edges_list:
                arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
                arrival = mm.State.zeros(self.model.matching_graph, self.model.capacity)
                arrival[arrival_edge] += 1.
                bellman_operator += self.V[new_state - matching, arrival] * arrival_probability
            bellman_operator -= self.V[self.state_ast, self.arrivals_ast]
            # We compute the difference in the fixed point Bellman equation
            fixed_point_diff[i] = self.V[state, arrivals] - bellman_operator
            return np.linalg.norm(fixed_point_diff)

    def run(self, nb_iterations=None, plot=False, verbose=False):
        self.occurrence_count.initialise_values()
        self.V.initialise_values()
        self.threshold = self.threshold_init
        self.threshold_policy.threshold = self.threshold
        current_state = self.model.init_state.copy()
        current_arrivals = self.model.init_arrival.copy()
        if verbose:
            plot = True
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
                if verbose:
                    critic_evaluation_traj = np.zeros(nb_iterations + 1)
                    critic_evaluation_traj[0] = self.evaluate_critic()
                    critic_breakpoints = []
                threshold_traj = np.zeros(nb_iterations + 1)
                threshold_traj[0] = self.threshold
                for iteration in np.arange(1, nb_iterations + 1):
                    current_state, current_arrivals = self.iterate(current_state=current_state,
                                                                   current_arrivals=current_arrivals, iteration=iteration)
                    threshold_traj[iteration] = self.threshold
                    if verbose:
                        critic_evaluation_traj[iteration] = self.evaluate_critic()
                        if np.round(self.threshold) != np.round(threshold_traj[iteration - 1]):
                            critic_breakpoints.append(iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            axes[0].plot(threshold_traj, label="threshold")
            axes[0].legend(loc='best')
            axes[0].set_title("Threshold trajectory")
            if verbose:
                for critic_breakpoint in critic_breakpoints:
                    axes[1].axvline(x=critic_breakpoint, color="red", linestyle=":", alpha=0.5)
                axes[1].plot(critic_evaluation_traj, label="fixed point diff")
                axes[1].legend(loc='best')
                axes[1].set_title("Critic evaluation")
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


class SalmutCB(SalmutC):
    # Compared to SalmutC, we dont sample the next state in actor iteration and instead compute exactly the gradient,
    # i.e the diff between the various value functions

    def compute_grad_action_smoother(self, matching_threshold, max_threshold):
        sigma = 3. / (2. * np.square(2. * np.log(2.)))
        numerator = stats.norm.pdf(self.threshold, loc=matching_threshold, scale=sigma) \
            * np.sum([stats.norm.pdf(self.threshold, loc=threshold, scale=sigma)
                      * (matching_threshold - threshold) / np.power(sigma, 2.)
                      for threshold in np.arange(max_threshold + 1)])
        denominator = np.power(np.sum([stats.norm.pdf(self.threshold, loc=threshold, scale=sigma)
                                       for threshold in np.arange(max_threshold + 1)]), 2.)
        return numerator / denominator

    def iterate_actor(self, current_state: mm.State, current_arrivals: mm.State, iteration: float):
        # We compute the current total state
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        current_total_state = current_state + current_arrivals
        # We sample the action chosen (match the threshold edge or not) and compute the new state
        actor_matching = mm.Matching.zeros(current_total_state)
        actor_matching[1, 1] += current_total_state[1, 1].min()
        actor_matching[2, 2] += current_total_state[2, 2].min()
        remaining_items = current_total_state - actor_matching
        nb_l3_items = np.min(remaining_items[1, 2])
        actor_state_list = []
        grad_action_smoother_list = []
        for matching_threshold in np.arange(nb_l3_items + 1):
            grad_action_smoother_list.append(self.compute_grad_action_smoother(matching_threshold=matching_threshold,
                                                                               max_threshold=nb_l3_items))
            final_matching = actor_matching.copy()
            final_matching[1, 2] += np.maximum(nb_l3_items - matching_threshold, 0.)
            # We compute the new state
            actor_state_list.append(current_total_state - final_matching)
        actor_arrival = self.model.sample_arrivals()
        # We update the actor through gradient based method
        new_threshold = self.threshold - self.slow_time_scale[iteration] \
            * np.sum([grad_action_smoother * self.V[actor_state, actor_arrival]
                      for (actor_state, grad_action_smoother) in zip(actor_state_list, grad_action_smoother_list)])
        self.threshold = np.minimum(np.maximum(0.01, new_threshold), self.model.capacity - 0.01)
        self.threshold_policy.threshold = self.threshold


class SalmutD:
    # We change the threshold policy used in critic iteration to choose whether to match or not with probability f(s,t)
    # instead of rounding the threshold. We also use a better definition of P_{s,s^\prime}(t).
    # We remove the (-1)^\alpha

    def __init__(self, model: mm.Model, fast_time_scale: TimeScale, slow_time_scale: TimeScale):
        self.model = model
        self.fast_time_scale = fast_time_scale
        self.slow_time_scale = slow_time_scale
        self.occurrence_count = ValueFunction(model=self.model)
        # We initialise the critic
        self.V = ValueFunction(model=self.model)
        self.state_ast, self.arrivals_ast = self.V.state_space[0]
        # We initialise the actor
        self.threshold_init = 0.
        self.threshold = self.threshold_init
        self.threshold_policy = po.Threshold_N_norm_dist_all(state_space="state_and_arrival", threshold=self.threshold)

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

    def iterate_actor(self, current_state: mm.State, current_arrivals: mm.State, iteration: float):
        # We compute the current total state
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        current_total_state = current_state + current_arrivals
        # We sample the action chosen (match the threshold edge or not) and compute the new state
        actor_matching = mm.Matching.zeros(current_total_state)
        actor_matching[1, 1] += current_total_state[1, 1].min()
        actor_matching[2, 2] += current_total_state[2, 2].min()
        remaining_items = current_total_state - actor_matching
        nb_l3_items = np.min(remaining_items[1, 2])
        matching_threshold = np.random.randint(nb_l3_items + 1)
        sigma = 3. / (2. * np.square(2. * np.log(2.)))
        grad_smoother_action = stats.norm.pdf(self.threshold, loc=matching_threshold, scale=sigma) \
            * (matching_threshold - self.threshold) / np.power(sigma, 2.)
        actor_matching[1, 2] += np.maximum(nb_l3_items - matching_threshold, 0.)
        # We compute the new state and arrivals
        actor_state = current_total_state - actor_matching
        actor_arrival = self.model.sample_arrivals()
        # We update the actor through gradient based method
        new_threshold = self.threshold - self.slow_time_scale[iteration] \
            * grad_smoother_action * self.V[actor_state, actor_arrival]
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

    def evaluate_critic(self):
        fixed_point_diff = np.zeros(len(self.V.state_space))
        for i, (state, arrivals) in enumerate(self.V.state_space):
            # We compute the Bellman operator for this state and arrivals under the current threshold
            bellman_operator = 0.
            if np.any(state.data + arrivals.data > self.model.capacity):
                new_state = state.copy()
                bellman_operator += self.model.penalty
            else:
                new_state = state + arrivals
            bellman_operator += np.dot(self.model.costs.data, new_state.data)
            matching = self.threshold_policy.match(state=new_state, arrivals=mm.State.zeros(
                matching_graph=self.model.matching_graph, capacity=self.model.capacity))
            for arrival_edge in self.V.complete_arrival_graph_edges_list:
                arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
                arrival = mm.State.zeros(self.model.matching_graph, self.model.capacity)
                arrival[arrival_edge] += 1.
                bellman_operator += self.V[new_state - matching, arrival] * arrival_probability
            bellman_operator -= self.V[self.state_ast, self.arrivals_ast]
            # We compute the difference in the fixed point Bellman equation
            fixed_point_diff[i] = self.V[state, arrivals] - bellman_operator
            return np.linalg.norm(fixed_point_diff)

    def run(self, nb_iterations=None, plot=False, verbose=False):
        self.occurrence_count.initialise_values()
        self.V.initialise_values()
        self.threshold = self.threshold_init
        self.threshold_policy.threshold = self.threshold
        current_state = self.model.init_state.copy()
        current_arrivals = self.model.init_arrival.copy()
        if verbose:
            plot = True
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
                iterator = np.arange(1, nb_iterations + 1)
                if verbose:
                    critic_evaluation_traj = np.zeros(nb_iterations + 1)
                    critic_evaluation_traj[0] = self.evaluate_critic()
                    critic_breakpoints = []
                    iterator = utils.progprint(iterator=iter(iterator), total=nb_iterations, perline=25,
                                               periteration=100, show_times=True)
                threshold_traj = np.zeros(nb_iterations + 1)
                threshold_traj[0] = self.threshold
                for iteration in iterator:
                    current_state, current_arrivals = self.iterate(current_state=current_state,
                                                                   current_arrivals=current_arrivals, iteration=iteration)
                    threshold_traj[iteration] = self.threshold
                    if verbose:
                        critic_evaluation_traj[iteration] = self.evaluate_critic()
                        if np.round(self.threshold) != np.round(threshold_traj[iteration - 1]):
                            critic_breakpoints.append(iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            axes[0].plot(threshold_traj, label="threshold")
            axes[0].legend(loc='best')
            axes[0].set_title("Threshold trajectory")
            if verbose:
                for critic_breakpoint in critic_breakpoints:
                    axes[1].axvline(x=critic_breakpoint, color="red", linestyle=":", alpha=0.5)
                axes[1].plot(critic_evaluation_traj, label="fixed point diff")
                axes[1].legend(loc='best')
                axes[1].set_title("Critic evaluation")
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


class SalmutDB(SalmutD):
    # Compared to SalmutD, we dont sample the next state in actor iteration and instead compute exactly the gradient,
    # i.e the diff between the various value functions

    def compute_grad_action_smoother(self, matching_threshold):
        sigma = 3. / (2. * np.square(2. * np.log(2.)))
        numerator = stats.norm.pdf(self.threshold, loc=matching_threshold, scale=sigma) \
            * np.sum([stats.norm.pdf(self.threshold, loc=threshold, scale=sigma)
                      * (matching_threshold - threshold) / np.power(sigma, 2.)
                      for threshold in np.arange(self.model.capacity + 1)])
        denominator = np.power(np.sum([stats.norm.pdf(self.threshold, loc=threshold, scale=sigma)
                                       for threshold in np.arange(self.model.capacity + 1)]), 2.)
        return numerator / denominator

    def iterate_actor(self, current_state: mm.State, current_arrivals: mm.State, iteration: float):
        # We compute the current total state
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        current_total_state = current_state + current_arrivals
        # We sample the action chosen (match the threshold edge or not) and compute the new state
        actor_matching = mm.Matching.zeros(current_total_state)
        actor_matching[1, 1] += current_total_state[1, 1].min()
        actor_matching[2, 2] += current_total_state[2, 2].min()
        remaining_items = current_total_state - actor_matching
        nb_l3_items = np.min(remaining_items[1, 2])
        actor_state_list = []
        grad_action_smoother_list = []
        for matching_threshold in np.arange(self.model.capacity + 1):
            grad_action_smoother_list.append(self.compute_grad_action_smoother(matching_threshold=matching_threshold))
            final_matching = actor_matching.copy()
            final_matching[1, 2] += np.maximum(nb_l3_items - matching_threshold, 0.)
            # We compute the new state
            actor_state_list.append(current_total_state - final_matching)
        actor_arrival = self.model.sample_arrivals()
        # We update the actor through gradient based method
        new_threshold = self.threshold - self.slow_time_scale[iteration] \
            * np.sum([grad_action_smoother * self.V[actor_state, actor_arrival]
                      for (actor_state, grad_action_smoother) in zip(actor_state_list, grad_action_smoother_list)])
        self.threshold = np.minimum(np.maximum(0.01, new_threshold), self.model.capacity - 0.01)
        self.threshold_policy.threshold = self.threshold


class SalmutDBWithoutActor(SalmutDB):

    def __init__(self, model: mm.Model, fast_time_scale: TimeScale, slow_time_scale: TimeScale):
        super(SalmutDBWithoutActor, self).__init__(model=model, fast_time_scale=fast_time_scale,
                                                   slow_time_scale=slow_time_scale)
        self.threshold_init = 3.
        self.threshold = 3.
        self.threshold_policy = po.Threshold_N_norm_dist_all(state_space="state_and_arrival", threshold=self.threshold)
        self.critic_exploration_policy = self.threshold_policy
        self.state_ast = mm.State(values=np.array([2., 0., 0., 2.]), matching_graph=self.model.matching_graph,
                                  capacity=self.model.capacity)
        self.arrivals_ast = mm.State(values=np.array([1., 0., 0., 1.]), matching_graph=self.model.matching_graph,
                                     capacity=self.model.capacity)

    def iterate_critic(self, current_state: mm.State, current_arrivals: mm.State, iteration):
        # We sample the next state and arrivals and get the costs of the current state and arrivals
        # next_state_list, next_arrivals, costs_list = self.model.iterate_state_with_arrival(
        #     states_list=[current_state.copy()], arrivals=current_arrivals.copy(), policies=[self.threshold_policy])
        # next_state = next_state_list[0]
        # current_costs = costs_list[0]
        # current_occurence_count = self.occurrence_count[current_state, current_arrivals]
        # # We update the critic through Relative Value Iteration Algorithm
        # self.V[current_state, current_arrivals] = (1. - self.fast_time_scale[current_occurence_count]) \
        #     * self.V[current_state, current_arrivals] + self.fast_time_scale[current_occurence_count] \
        #     * (current_costs + self.V[next_state, next_arrivals] - self.V[self.state_ast, self.arrivals_ast])
        # self.V[current_state, current_arrivals] = (1. - self.fast_time_scale[iteration]) \
        #     * self.V[current_state, current_arrivals] + self.fast_time_scale[iteration] \
        #     * (current_costs + self.V[next_state, next_arrivals] - self.V[self.state_ast, self.arrivals_ast])
        # We compute the current total state
        current_costs = 0.
        if np.any(current_state.data + current_arrivals.data > self.model.capacity):
            current_costs += self.model.penalty
            new_arrivals = mm.State.zeros(matching_graph=self.model.matching_graph, capacity=self.model.capacity)
        else:
            new_arrivals = current_arrivals.copy()
        current_total_state = current_state + new_arrivals
        current_costs = np.dot(current_total_state.data, self.model.costs.data)
        critic_state_list = []
        critic_state_probability_list = []
        for matching, matching_probability in self.threshold_policy.distribution(state=current_state,
                                                                                 arrivals=new_arrivals):
            critic_state_list.append(current_total_state - matching)
            critic_state_probability_list.append(matching_probability)
        critic_arrivals = self.model.sample_arrivals()
        current_occurence_count = self.occurrence_count[current_state, current_arrivals]
        # # We update the critic through Relative Value Iteration Algorithm
        self.V[current_state, current_arrivals] = (1. - self.fast_time_scale[current_occurence_count]) \
            * self.V[current_state, current_arrivals] + self.fast_time_scale[current_occurence_count] \
            * (current_costs + np.sum([self.V[critic_state, critic_arrivals] * matching_probability
                                       for critic_state, matching_probability in zip(critic_state_list,
                                                                                     critic_state_probability_list)])
               - self.V[self.state_ast, self.arrivals_ast])
        # We sample next state according to a exploratory distribution
        exploration_matching = self.critic_exploration_policy.match(state=current_state, arrivals=new_arrivals)
        next_state = current_total_state - exploration_matching
        return next_state, critic_arrivals

    def iterate(self, current_state: mm.State, current_arrivals: mm.State, iteration):
        # We update the critic
        # For debug
        actor_l3_state_list = [mm.State(values=np.array([float(k), 0., 0., float(k)]),
                                        matching_graph=self.model.matching_graph, capacity=self.model.capacity)
                               for k in np.arange(self.model.capacity + 1)]
        actor_l4_state_list = [mm.State(values=np.array([0., float(k), float(k), 0.]),
                                        matching_graph=self.model.matching_graph, capacity=self.model.capacity)
                               for k in np.arange(self.model.capacity + 1)]
        actor_state_list = actor_l3_state_list + actor_l4_state_list[1:]
        arrival_list = [mm.State(values=np.array([1., 0., 1., 0.]), matching_graph=self.model.matching_graph,
                                 capacity=self.model.capacity),
                        mm.State(values=np.array([0., 1., 0., 1.]), matching_graph=self.model.matching_graph,
                                 capacity=self.model.capacity),
                        mm.State(values=np.array([1., 0., 0., 1.]), matching_graph=self.model.matching_graph,
                                 capacity=self.model.capacity),
                        mm.State(values=np.array([0., 1., 1., 0.]), matching_graph=self.model.matching_graph,
                                 capacity=self.model.capacity)]
        value_function_test = [self.V[state, arrival_list[0]] for state in actor_l3_state_list]
        occurence_counts_test = np.array([self.occurrence_count[state, arrival]
                                 for state, arrival in itertools.product(actor_state_list, arrival_list)])
        next_state, next_arrivals = self.iterate_critic(current_state=current_state, current_arrivals=current_arrivals,
                                                        iteration=iteration)
        # We update the actor
        # self.iterate_actor(current_state=current_state, current_arrivals=current_arrivals, iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state, next_arrivals] += 1.
        return next_state, next_arrivals
