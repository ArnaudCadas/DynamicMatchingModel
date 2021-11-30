import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time as time
import pickle
from typing import List, Tuple
from itertools import product

import ReinforcementLearning as rl
import utils as utils


class AdmissionPolicy:

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def admit(self, state, arrival_type):
        if arrival_type == 0:
            return True
        if state < self.thresholds[arrival_type - 1]:
            return True
        else:
            return False


class ExplorationPolicy:

    def __init__(self, buffer_size: int):
        self.state_counts = ValueFunction(buffer_size=buffer_size)
        self.initialise_policy()

    def admit(self, state, arrival_type):
        if arrival_type == 0:
            self.state_counts[state - 1] += 1
            return True
        less_visited_state = self.state_counts.state_space[np.argmin(self.state_counts.values)]
        if state < less_visited_state:
            self.state_counts[state + 1] += 1
            return True
        else:
            self.state_counts[state] += 1
            return False

    def initialise_policy(self):
        self.state_counts.initialise_values()


class RandomizedAdmissionPolicy:

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def admit(self, state, arrival_type):
        raise NotImplementedError

    def not_admitting_probability(self, state, arrival_type):
        raise NotImplementedError

    def not_admitting_grad(self, state, arrival_type):
        raise NotImplementedError


class ExpoPolicy(RandomizedAdmissionPolicy):

    def admit(self, state, arrival_type):
        if arrival_type == 0:
            return True
        if np.random.rand() < self.not_admitting_probability(state=state, arrival_type=arrival_type):
            return False
        else:
            return True

    def not_admitting_probability(self, state, arrival_type):
        return np.exp(state - self.thresholds[arrival_type - 1]) \
            / (1. + np.exp(state - self.thresholds[arrival_type - 1]))

    def not_admitting_grad(self, state, arrival_type):
        return - np.exp(state - self.thresholds[arrival_type - 1]) \
               / np.power((1. + np.exp(state - self.thresholds[arrival_type - 1])),
                          2.)


class ExpoSquaredPolicy(RandomizedAdmissionPolicy):

    def admit(self, state, arrival_type):
        if arrival_type == 0:
            return True
        if np.random.rand() < self.not_admitting_probability(state=state, arrival_type=arrival_type):
            return False
        else:
            return True

    def not_admitting_probability(self, state, arrival_type):
        return np.exp(np.power(state + 0.5, 2.) - np.power(self.thresholds[arrival_type - 1] + 0.01, 2.)) \
            / (1. + np.exp(np.power(state + 0.5, 2.) - np.power(self.thresholds[arrival_type - 1] + 0.01, 2.)))

    def not_admitting_grad(self, state, arrival_type):
        return (- 2. * (self.thresholds[arrival_type - 1] + 0.01)
                * np.exp(np.power(state + 0.5, 2.) - np.power(self.thresholds[arrival_type - 1] + 0.01, 2.))) \
               / np.power((1. + np.exp(np.power(state + 0.5, 2.)
                                       - np.power(self.thresholds[arrival_type - 1] + 0.01, 2.))), 2.)


class ExpoPowPolicy(RandomizedAdmissionPolicy):

    def __init__(self, thresholds, power=1.):
        super(ExpoPowPolicy, self).__init__(thresholds=thresholds)
        self.power = power

    def admit(self, state, arrival_type):
        if arrival_type == 0:
            return True
        if np.random.rand() < self.not_admitting_probability(state=state, arrival_type=arrival_type):
            return False
        else:
            return True

    def not_admitting_probability(self, state, arrival_type):
        return np.exp(np.power(state + 0.5, self.power)
                      - np.power(self.thresholds[arrival_type - 1] + 0.01, self.power)) \
            / (1. + np.exp(np.power(state + 0.5, self.power)
                           - np.power(self.thresholds[arrival_type - 1] + 0.01, self.power)))

    def not_admitting_grad(self, state, arrival_type):
        return (- self.power * (self.thresholds[arrival_type - 1] + 0.01)
                * np.exp(np.power(state + 0.5, self.power)
                         - np.power(self.thresholds[arrival_type - 1] + 0.01, self.power))) \
               / np.power((1. + np.exp(np.power(state + 0.5, self.power)
                                       - np.power(self.thresholds[arrival_type - 1] + 0.01, self.power))), 2.)


class AdmissionControlModel:

    def __init__(self, buffer_size, service_rate, arrival_rates, costs, policy, init_state):
        self.buffer_size = buffer_size
        self.service_rate = service_rate
        self.arrival_rates = arrival_rates
        self.nb_arrival_types = len(self.arrival_rates)
        assert len(costs) == self.nb_arrival_types
        self.costs = costs
        self.policy = policy
        self.init_state = init_state
        # We construct the event distribution which gives the probability of each event to occur.
        # 0 is a service and 1 to N is an arrival of that type
        distribution_unormalized = np.zeros(self.nb_arrival_types + 1)
        distribution_unormalized[0] = self.service_rate
        for i in np.arange(1, self.nb_arrival_types + 1):
            distribution_unormalized[i] = self.arrival_rates[i - 1]
        self.event_distribution = distribution_unormalized / np.sum(distribution_unormalized)
        self.empty_event_distribution = distribution_unormalized[1:] / np.sum(distribution_unormalized[1:])

    def iterate(self, state: int, policy=None):
        cost = 0.
        current_arrival_type = None
        # We sample the next event
        if state == 0:
            sample = stats.multinomial(n=1, p=self.empty_event_distribution).rvs()
            event = np.where(sample == 1)[1][0]
            event += 1
        else:
            sample = stats.multinomial(n=1, p=self.event_distribution).rvs()
            event = np.where(sample == 1)[1][0]
        # We get the action from the policy
        if policy is not None:
            action = policy.admit(state=state, arrival_type=event)
        else:
            action = self.policy.admit(state=state, arrival_type=event)
        if event == 0:
            # If it is a service, we remove one item in the queue
            next_state = state - 1
        else:
            # If it is an arrival, we compute if we admit it in the queue based on our policy
            if state == self.buffer_size:
                # If the system is full we do not admit the arrival
                next_state = state
                cost = self.costs[event - 1]
            else:
                if action:
                    next_state = state + 1
                else:
                    # If the arrival is not admitted, we have a cost based on the arrival type
                    next_state = state
                    cost = self.costs[event - 1]
            current_arrival_type = event
        return next_state, current_arrival_type, cost


class ValueFunction:

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        # We build the state space which is a list of all possible states
        self.state_space = np.arange(self.buffer_size + 1)
        # We initialise the value function to 0 for each state in the state space.
        self.values = np.zeros(self.buffer_size + 1)

    def initialise_values(self):
        for state in self.state_space:
            self.values[state] = 0.

    def __getitem__(self, state: int):
        if type(state) == int or type(state) == np.int64:
            return self.values[state]
        else:
            return NotImplemented

    def __setitem__(self, state: int, value):
        if type(state) == int or type(state) == np.int64:
            self.values[state] = value
        else:
            return NotImplemented

    def copy(self):
        new_value_function = ValueFunction(buffer_size=self.buffer_size)
        for state in self.state_space:
            new_value_function[state] = self[state]
        return new_value_function


class QFunction:

    def __init__(self, model: AdmissionControlModel):
        self.model = model
        # We build the state action space which is a list of all possible states with all possible actions
        self.state_action_space = [(state, action) for state, action in product(np.arange(self.model.buffer_size + 1),
                                                                                [0, 1])]
        # We initialise the value function to 0 for each state in the state space.
        self.values = {}
        self.initialise_values()

    def initialise_values(self):
        for state, action in self.state_action_space:
            self.values[state, action] = 0.

    def __getitem__(self, item: Tuple[int, int]):
        state, action = item
        if (type(state) == int or type(state) == np.int64) and (action == 0 or action == 1):
            return self.values[(state, action)]
        else:
            return NotImplemented

    def __setitem__(self, item: Tuple[int, int], value):
        state, action = item
        if (type(state) == int or type(state) == np.int64) and (action == 0 or action == 1):
            self.values[(state, action)] = value
        else:
            return NotImplemented

    def copy(self):
        new_value_function = QFunction(model=self.model)
        for state, action in self.state_action_space:
            new_value_function[state, action] = self[state, action]
        return new_value_function


class RelativeValueIteration:

    def __init__(self, model: AdmissionControlModel, policy=None):
        self.model = model
        self.V = ValueFunction(buffer_size=self.model.buffer_size)
        self.state_ast = 0
        self.policy = policy
        if self.policy is not None:
            if isinstance(self.policy, RandomizedAdmissionPolicy):
                self.bellman_operator = self.bellman_operator_with_randomized_policy
            if isinstance(self.policy, AdmissionPolicy):
                self.bellman_operator = self.bellman_operator_with_policy
        else:
            self.bellman_operator = self.bellman_operator_classic

    def bellman_operator_with_admissions(self, state: int, admissions: List[bool]) -> float:
        # We compute the Bellman operator for this state under the current threshold
        operator = 0.
        if state > 0:
            # We add the case when there is a service
            operator += self.model.event_distribution[0] * self.V[state - 1]
            # We add the cases when there is an arrival
            if state < self.model.buffer_size:
                for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                    if admissions[arrival_type - 1]:
                        # We add the case when we admit the arrival
                        operator += self.model.event_distribution[arrival_type] * self.V[state + 1]
                    else:
                        # We add the case when we do not admit the arrival
                        operator += self.model.event_distribution[arrival_type] \
                                            * (self.model.costs[arrival_type - 1] + self.V[state])
            else:
                for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                    # We add the case when we do not admit the arrival
                    operator += self.model.event_distribution[arrival_type] \
                        * (self.model.costs[arrival_type - 1] + self.V[state])
        else:
            # In this case, there is no service so the event probability change
            for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                if admissions[arrival_type - 1]:
                    # We add the case when we admit the arrival
                    operator += self.model.empty_event_distribution[arrival_type - 1] * self.V[state + 1]
                else:
                    # We add the case when we do not admit the arrival
                    operator += self.model.empty_event_distribution[arrival_type - 1] \
                                        * (self.model.costs[arrival_type - 1] + self.V[state])
        operator -= self.V[self.state_ast]
        return operator

    def bellman_operator_with_policy(self, state: int):
        admissions = [self.policy.admit(state=state, arrival_type=1), self.policy.admit(state=state, arrival_type=2)]
        return self.bellman_operator_with_admissions(state=state, admissions=admissions)

    def bellman_operator_with_randomized_policy(self, state: int):
        # We compute the Bellman operator for this state under the current threshold
        operator = 0.
        if state > 0:
            # We add the case when there is a service
            operator += self.model.event_distribution[0] * self.V[state - 1]
            # We add the cases when there is an arrival
            if state < self.model.buffer_size:
                for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                    not_admitting_proba = self.policy.not_admitting_probability(state=state, arrival_type=arrival_type)
                    # We add the case when we do not admit the arrival
                    operator += self.model.event_distribution[arrival_type] * not_admitting_proba \
                        * (self.model.costs[arrival_type - 1] + self.V[state])
                    # We add the case when we admit the arrival
                    operator += self.model.event_distribution[arrival_type] \
                        * (1. - not_admitting_proba) * self.V[state + 1]
            else:
                for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                    # We add the case when we do not admit the arrival
                    operator += self.model.event_distribution[arrival_type] \
                        * (self.model.costs[arrival_type - 1] + self.V[state])
        else:
            # In this case, there is no service so the event probability change
            for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                not_admitting_proba = self.policy.not_admitting_probability(state=state, arrival_type=arrival_type)
                # We add the case when we do not admit the arrival
                operator += self.model.empty_event_distribution[arrival_type - 1] \
                    * not_admitting_proba * (self.model.costs[arrival_type - 1] + self.V[state])
                # We add the case when we admit the arrival
                operator += self.model.empty_event_distribution[arrival_type - 1] \
                    * (1. - not_admitting_proba) * self.V[state + 1]
        operator -= self.V[self.state_ast]
        return operator

    def bellman_operator_classic(self, state: int):
        res_for_all_actions = [self.bellman_operator_with_admissions(state=state, admissions=[True, True]),
                               self.bellman_operator_with_admissions(state=state, admissions=[True, False]),
                               self.bellman_operator_with_admissions(state=state, admissions=[False, True]),
                               self.bellman_operator_with_admissions(state=state, admissions=[False, False])]
        return np.min(res_for_all_actions)

    def best_action(self, state: int):
        res_for_all_actions = [self.bellman_operator_with_admissions(state=state, admissions=[True, True]),
                               self.bellman_operator_with_admissions(state=state, admissions=[True, False]),
                               self.bellman_operator_with_admissions(state=state, admissions=[False, True]),
                               self.bellman_operator_with_admissions(state=state, admissions=[False, False])]
        return np.argmin(res_for_all_actions)

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

    def evaluate_value_function(self):
        fixed_point_eq_diff = np.zeros(len(self.V.state_space))
        for i, state in enumerate(self.V.state_space):
            # We compute the difference in the fixed point Bellman equation
            fixed_point_eq_diff[i] = self.V[state] - self.bellman_operator(state=state)
            return np.linalg.norm(fixed_point_eq_diff, ord=np.inf)

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


class Salmut:

    def __init__(self, model: AdmissionControlModel, fast_time_scale: rl.TimeScale, slow_time_scale: rl.TimeScale,
                 thresholds_init):
        self.model = model
        self.fast_time_scale = fast_time_scale
        self.slow_time_scale = slow_time_scale
        self.occurrence_count = ValueFunction(buffer_size=self.model.buffer_size)
        # We initialise the critic
        self.V = ValueFunction(buffer_size=self.model.buffer_size)
        self.state_ast = 0
        # We initialise the actor
        self.thresholds_init = thresholds_init
        self.thresholds = self.thresholds_init

    def iterate_critic(self, current_state: int, current_cost: int, next_state: int):
        current_occurence_count = self.occurrence_count[current_state]
        # We update the critic through Relative Value Iteration Algorithm
        self.V[current_state] = (1. - self.fast_time_scale[current_occurence_count]) \
            * self.V[current_state] + self.fast_time_scale[current_occurence_count] \
            * (current_cost + self.V[next_state] - self.V[self.state_ast])

    def iterate_actor(self, current_state: int, current_arrival_type: int, iteration: float):
        # We update the actor through gradient based method
        grad_smoother_action = - np.exp(current_state - self.thresholds[current_arrival_type - 1] - 0.5) \
            / np.power((1. + np.exp(current_state - self.thresholds[current_arrival_type - 1] - 0.5)), 2.)
        not_admitting_probability = np.random.rand()
        if np.random.rand() < not_admitting_probability:
            actor_state = current_state
            self.thresholds[current_arrival_type - 1] = self.thresholds[current_arrival_type - 1] \
                + self.slow_time_scale[iteration] * grad_smoother_action * self.V[actor_state]
        else:
            actor_state = current_state + 1
            self.thresholds[current_arrival_type - 1] = self.thresholds[current_arrival_type - 1] \
                - self.slow_time_scale[iteration] * grad_smoother_action * self.V[actor_state]
        # We make the projection
        for arrival_type in np.arange(current_arrival_type, self.model.nb_arrival_types + 1):
            if arrival_type == 1:
                self.thresholds[arrival_type - 1] = np.minimum(np.maximum(0., self.thresholds[arrival_type - 1]),
                                                               self.model.buffer_size)
            else:
                self.thresholds[arrival_type - 1] = np.minimum(np.maximum(self.thresholds[arrival_type - 2],
                                                                          self.thresholds[arrival_type - 1]),
                                                               self.model.buffer_size)
        self.model.policy.thresholds = self.thresholds

    def iterate(self, current_state: int, iteration):
        # We sample the next state and get the reward of the current state
        next_state, current_arrival_type, current_cost = self.model.iterate(state=current_state)
        # We update the critic
        self.iterate_critic(current_state=current_state, current_cost=current_cost, next_state=next_state)

        if current_arrival_type is not None and current_state < self.model.buffer_size:
            # We update the actor
            self.iterate_actor(current_state=current_state, current_arrival_type=current_arrival_type,
                               iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state] += 1.
        return next_state, current_cost

    def is_optimal(self, atol=0.05):
        pass

    def evaluate_critic(self):
        pass
        # fixed_point_diff = np.zeros(len(self.V.state_space))
        # for i, (state, arrivals) in enumerate(self.V.state_space):
        #     # We compute the Bellman operator for this state and arrivals under the current threshold
        #     bellman_operator = 0.
        #     if np.any(state.data + arrivals.data > self.model.capacity):
        #         new_state = state.copy()
        #         bellman_operator += self.model.penalty
        #     else:
        #         new_state = state + arrivals
        #     bellman_operator += np.dot(self.model.costs.data, new_state.data)
        #     matching = self.threshold_policy.match(state=new_state, arrivals=mm.State.zeros(
        #         matching_graph=self.model.matching_graph, capacity=self.model.capacity))
        #     for arrival_edge in self.V.complete_arrival_graph_edges_list:
        #         arrival_probability = np.prod(self.model.arrival_dist[arrival_edge])
        #         arrival = mm.State.zeros(self.model.matching_graph, self.model.capacity)
        #         arrival[arrival_edge] += 1.
        #         bellman_operator += self.V[new_state - matching, arrival] * arrival_probability
        #     bellman_operator -= self.V[self.state_ast, self.arrivals_ast]
        #     # We compute the difference in the fixed point Bellman equation
        #     fixed_point_diff[i] = self.V[state, arrivals] - bellman_operator
        #     return np.linalg.norm(fixed_point_diff)

    def run(self, nb_iterations=None, plot=False, verbose=False):
        self.occurrence_count.initialise_values()
        self.V.initialise_values()
        self.thresholds = self.thresholds_init
        self.model.policy.threshold = self.thresholds
        current_state = self.model.init_state
        if verbose:
            plot = True
        if plot:
            if nb_iterations is None:
                threshold_traj = []
                threshold_traj.append(self.thresholds)
                cost_traj = []
                iteration = 1.
                while not self.is_optimal():
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    threshold_traj.append(self.thresholds)
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
                threshold_traj = np.zeros((len(self.thresholds), nb_iterations + 1))
                threshold_traj[:, 0] = self.thresholds
                cost_traj = np.zeros(nb_iterations + 1)
                for iteration in iterator:
                    current_state, cost = self.iterate(current_state=current_state, iteration=iteration)
                    threshold_traj[:, iteration] = self.thresholds
                    cost_traj[iteration] = cost
                    if verbose:
                        critic_evaluation_traj[iteration] = self.evaluate_critic()
                        if np.round(self.thresholds) != np.round(threshold_traj[iteration - 1]):
                            critic_breakpoints.append(iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            for i in np.arange(len(self.thresholds)):
                axes[0].plot(threshold_traj[i, :], label="threshold {}".format(i + 1))
            axes[0].legend(loc='best')
            axes[0].set_title("Threshold trajectory")
            average_cost = np.cumsum(cost_traj) / np.arange(1, nb_iterations + 2)
            axes[1].plot(average_cost, label="average cost")
            axes[1].legend(loc='best')
            if verbose:
                for critic_breakpoint in critic_breakpoints:
                    axes[2].axvline(x=critic_breakpoint, color="red", linestyle=":", alpha=0.5)
                axes[2].plot(critic_evaluation_traj, label="fixed point diff")
                axes[2].legend(loc='best')
                axes[2].set_title("Critic evaluation")
            fig.canvas.draw()
        else:
            if nb_iterations is None:
                iteration = 1.
                while not self.is_optimal():
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    iteration += 1.
            else:
                for iteration in np.arange(1, nb_iterations + 1):
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
        return self.thresholds


class SalmutB:
    # We remove the sampling in the actor update and instead compute the diff between the two actions

    def __init__(self, model: AdmissionControlModel, fast_time_scale: rl.TimeScale, slow_time_scale: rl.TimeScale,
                 thresholds_init):
        self.model = model
        self.fast_time_scale = fast_time_scale
        self.slow_time_scale = slow_time_scale
        self.occurrence_count = ValueFunction(buffer_size=self.model.buffer_size)
        # We initialise the critic
        self.V = ValueFunction(buffer_size=self.model.buffer_size)
        self.state_ast = 0
        # We initialise the actor
        self.thresholds_init = thresholds_init
        self.thresholds = self.thresholds_init

    def iterate_critic(self, current_state: int, current_cost: int, next_state: int):
        current_occurence_count = self.occurrence_count[current_state]
        # We update the critic through Relative Value Iteration Algorithm
        self.V[current_state] = (1. - self.fast_time_scale[current_occurence_count]) \
            * self.V[current_state] + self.fast_time_scale[current_occurence_count] \
            * (current_cost + self.V[next_state] - self.V[self.state_ast])

    def iterate_actor(self, current_state: int, current_arrival_type: int, iteration: float):
        # We update the actor through gradient based method
        grad_smoother_action = - np.exp(current_state - self.thresholds[current_arrival_type - 1] - 0.5) \
            / np.power((1. + np.exp(current_state - self.thresholds[current_arrival_type - 1] - 0.5)), 2.)
        self.thresholds[current_arrival_type - 1] = self.thresholds[current_arrival_type - 1] \
            - self.slow_time_scale[iteration] * grad_smoother_action * (self.V[current_state]
                                                                        - self.V[current_state + 1])
        # We make the projection
        for arrival_type in np.arange(current_arrival_type, self.model.nb_arrival_types + 1):
            if arrival_type == 1:
                self.thresholds[arrival_type - 1] = np.minimum(np.maximum(0., self.thresholds[arrival_type - 1]),
                                                               self.model.buffer_size)
            else:
                self.thresholds[arrival_type - 1] = np.minimum(np.maximum(self.thresholds[arrival_type - 2],
                                                                          self.thresholds[arrival_type - 1]),
                                                               self.model.buffer_size)
        self.model.policy.thresholds = self.thresholds

    def iterate(self, current_state: int, iteration):
        # We sample the next state and get the reward of the current state
        next_state, current_arrival_type, current_cost = self.model.iterate(state=current_state)
        # We update the critic
        self.iterate_critic(current_state=current_state, current_cost=current_cost, next_state=next_state)

        if current_arrival_type is not None and current_state < self.model.buffer_size:
            # We update the actor
            self.iterate_actor(current_state=current_state, current_arrival_type=current_arrival_type,
                               iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state] += 1.
        return next_state, current_cost

    def is_optimal(self, atol=0.05):
        pass

    def evaluate_critic(self):
        fixed_point_diff = np.zeros(len(self.V.state_space))
        rvi = RelativeValueIteration(model=self.model, policy=self.model.policy)
        for i, state in enumerate(self.V.state_space):
            operator = rvi.bellman_operator(state=state)
            # We compute the difference in the fixed point Bellman equation
            fixed_point_diff[i] = self.V[state] - operator
        return np.linalg.norm(fixed_point_diff, ord=np.inf)

    def run(self, nb_iterations=None, plot=False, verbose=False):
        self.occurrence_count.initialise_values()
        self.V.initialise_values()
        self.thresholds = self.thresholds_init
        self.model.policy.threshold = self.thresholds
        current_state = self.model.init_state
        if verbose:
            plot = True
        if plot:
            if nb_iterations is None:
                threshold_traj = []
                threshold_traj.append(self.thresholds)
                cost_traj = []
                iteration = 1.
                while not self.is_optimal():
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    threshold_traj.append(self.thresholds)
                    iteration += 1.
                threshold_traj = np.array(threshold_traj)
            else:
                iterator = np.arange(1, nb_iterations + 1)
                if verbose:
                    critic_evaluation_traj = np.zeros(nb_iterations + 1)
                    critic_evaluation_traj[0] = self.evaluate_critic()
                    # critic_breakpoints = []
                    iterator = utils.progprint(iterator=iter(iterator), total=nb_iterations, perline=25,
                                               periteration=100, show_times=True)
                threshold_traj = np.zeros((len(self.thresholds), nb_iterations + 1))
                threshold_traj[:, 0] = self.thresholds
                cost_traj = np.zeros(nb_iterations + 1)
                for iteration in iterator:
                    current_state, cost = self.iterate(current_state=current_state, iteration=iteration)
                    threshold_traj[:, iteration] = self.thresholds
                    cost_traj[iteration] = cost
                    if verbose:
                        critic_evaluation_traj[iteration] = self.evaluate_critic()
                        # if np.round(self.thresholds) != np.round(threshold_traj[iteration - 1]):
                        #     critic_breakpoints.append(iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            for i in np.arange(len(self.thresholds)):
                axes[0].plot(threshold_traj[i, :], label="threshold {}".format(i + 1))
            axes[0].legend(loc='best')
            axes[0].set_title("Threshold trajectory")
            average_cost = np.cumsum(cost_traj) / np.arange(1, nb_iterations + 2)
            axes[1].plot(average_cost, label="average cost")
            axes[1].legend(loc='best')
            if verbose:
                # for critic_breakpoint in critic_breakpoints:
                #     axes[2].axvline(x=critic_breakpoint, color="red", linestyle=":", alpha=0.5)
                axes[2].plot(critic_evaluation_traj, label="fixed point diff")
                axes[2].legend(loc='best')
                axes[2].set_title("Critic evaluation")
            fig.canvas.draw()
        else:
            if nb_iterations is None:
                iteration = 1.
                while not self.is_optimal():
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    iteration += 1.
            else:
                for iteration in np.arange(1, nb_iterations + 1):
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
        return self.thresholds


class SalmutBWithRVICritic(SalmutB):
    # We replace the critic with relative value iteration

    def __init__(self, model, fast_time_scale, slow_time_scale, thresholds_init):
        super(SalmutBWithRVICritic, self).__init__(model=model, fast_time_scale=fast_time_scale,
                                                   slow_time_scale=slow_time_scale, thresholds_init=thresholds_init)
        self.rvi = RelativeValueIteration(model=self.model, policy=self.model.policy)

    def iterate_critic(self, current_state: int):
        current_occurence_count = self.occurrence_count[current_state]
        operator = self.rvi.bellman_operator(state=current_state)
        # We update the critic through Relative Value Iteration Algorithm
        self.V[current_state] = (1. - self.fast_time_scale[current_occurence_count]) * self.V[current_state] \
            + self.fast_time_scale[current_occurence_count] * operator

    def iterate(self, current_state: int, iteration):
        # We sample the next state and get the reward of the current state
        next_state, current_arrival_type, current_cost = self.model.iterate(state=current_state)
        # We update the critic
        self.iterate_critic(current_state=current_state)

        if current_arrival_type is not None and current_state < self.model.buffer_size:
            # We update the actor
            self.iterate_actor(current_state=current_state, current_arrival_type=current_arrival_type,
                               iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state] += 1.
        return next_state, current_cost


class SalmutC:
    # We change the actor to perform a policy gradient based on Sutton formula

    def __init__(self, model: AdmissionControlModel, fast_time_scale: rl.TimeScale, slow_time_scale: rl.TimeScale,
                 threshold_policy: RandomizedAdmissionPolicy):
        self.model = model
        self.fast_time_scale = fast_time_scale
        self.slow_time_scale = slow_time_scale
        self.occurrence_count = ValueFunction(buffer_size=self.model.buffer_size)
        # We initialise the critic
        self.V = ValueFunction(buffer_size=self.model.buffer_size)
        self.state_ast = 0
        # We initialise the actor
        self.threshold_policy = threshold_policy
        self.thresholds_init = self.threshold_policy.thresholds

    def iterate_critic(self, current_state: int):
        current_occurence_count = self.occurrence_count[current_state]
        # We sample the next state and compute the current cost
        next_state, current_arrival_type, current_cost = self.model.iterate(state=current_state,
                                                                            policy=self.threshold_policy)
        # We update the critic through Relative Value Iteration Algorithm
        self.V[current_state] = (1. - self.fast_time_scale[current_occurence_count]) \
            * self.V[current_state] + self.fast_time_scale[current_occurence_count] \
            * (current_cost + self.V[next_state] - self.V[self.state_ast])

    def iterate_actor(self, current_state: int, current_arrival_type: int, iteration: float):
        # We update the actor through gradient based method
        grad_smoother_action = self.threshold_policy.not_admitting_grad(state=current_state,
                                                                        arrival_type=current_arrival_type)
        self.threshold_policy.thresholds[current_arrival_type - 1] -= self.slow_time_scale[iteration] \
            * grad_smoother_action * (self.model.costs[current_arrival_type - 1] + self.V[current_state]
                                      - self.V[current_state + 1])
        # We make the projection
        for arrival_type in np.arange(current_arrival_type, self.model.nb_arrival_types + 1):
            if arrival_type == 1:
                self.threshold_policy.thresholds[arrival_type - 1] = np.minimum(
                    np.maximum(0., self.threshold_policy.thresholds[arrival_type - 1]), self.model.buffer_size)
            else:
                self.threshold_policy.thresholds[arrival_type - 1] = np.minimum(
                    np.maximum(self.threshold_policy.thresholds[arrival_type - 2],
                               self.threshold_policy.thresholds[arrival_type - 1]), self.model.buffer_size)

    def iterate(self, current_state: int, iteration):
        # We sample the next state and get the reward of the current state
        next_state, current_arrival_type, current_cost = self.model.iterate(state=current_state)
        # We update the critic
        self.iterate_critic(current_state=current_state)

        if current_arrival_type is not None and current_state < self.model.buffer_size:
            # We update the actor
            self.iterate_actor(current_state=current_state, current_arrival_type=current_arrival_type,
                               iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state] += 1.
        return next_state, current_cost

    def is_optimal(self, atol=0.05):
        pass

    def evaluate_critic(self):
        fixed_point_diff = np.zeros(len(self.V.state_space))
        for i, state in enumerate(self.V.state_space):
            # We compute the Bellman operator for this state under the current threshold
            operator = 0.
            if state > 0:
                # We add the case when there is a service
                operator += self.model.event_distribution[0] * self.V[state - 1]
                # We add the cases when there is an arrival
                if state < self.model.buffer_size:
                    for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                        not_admitting_proba = self.threshold_policy.not_admitting_probability(state=state,
                                                                                              arrival_type=arrival_type)
                        # We add the case when we do not admit the arrival
                        operator += self.model.event_distribution[arrival_type] * not_admitting_proba \
                            * (self.model.costs[arrival_type - 1] + self.V[state])
                        # We add the case when we admit the arrival
                        operator += self.model.event_distribution[arrival_type] \
                            * (1. - not_admitting_proba) * self.V[state + 1]
                else:
                    for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                        # We add the case when we do not admit the arrival
                        operator += self.model.event_distribution[arrival_type] \
                                    * (self.model.costs[arrival_type - 1] + self.V[state])
            else:
                # In this case, there is no service so the event probability change
                for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                    not_admitting_proba = self.threshold_policy.not_admitting_probability(state=state,
                                                                                          arrival_type=arrival_type)
                    # We add the case when we do not admit the arrival
                    operator += self.model.empty_event_distribution[arrival_type - 1] \
                        * not_admitting_proba * (self.model.costs[arrival_type - 1] + self.V[state])
                    # We add the case when we admit the arrival
                    operator += self.model.empty_event_distribution[arrival_type - 1] \
                        * (1. - not_admitting_proba) * self.V[state + 1]
            operator -= self.V[self.state_ast]
            # We compute the difference in the fixed point Bellman equation
            fixed_point_diff[i] = self.V[state] - operator
        return np.linalg.norm(fixed_point_diff, ord=np.inf)

    def run(self, nb_iterations=None, plot=False, verbose=False):
        self.occurrence_count.initialise_values()
        self.V.initialise_values()
        self.threshold_policy.thresholds = self.thresholds_init
        self.model.policy.initialise_policy()
        current_state = self.model.init_state
        if verbose:
            plot = True
        if plot:
            if nb_iterations is None:
                threshold_traj = []
                threshold_traj.append(self.threshold_policy.thresholds)
                cost_traj = []
                iteration = 1.
                while not self.is_optimal():
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    threshold_traj.append(self.threshold_policy.thresholds)
                    iteration += 1.
                threshold_traj = np.array(threshold_traj)
            else:
                iterator = np.arange(1, nb_iterations + 1)
                if verbose:
                    critic_evaluation_traj = np.zeros(nb_iterations + 1)
                    critic_evaluation_traj[0] = self.evaluate_critic()
                    # critic_breakpoints = []
                    iterator = utils.progprint(iterator=iter(iterator), total=nb_iterations, perline=25,
                                               periteration=100, show_times=True)
                threshold_traj = np.zeros((len(self.threshold_policy.thresholds), nb_iterations + 1))
                threshold_traj[:, 0] = self.threshold_policy.thresholds
                cost_traj = np.zeros(nb_iterations + 1)
                for iteration in iterator:
                    current_state, cost = self.iterate(current_state=current_state, iteration=iteration)
                    threshold_traj[:, iteration] = self.threshold_policy.thresholds
                    cost_traj[iteration] = cost
                    if verbose:
                        critic_evaluation_traj[iteration] = self.evaluate_critic()
                        # if np.round(self.thresholds) != np.round(threshold_traj[iteration - 1]):
                        #     critic_breakpoints.append(iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            for i in np.arange(len(self.threshold_policy.thresholds)):
                axes[0].plot(threshold_traj[i, :], label="threshold {}".format(i + 1))
            axes[0].legend(loc='best')
            axes[0].set_title("Threshold trajectory")
            average_cost = np.cumsum(cost_traj) / np.arange(1, nb_iterations + 2)
            axes[1].plot(average_cost, label="average cost")
            axes[1].legend(loc='best')
            if verbose:
                # for critic_breakpoint in critic_breakpoints:
                #     axes[2].axvline(x=critic_breakpoint, color="red", linestyle=":", alpha=0.5)
                axes[2].plot(critic_evaluation_traj, label="fixed point diff")
                axes[2].legend(loc='best')
                axes[2].set_title("Critic evaluation")
            fig.canvas.draw()
        else:
            if nb_iterations is None:
                iteration = 1.
                while not self.is_optimal():
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    iteration += 1.
            else:
                for iteration in np.arange(1, nb_iterations + 1):
                    current_state = self.iterate(current_state=current_state, iteration=iteration)
                    # print("iteration: {}, threshold: {}".format(iteration, self.threshold_policy.threshold))
        return self.threshold_policy.thresholds, threshold_traj


class SalmutCWithRVICritic(SalmutC):
    # We replace the critic with relative value iteration

    def iterate_critic(self, current_state: int):
        current_occurence_count = self.occurrence_count[current_state]
        # We compute the Bellman operator for this state under the current threshold
        operator = 0.
        if current_state > 0:
            # We add the case when there is a service
            operator += self.model.event_distribution[0] * self.V[current_state - 1]
            # We add the cases when there is an arrival
            if current_state < self.model.buffer_size:
                for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                    not_admitting_proba = self.threshold_policy.not_admitting_probability(state=current_state,
                                                                                          arrival_type=arrival_type)
                    # We add the case when we do not admit the arrival
                    operator += self.model.event_distribution[arrival_type] * not_admitting_proba \
                        * (self.model.costs[arrival_type - 1] + self.V[current_state])
                    # We add the case when we admit the arrival
                    operator += self.model.event_distribution[arrival_type] \
                        * (1. - not_admitting_proba) * self.V[current_state + 1]
            else:
                for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                    # We add the case when we do not admit the arrival
                    operator += self.model.event_distribution[arrival_type] \
                                * (self.model.costs[arrival_type - 1] + self.V[current_state])
        else:
            # In this case, there is no service so the event probability change
            for arrival_type in np.arange(1, self.model.nb_arrival_types + 1):
                not_admitting_proba = self.threshold_policy.not_admitting_probability(state=current_state,
                                                                                      arrival_type=arrival_type)
                # We add the case when we do not admit the arrival
                operator += self.model.empty_event_distribution[arrival_type - 1] \
                    * not_admitting_proba * (self.model.costs[arrival_type - 1] + self.V[current_state])
                # We add the case when we admit the arrival
                operator += self.model.empty_event_distribution[arrival_type - 1] \
                    * (1. - not_admitting_proba) * self.V[current_state + 1]
        operator -= self.V[self.state_ast]
        # We update the critic through Relative Value Iteration Algorithm
        self.V[current_state] = (1. - self.fast_time_scale[current_occurence_count]) * self.V[current_state] \
            + self.fast_time_scale[current_occurence_count] * operator

    def iterate(self, current_state: int, iteration):
        # We sample the next state and get the reward of the current state
        next_state, current_arrival_type, current_cost = self.model.iterate(state=current_state)
        # We update the critic
        self.iterate_critic(current_state=current_state)

        if current_arrival_type is not None and current_state < self.model.buffer_size:
            # We update the actor
            self.iterate_actor(current_state=current_state, current_arrival_type=current_arrival_type,
                               iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state] += 1.
        return next_state, current_cost


class SalmutCWithoutOccurences(SalmutC):
    # We remove the occurrence count

    def iterate_critic(self, current_state: int, iteration):
        # We sample the next state and compute the current cost
        next_state, current_arrival_type, current_cost = self.model.iterate(state=current_state,
                                                                            policy=self.threshold_policy)
        # We update the critic through Relative Value Iteration Algorithm
        self.V[current_state] = (1. - self.fast_time_scale[iteration]) \
                                * self.V[current_state] + self.fast_time_scale[iteration] \
                                * (current_cost + self.V[next_state] - self.V[self.state_ast])

    def iterate(self, current_state: int, iteration):
        # We sample the next state and get the reward of the current state
        next_state, current_arrival_type, current_cost = self.model.iterate(state=current_state)
        # We update the critic
        self.iterate_critic(current_state=current_state, iteration=iteration)

        if current_arrival_type is not None and current_state < self.model.buffer_size:
            # We update the actor
            self.iterate_actor(current_state=current_state, current_arrival_type=current_arrival_type,
                               iteration=iteration)
        # We update the occurrence count
        self.occurrence_count[next_state] += 1.
        return next_state, current_cost


class ModelA:

    def __init__(self):
        buffer_size = 5
        service_rate = 2.
        arrival_rate = 6.
        arrival_type_probabilities = np.array([0.8, 0.2])
        arrival_rates = np.multiply(arrival_rate, arrival_type_probabilities)
        costs = np.array([10., 20.])
        init_state = 0
        exploration_policy = ExplorationPolicy(buffer_size=buffer_size)

        self.model = AdmissionControlModel(buffer_size=buffer_size, service_rate=service_rate,
                                           arrival_rates=arrival_rates, costs=costs, policy=exploration_policy,
                                           init_state=init_state)
        # Optimal thresholds: [3., 5.]


class ModelB:

    def __init__(self):
        buffer_size = 5
        service_rate = 2.
        arrival_rate = 6.
        arrival_type_probabilities = np.array([0.8, 0.2])
        arrival_rates = np.multiply(arrival_rate, arrival_type_probabilities)
        costs = np.array([19., 20.])
        init_state = 0
        exploration_policy = ExplorationPolicy(buffer_size=buffer_size)

        self.model = AdmissionControlModel(buffer_size=buffer_size, service_rate=service_rate,
                                           arrival_rates=arrival_rates, costs=costs, policy=exploration_policy,
                                           init_state=init_state)
        # Optimal thresholds: [4., 5.]


class ModelC:

    def __init__(self):
        buffer_size = 5
        service_rate = 2.
        arrival_rate = 6.
        arrival_type_probabilities = np.array([0.8, 0.2])
        arrival_rates = np.multiply(arrival_rate, arrival_type_probabilities)
        costs = np.array([5., 20.])
        init_state = 0
        exploration_policy = ExplorationPolicy(buffer_size=buffer_size)

        self.model = AdmissionControlModel(buffer_size=buffer_size, service_rate=service_rate,
                                           arrival_rates=arrival_rates, costs=costs, policy=exploration_policy,
                                           init_state=init_state)
        # Optimal thresholds: [2., 5.]


class ModelD:

    def __init__(self):
        buffer_size = 5
        service_rate = 2.
        arrival_rate = 6.
        arrival_type_probabilities = np.array([0.8, 0.2])
        arrival_rates = np.multiply(arrival_rate, arrival_type_probabilities)
        costs = np.array([1., 20.])
        init_state = 0
        exploration_policy = ExplorationPolicy(buffer_size=buffer_size)

        self.model = AdmissionControlModel(buffer_size=buffer_size, service_rate=service_rate,
                                           arrival_rates=arrival_rates, costs=costs, policy=exploration_policy,
                                           init_state=init_state)
        # Optimal thresholds: [1., 5.]


def demo_salmut():
    np.random.seed(42)
    model = ModelB().model
    thresholds_init = np.array([0., 0.])
    threshold_policy = ExpoPowPolicy(thresholds=thresholds_init, power=2.)

    # fast_time_scale = rl.BorkarFastTimeScale(power=0.51, scale=100., shift=2.)
    # fast_time_scale = rl.ClassicTimeScale(power=0.8, shift=1.)
    # slow_time_scale = rl.ClassicTimeScale(power=0.9, scalar=10.)
    fast_time_scale = rl.ClassicTimeScale(power=0.9, shift=1.)
    slow_time_scale = rl.ClassicTimeScale(power=0.8, scalar=10.)
    # algo = SalmutCWithRVICritic(model=model, fast_time_scale=fast_time_scale, slow_time_scale=slow_time_scale,
    #                             threshold_policy=threshold_policy)
    algo = SalmutC(model=model, fast_time_scale=fast_time_scale, slow_time_scale=slow_time_scale,
                   threshold_policy=threshold_policy)

    nb_iterations = 1000000
    t = time.time()
    final_thresholds, threshold_traj = algo.run(nb_iterations=nb_iterations, plot=True, verbose=True)
    print("Run time: {}".format(time.time() - t))
    print("final thresholds: {}".format(final_thresholds))
    np.savetxt("Demo_Expo_Experiment_ModelB_Sampling_CorrectStepSize_1M.csv", threshold_traj,
               delimiter=",")

    plt.show()


def demo_rvi(nb_value_iterations=100):
    np.random.seed(42)
    service_rate = 2.
    arrival_rate = 6.
    arrival_type_probabilities = np.array([0.8, 0.2])
    arrival_rates = np.multiply(arrival_rate, arrival_type_probabilities)
    costs = np.array([1., 20.])
    thresholds_init = np.array([0., 5.])
    policy = ExpoPolicy(thresholds=thresholds_init)
    model = AdmissionControlModel(buffer_size=5, service_rate=service_rate, arrival_rates=arrival_rates,
                                  costs=costs, policy=policy, init_state=0)
    # rvi = RelativeValueIteration(model=model, policy=policy)
    rvi = RelativeValueIteration(model=model, policy=None)

    print("Start of relative value iteration...")
    t = time.time()
    rvi.run(nb_iterations=nb_value_iterations, plot=True)
    print("End of relative value iteration, runtime: {}".format(time.time() - t))
    print("Is the solution optimal ? {}".format(rvi.is_optimal()))
    # for debug
    policy = np.empty((6, 2), dtype=bool)
    actions = np.array([[True, True], [True, False], [False, True], [False, False]], dtype=bool)
    for state in np.arange(5):
        policy[state, :] = actions[rvi.best_action(state=state), :]
    policy[5, :] = [False, False]
    print(policy)


if __name__ == "__main__":
    demo_salmut()
    # demo_rvi()

