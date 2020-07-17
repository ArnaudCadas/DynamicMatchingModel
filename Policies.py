import numpy as np
import scipy.stats as stats
import pickle
import mipcl_py.mipshell.mipshell as mip
import MatchingModel as mm
import ReinforcementLearning as rl


# TODO: Change the inheritance of all policies to the respective Policy class based on state space
# We define a class for policies
class Policy:

    def __init__(self, state_space: str):
        if state_space == "state":
            self.match = self.compute_matchings_state
        elif state_space == "state_and_arrival":
            self.match = self.compute_matchings_state_and_arrival
        else:
            raise ValueError("State space should be equal to 'state' or 'state_and_arrival'.")

    def compute_matchings_state(self, state: mm.State):
        pass

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        pass

    def reset_policy(self, x_0):
        pass

    def __str__(self):
        pass


class PolicyOnState(Policy):

    def compute_matchings_state(self, state: mm.State):
        raise NotImplementedError


class PolicyOnStateAndArrivals(Policy):

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        raise NotImplementedError


class RandomizedPolicy(Policy):

    def __init__(self, state_space: str):
        super(RandomizedPolicy, self).__init__(state_space=state_space)
        if state_space == "state":
            self.distribution = self.compute_distribution_state
        elif state_space == "state_and_arrival":
            self.distribution = self.compute_distribution_state_and_arrival
        else:
            raise ValueError("State space should be equal to 'state' or 'state_and_arrival'.")

    def compute_distribution_state(self, state: mm.State):
        pass

    def compute_distribution_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        pass


class RandomizedPolicyOnState(RandomizedPolicy):

    def compute_matchings_state(self, state: mm.State):
        raise NotImplementedError

    def compute_distribution_state(self, state: mm.State):
        raise NotImplementedError


class RandomizedPolicyOnStateAndArrivals(RandomizedPolicy):

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        raise NotImplementedError

    def compute_distribution_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        raise NotImplementedError


# We define various policies by creating child class from Policy and implementing the function match()

class NoMatchings(PolicyOnState, PolicyOnStateAndArrivals):

    def compute_matchings_state(self, state: mm.State):
        return mm.Matching.zeros(state=state)

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_matchings_state(state=new_state)

    def __str__(self):
        return "No matchings policy"


class ValueIterationOptimal(PolicyOnStateAndArrivals):

    def __init__(self, state_space: str, model: mm.Model, nb_iterations=None, load_file=None):
        assert state_space == "state_and_arrival"
        super(ValueIterationOptimal, self).__init__(state_space=state_space)
        self.model = model
        if load_file is None:
            self.value_iteration = rl.ValueIteration(model=self.model)
            print("The value iteration algorithm starts. Beware, it can be very long.")
            self.value_iteration.run(nb_iterations=nb_iterations)
        else:
            with open(load_file, 'rb') as pickle_file:
                self.value_iteration = pickle.load(pickle_file)
                assert self.value_iteration.model == self.model

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        if np.any(state.data + arrivals.data > self.model.capacity):
            new_state = state.copy()
        else:
            new_state = state + arrivals
        matchings_available = new_state.complete_matchings_available()
        res_for_all_matchings = np.zeros(len(matchings_available))
        for i, matching in enumerate(matchings_available):
            res_for_all_matchings[i] = self.value_iteration.bellman_operator_with_matching(
                state=state, arrivals=arrivals, matching=matching)
        return matchings_available[np.argmin(res_for_all_matchings)]

    def __str__(self):
        return "Optimal policy from Value Iteration"


class RelativeValueIterationOptimal(PolicyOnStateAndArrivals):

    def __init__(self, state_space: str, model: mm.Model, nb_iterations=None, load_file=None):
        assert state_space == "state_and_arrival"
        super(RelativeValueIterationOptimal, self).__init__(state_space=state_space)
        self.model = model
        if load_file is None:
            self.relative_value_iteration = rl.RelativeValueIteration(model=self.model)
            print("The value iteration algorithm starts. Beware, it can be very long.")
            self.relative_value_iteration.run(nb_iterations=nb_iterations)
        else:
            with open(load_file, 'rb') as pickle_file:
                self.relative_value_iteration = pickle.load(pickle_file)
                assert self.relative_value_iteration.model == self.model

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        if np.any(state.data + arrivals.data > self.model.capacity):
            new_state = state.copy()
        else:
            new_state = state + arrivals
        matchings_available = new_state.complete_matchings_available()
        res_for_all_matchings = np.zeros(len(matchings_available))
        for i, matching in enumerate(matchings_available):
            res_for_all_matchings[i] = self.relative_value_iteration.bellman_operator_with_matching(
                state=state, arrivals=arrivals, matching=matching)
        return matchings_available[np.argmin(res_for_all_matchings)]

    def __str__(self):
        return "Optimal policy from Relative Value Iteration"


# We define a random policy which choose a random possible (depending on the State) matching.
# The policy has a parameter that gives the maximum number of times we repeat the last operation.
class Random_policy(Policy):

    def __init__(self, nb_matchings_max=np.inf):
        assert nb_matchings_max == np.inf or (type(nb_matchings_max) == int and nb_matchings_max >= 1)
        self.nb_matchings_max = nb_matchings_max

    def match(self, x):
        nb_matchings = 0
        u = mm.Matching.zeros(x)
        possible_matchings = x.matchings_available()
        while nb_matchings < self.nb_matchings_max and possible_matchings:
            edge = possible_matchings[np.random.randint(len(possible_matchings))]
            u[edge] += 1.
            nb_matchings += 1
            new_state = x - u
            possible_matchings = new_state.matchings_available()
        return u

    def __str__(self):
        return 'Random policy m={}'.format(self.nb_matchings_max)


class Priority_N(PolicyOnState, PolicyOnStateAndArrivals):

    def compute_matchings_state(self, state: mm.State):
        u = mm.Matching.zeros(state)
        # We match all l1
        u[1, 1] += state[1, 1].min()
        # We match all l2
        u[2, 2] += state[2, 2].min()
        return u

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_matchings_state(state=new_state)

    def __str__(self):
        return 'Priority N policy'


class Threshold_N(PolicyOnState, PolicyOnStateAndArrivals):

    def __init__(self, state_space: str, threshold: int):
        super(Threshold_N, self).__init__(state_space=state_space)
        self.threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = np.round(threshold)

    def compute_matchings_state(self, state: mm.State):
        u = mm.Matching.zeros(state)
        # We match all l1
        u[1, 1] += state[1, 1].min()
        # We match all l2
        u[2, 2] += state[2, 2].min()
        # We update the state with the matchings in l1 and l2 because they have priority and they influence the ones in
        # l3
        new_state = state - u
        # We match all l3 above the threshold
        l3_matchings = np.maximum(new_state[1, 2].min() - self.threshold, 0.)
        u[1, 2] += l3_matchings
        return u

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_matchings_state(state=new_state)

    def __str__(self):
        return 'Threshold N policy t={}'.format(self.threshold)


class Threshold_N_continuous(RandomizedPolicyOnState, RandomizedPolicyOnStateAndArrivals):

    def __init__(self, state_space: str, threshold: float):
        super(Threshold_N_continuous, self).__init__(state_space=state_space)
        self.threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold
        self._threshold_floor = np.floor(self._threshold)
        self._threshold_ceil = np.ceil(self._threshold)
        self._threshold_is_int = (self._threshold_floor == self._threshold_ceil)
        self._threshold_probability = self._threshold - self._threshold_floor

    def compute_matchings_state(self, state: mm.State):
        u = mm.Matching.zeros(state)
        # We match all l1
        u[1, 1] += state[1, 1].min()
        # We match all l2
        u[2, 2] += state[2, 2].min()
        # We test if we match in l3 or not
        if self._threshold_is_int:
            matching_threshold = self._threshold
        else:
            if np.random.rand() <= self._threshold_probability:
                matching_threshold = self._threshold_ceil
            else:
                matching_threshold = self._threshold_floor
        # We update the state with the matchings in l1 and l2 because they have priority and they influence the ones in
        # l3
        new_state = state - u
        # We match all l3 above the threshold
        l3_matchings = np.maximum(new_state[1, 2].min() - matching_threshold, 0.)
        u[1, 2] += l3_matchings
        return u

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_matchings_state(state=new_state)

    def compute_distribution_state(self, state: mm.State):
        dist = []
        u = mm.Matching.zeros(state)
        u[1, 1] += state[1, 1].min()
        u[2, 2] += state[2, 2].min()
        if self._threshold_is_int:
            new_state = state - u
            u[1, 2] += np.maximum(new_state[1, 2].min() - self._threshold, 0.)
            dist.append((u, 1.))
        else:
            new_state = state - u
            u_ceil = u.copy()
            u_ceil[1, 2] += np.maximum(new_state[1, 2].min() - self._threshold_ceil, 0.)
            dist.append((u_ceil, self._threshold_probability))
            u_floor = u.copy()
            u_floor[1, 2] += np.maximum(new_state[1, 2].min() - self._threshold_floor, 0.)
            dist.append((u_ceil, 1. - self._threshold_probability))
        return dist

    def compute_distribution_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_distribution_state(state=new_state)

    def __str__(self):
        return 'Threshold N continuous policy t={}'.format(self.threshold)


class Threshold_N_norm_dist(RandomizedPolicyOnState, RandomizedPolicyOnStateAndArrivals):

    def __init__(self, state_space: str, threshold: float):
        super(Threshold_N_norm_dist, self).__init__(state_space=state_space)
        self.threshold = threshold

    def compute_matchings_state(self, state: mm.State):
        u = mm.Matching.zeros(state)
        # We match all l1
        u[1, 1] += state[1, 1].min()
        # We match all l2
        u[2, 2] += state[2, 2].min()
        # We update the state with the matchings in l1 and l2 because they have priority and they influence the ones in
        # l3
        new_state = state - u
        nb_l3_items = np.min(new_state[1, 2])
        # We construct the distribution from which we will sample the real threshold
        sigma = 3. / (2. * np.square(2. * np.log(2.)))
        unormalized_dist = np.array([stats.norm.pdf(self.threshold,
                                                    loc=k, scale=sigma) for k in np.arange(nb_l3_items + 1)])
        threshold_distribution = unormalized_dist / np.sum(unormalized_dist)
        # We sample the real threshold according to this distribution
        sample = stats.multinomial(n=1, p=threshold_distribution).rvs()
        matching_threshold = np.where(sample == 1)[1][0]
        # We match all l3 above the threshold
        l3_matchings = np.maximum(nb_l3_items - matching_threshold, 0.)
        u[1, 2] += l3_matchings
        return u

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_matchings_state(state=new_state)

    def compute_distribution_state(self, state: mm.State):
        dist = []
        u = mm.Matching.zeros(state)
        u[1, 1] += state[1, 1].min()
        u[2, 2] += state[2, 2].min()
        new_state = state - u
        nb_l3_items = np.min(new_state[1, 2])
        # We construct the distribution from which we will sample the real threshold
        sigma = 3. / (2. * np.square(2. * np.log(2.)))
        unormalized_dist = np.array([stats.norm.pdf(self.threshold, loc=k, scale=sigma)
                                     for k in np.arange(nb_l3_items + 1)])
        threshold_distribution = unormalized_dist / np.sum(unormalized_dist)
        for matching_threshold in np.arange(nb_l3_items + 1):
            matching = u.copy()
            matching[1, 2] += np.maximum(nb_l3_items - matching_threshold, 0.)
            dist.append((matching, threshold_distribution[matching_threshold]))
        return dist

    def compute_distribution_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_distribution_state(state=new_state)

    def __str__(self):
        return 'Threshold N norm dist policy t={}'.format(self.threshold)


class Threshold_N_norm_dist_all(RandomizedPolicyOnState, RandomizedPolicyOnStateAndArrivals):

    def __init__(self, state_space: str, threshold: float):
        super(Threshold_N_norm_dist_all, self).__init__(state_space=state_space)
        self.threshold = threshold

    def compute_matchings_state(self, state: mm.State):
        # We construct the distribution from which we will sample the real threshold
        max_threshold = state.capacity
        sigma = 3. / (2. * np.square(2. * np.log(2.)))
        unormalized_dist = np.array([stats.norm.pdf(self.threshold,
                                                    loc=k, scale=sigma) for k in np.arange(max_threshold + 1)])
        threshold_distribution = unormalized_dist / np.sum(unormalized_dist)
        # We sample the real threshold according to this distribution
        sample = stats.multinomial(n=1, p=threshold_distribution).rvs()
        matching_threshold = np.where(sample == 1)[1][0]
        # We construct the matching
        u = mm.Matching.zeros(state)
        # We match all l1
        u[1, 1] += state[1, 1].min()
        # We match all l2
        u[2, 2] += state[2, 2].min()
        # We update the state with the matchings in l1 and l2 because they have priority and they influence the ones in
        # l3
        new_state = state - u
        # We match all l3 above the threshold
        u[1, 2] += np.maximum(new_state[1, 2].min() - matching_threshold, 0.)
        return u

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_matchings_state(state=new_state)

    def compute_distribution_state(self, state: mm.State):
        policy_dist = []
        u = mm.Matching.zeros(state)
        u[1, 1] += state[1, 1].min()
        u[2, 2] += state[2, 2].min()
        new_state = state - u
        remaining_items = new_state[1, 2].min()
        max_threshold = state.capacity
        # We construct the distribution from which we will sample the real threshold
        sigma = 3. / (2. * np.square(2. * np.log(2.)))
        unormalized_dist = np.array([stats.norm.pdf(self.threshold, loc=k, scale=sigma)
                                     for k in np.arange(max_threshold + 1)])
        threshold_distribution = unormalized_dist / np.sum(unormalized_dist)
        for matching_threshold in np.arange(int(max_threshold) + 1):
            matching = u.copy()
            matching[1, 2] += np.maximum(remaining_items - matching_threshold, 0.)
            policy_dist.append((matching, threshold_distribution[matching_threshold]))
        return policy_dist

    def compute_distribution_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_distribution_state(state=new_state)

    def __str__(self):
        return 'Threshold N continuous policy t={}'.format(self.threshold)


class Threshold_N_uniform_dist_all(RandomizedPolicyOnState, RandomizedPolicyOnStateAndArrivals):

    def __init__(self, state_space: str):
        super(Threshold_N_uniform_dist_all, self).__init__(state_space=state_space)

    def compute_matchings_state(self, state: mm.State):
        max_threshold = state.capacity
        # We sample the real threshold according to a uniform distribution
        matching_threshold = np.random.randint(low=0, high=int(max_threshold))
        # We construct the matching
        u = mm.Matching.zeros(state)
        # We match all l1
        u[1, 1] += state[1, 1].min()
        # We match all l2
        u[2, 2] += state[2, 2].min()
        # We update the state with the matchings in l1 and l2 because they have priority and they influence the ones in
        # l3
        new_state = state - u
        # We match all l3 above the threshold
        u[1, 2] += np.maximum(new_state[1, 2].min() - float(matching_threshold), 0.)
        return u

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_matchings_state(state=new_state)

    def compute_distribution_state(self, state: mm.State):
        policy_dist = []
        u = mm.Matching.zeros(state)
        u[1, 1] += state[1, 1].min()
        u[2, 2] += state[2, 2].min()
        new_state = state - u
        remaining_items = new_state[1, 2].min()
        max_threshold = state.capacity
        for matching_threshold in np.arange(int(max_threshold) + 1):
            matching = u.copy()
            matching[1, 2] += np.maximum(remaining_items - matching_threshold, 0.)
            policy_dist.append((matching, 1. / max_threshold))
        return policy_dist

    def compute_distribution_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_distribution_state(state=new_state)

    def __str__(self):
        return 'Threshold N uniform policy'


class Threshold_policy(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [s_1, s_2]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all l4
        u[3, 2] += x[3, 2].min()
        # We match all l1
        u[1, 1] += x[1, 1].min()
        # We update the state with the matchings in l4 and l1 because they have priority and they influence the ones in l2 and l3
        new_state = x - u
        # We match all l2 above the s_1 threshold
        l2_matchings = np.maximum(new_state[2, 1].min() - self.thresholds[0], 0.)
        u[2, 1] += l2_matchings
        # We match all l3 above the s_2 threshold
        l3_matchings = np.maximum(new_state[2, 2].min() - self.thresholds[1], 0.)
        u[2, 2] += l3_matchings
        return u

    def __str__(self):
        return 'Threshold policy t={}'.format(self.thresholds)


class TwP_policy(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [d_1, ds_2, s_3]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (3,3)
        u[1, 1] += x[1, 1].min()
        u[3, 3] += x[3, 3].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (1,2) above the threshold in d_1 and we match all (2,3) above the threshold in s_3
        u[1, 2] += np.minimum(np.maximum(new_state.demand(1) - self.thresholds[0], 0.), new_state.supply(2))
        u[2, 3] += np.minimum(np.maximum(new_state.supply(3) - self.thresholds[2], 0.), new_state.demand(2))
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (2,2) above the threshold
        u[2, 2] += np.maximum(new_state[2, 2].min() - self.thresholds[1], 0.)

        return u

    def __str__(self):
        # return 'TwP policy t={}'.format(self.thresholds)
        return 'Thresholds with Priority policy'


class P14T23(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [s_1, s_2]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (3,2)
        u[1, 1] += x[1, 1].min()
        u[3, 2] += x[3, 2].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (2,1) above the threshold in s_1 and we match all (2,2) above the threshold in s_2
        u[2, 1] += np.minimum(np.maximum(new_state.supply(1) - self.thresholds[0], 0.), new_state.demand(2))
        u[2, 2] += np.minimum(np.maximum(new_state.supply(2) - self.thresholds[1], 0.), new_state.demand(2))
        return u

    def __str__(self):
        # return 'P14T23 policy t={}'.format(self.thresholds)
        return "Thresholds in (2,1) and (2,2) with priority in (1,1) and (3,2)"


class P14T23D2(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [s_1, d_2 + d_3]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (3,2)
        u[1, 1] += x[1, 1].min()
        u[3, 2] += x[3, 2].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (2,1) above the threshold in s_1 and we match all (2,2) above the threshold in s_2
        u[2, 1] += np.min([np.maximum(new_state.supply(1) - self.thresholds[0], 0.),
                           np.maximum(new_state.demand(2) + new_state.demand(3) - self.thresholds[1], 0.),
                           new_state.demand(2)])
        u[2, 2] += new_state[2, 2].min()
        return u

    def __str__(self):
        return 'P14T23D2 policy t={}'.format(self.thresholds)


class P13T24D2(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [s_1, d_2 + d_3]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (3,2)
        u[1, 1] += x[1, 1].min()
        u[2, 2] += x[2, 2].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (2,1) above the threshold in s_1 and we match all (2,2) above the threshold in s_2
        u[3, 2] += new_state[3, 2].min()
        new_state = x - u
        u[2, 1] += np.min([np.maximum(new_state.supply(1) - self.thresholds[0], 0.),
                           np.maximum(new_state.demand(2) + new_state.demand(3) - self.thresholds[1], 0.),
                           new_state.demand(2)])
        return u

    def __str__(self):
        # return 'P13T24D2 policy t={}'.format(self.thresholds)
        return "Thresholds in (2,1) and (3,2) with priority in (1,1) and (2,2)"


class OptimalW(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [s_1 or d_3]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (3,2)
        u[1, 1] += x[1, 1].min()
        if x.demand(1) >= x.supply(1):
            u[3, 2] += x[3, 2].min()
            u[2, 2] += x[2, 2].min()
            return u
        else:
            # We update the state with the matchings because they have priority and they influence the future ones
            new_state = x - u
            # We match all (2,1) above the threshold in s_1 and we match all (3,2) above the threshold in d_3
            u[2, 1] += np.minimum(np.maximum(new_state.supply(1) - self.thresholds[0], 0.), new_state.demand(2))
            u[3, 2] += np.minimum(np.maximum(new_state.demand(3) - self.thresholds[0], 0.), new_state.supply(2))
            # We update the state with the matchings because they have priority and they influence the future ones
            new_state = x - u
            # We match all (2,2) remaining
            u[2, 2] += new_state[2, 2].min()
            # We update the state with the matchings because they have priority and they influence the future ones
            new_state = x - u
            # We match all (3,2) remaining
            u[3, 2] += new_state[3, 2].min()
            return u

    def __str__(self):
        return 'OptimalW policy t={}'.format(self.thresholds)


class OptimalWBis(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [s_1 or d_3]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (3,2)
        u[1, 1] += x[1, 1].min()
        if x.demand(1) >= x.supply(1):
            u[3, 2] += x[3, 2].min()
            u[2, 2] += x[2, 2].min()
            return u
        else:
            # We update the state with the matchings because they have priority and they influence the future ones
            new_state = x - u
            # We match all (2,1) above the threshold in s_1 and we match all (3,2) above the threshold in d_3
            u[2, 1] += np.minimum(np.maximum(new_state.supply(1) - self.thresholds[0], 0.), new_state.demand(2))
            # We update the state with the matchings because they have priority and they influence the future ones
            new_state = x - u
            if new_state.demand(2) > 0:
                u[3, 2] += np.minimum(np.maximum(new_state.demand(3) - self.thresholds[0], 0.), new_state.supply(2))
            else:
                u[3, 2] += new_state[3, 2].min()
            # We update the state with the matchings because they have priority and they influence the future ones
            new_state = x - u
            # We match all (2,2) remaining
            u[2, 2] += new_state[2, 2].min()
            return u

    def __str__(self):
        return 'OptimalWBis policy t={}'.format(self.thresholds)


class P13T24(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [s_1, s_2]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (2,2)
        u[1, 1] += x[1, 1].min()
        u[2, 2] += x[2, 2].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (2,1) above the threshold in s_1 and we match all (3,2) above the threshold in s_2
        u[2, 1] += np.minimum(np.maximum(new_state.supply(1) - self.thresholds[0], 0.), new_state.demand(2))
        u[3, 2] += np.minimum(np.maximum(new_state.supply(2) - self.thresholds[1], 0.), new_state.demand(2))
        return u

    def __str__(self):
        return 'P13T24 policy t={}'.format(self.thresholds)


class ThresholdsWithPriorities(PolicyOnState):

    def __init__(self, state_space: str, matching_order: mm.EdgesData, thresholds: mm.EdgesData):
        """
        :param state_space: State space on which is defined the policy.
        :param matching_order: EdgeData giving the order in which each edge will be matched.
        :param thresholds: EdgeData giving the threshold above which each edge will be matched.
        """
        super(ThresholdsWithPriorities, self).__init__(state_space=state_space)
        assert matching_order.matching_graph is thresholds.matching_graph
        self.matching_order = matching_order
        self.thresholds = thresholds

    def compute_matchings_state(self, state: mm.State) -> mm.Matching:
        """
        :param: State from which to match from.
        :return: Matching based on the State and the Policy used.
        """
        u = mm.Matching.zeros(state)
        current_state = state.copy()
        for edge_index in self.matching_order.data:
            edge = state.matching_graph.edges[edge_index]
            # We match all above the threshold
            u[edge] += np.maximum(current_state[edge].min() - self.thresholds[edge], 0.)
            # We update the state with the matchings because they have priority and they influence the future ones
            current_state = state - u
        return u

    def __str__(self):
        return 'TwP p={} t={}'.format(self.matching_order, self.thresholds)


class TwPbis_policy(Policy):

    def __init__(self, thresholds):
        # We store the thresholds as [d_1, ds_2, s_3]
        self.thresholds = thresholds

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (3,3)
        u[1, 1] += x[1, 1].min()
        u[3, 3] += x[3, 3].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (1,2) above the threshold in d_1 and we match all (2,3) above the threshold in s_3
        u[1, 2] += np.maximum(new_state[1, 2].min() - self.thresholds[0], 0.)
        u[2, 3] += np.maximum(new_state[2, 3].min() - self.thresholds[2], 0.)
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (2,2) above the threshold
        u[2, 2] += np.maximum(new_state[2, 2].min() - self.thresholds[1], 0.)

        return u

    def __str__(self):
        # return 'TwP policy t={}'.format(self.thresholds)
        return 'Thresholds with Priority policy'


class TwMW_policy(Policy):

    def __init__(self, thresholds, costs):
        # We store the thresholds as [d_1, s_3]
        self.thresholds = thresholds
        self.costs = costs

    def match(self, x):
        u = mm.Matching.zeros(x)
        # We match all (1,1) and all (3,3)
        u[1, 1] += x[1, 1].min()
        u[3, 3] += x[3, 3].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We remove in d_1 and s_3
        uprime = mm.Matching.zeros(x)
        uprime[1, 2] = np.minimum(new_state[1, 2].min(), self.thresholds[0])
        uprime[2, 3] = np.minimum(new_state[2, 3].min(), self.thresholds[1])
        xprime = new_state - uprime

        if not xprime.matchings_available():
            return u

        # We do MaxWeight on xprime
        # We will use a list of edges candidates that could achieve the max weight
        candidate_list = xprime.matchings_available_subgraph().maximal_matchings()
        candidates_costs = np.zeros(len(candidate_list))
        for i, candidate in enumerate(candidate_list):
            # For each candidate we compute the costs induced by its matchings
            for edge in candidate:
                candidates_costs[i] += np.sum(np.multiply(self.costs[edge], xprime[edge]))
        # We create a Matching out of the most expensive candidate
        for edge in candidate_list[np.argmax(candidates_costs)]:
            u[edge] += xprime[edge].min()
        return u

    def __str__(self):
        return 'TwMW policy t={}'.format(self.thresholds)


class MaxWeight_policy(Policy):

    def __init__(self, costs):
        self.costs = costs

    def match(self, x):
        # We suppose that state is a stable state (i.e, before the arrivals, no more matching could have been done)
        u = mm.Matching.zeros(x)
        # We will use a list of edges candidates that could achieve the max weight
        candidate_list = x.matchings_available_subgraph().maximal_matchings()
        candidates_costs = np.zeros(len(candidate_list))
        for i, candidate in enumerate(candidate_list):
            # For each candidate we compute the costs induced by its matchings
            for edge in candidate:
                candidates_costs[i] += np.sum(np.multiply(self.costs[edge], x[edge]))
        # We create a Matching out of the most expensive candidate
        for edge in candidate_list[np.argmax(candidates_costs)]:
            u[edge] += 1.
        return u

    def __str__(self):
        return 'MaxWeight policy'


class OptimiseMW(mip.Problem):
    """ Implementation of the optimisation program that is used in the MaxWeight Policy. """

    def model(self, x: mm.State, costs: mm.NodesData):
        """
        :param x: State upon which the matchings are done.
        :param costs: NodesData giving the cost at each node.
        """
        nb_edges = x.matching_graph.nb_edges
        nb_nodes = x.matching_graph.n
        edges_to_nodes = x.matching_graph.edges_to_nodes
        # The variables are the number of matching in each edge
        self.u = u = mip.VarVector([nb_edges], "u", mip.INT, lb=0, ub=mip.VAR_INF)
        # The goal is to maximize u\\cdot \\nabla h(state) where h(state)=\\sum_{i\\in \\mathcal{D}\\cup\\mathcal{S}} c_i x_i^2.
        # mip.maximize(mip.sum_(costs.data[i] * state.data[i] * mip.sum_(edges_to_nodes[i, j] * u[j] for j in range(nb_edges))
        #                       for i in range(nb_nodes)))
        mip.maximize(
            mip.sum_(np.sum(np.multiply(costs[edge], x[edge])) * u[i] for i, edge in enumerate(x.matching_graph.edges)))

        # The inequalities constraints
        # The number of matchings can not be higher than the number of items in the system
        for i in range(nb_nodes):
            mip.sum_(edges_to_nodes[i, j] * u[j] for j in range(nb_edges)) <= x.data[i]


class MaxWeight(PolicyOnState, PolicyOnStateAndArrivals):
    """
    This policy, given a State state, gives any feasible Matching u which maximise the following optimisation problem:
    u\\cdot \\nabla h(state) where h(state)=\\sum_{i\\in \\mathcal{D}\\cup\\mathcal{S}} c_i x_i^2. i.e, it matches the most
    costly nodes (a product of the individual cost and the number of item in the node) in priority.
    """

    def __init__(self, state_space: str, costs: mm.NodesData):
        """
        :param state_space: State space on which is defined the policy.
        :param costs: NodesData giving the cost at each node.
        """
        super(MaxWeight, self).__init__(state_space=state_space)
        self.costs = costs

    def compute_matchings_state(self, state: mm.State):
        """
        :param state: State upon which the matchings are done.
        :return: Matching resulted from using the MaxWeight Policy on the State state.
        """
        prob = OptimiseMW("MaxWeight")
        prob.model(x=state, costs=self.costs)
        prob.optimize(silent=True)
        if prob.is_solution:
            u_star = mm.Matching.zeros(state)
            for i in np.arange(state.matching_graph.nb_edges):
                u_star[state.matching_graph.edges[i]] += prob.u[i].val
            return u_star
        else:
            raise ValueError('The MIP optimizer has not found a solution')

    def compute_matchings_state_and_arrival(self, state: mm.State, arrivals: mm.State):
        new_state = state + arrivals
        return self.compute_matchings_state(state=new_state)

    def __str__(self):
        return 'MaxWeight policy'


class Stolyar_policy(Policy):

    def __init__(self, x_0, rewards, beta, costs=None):
        self.previous_x = x_0.copy()
        self.previous_match = mm.Matching.zeros(x_0)
        self.virtual_x = mm.Virtual_State(x_0.data.copy(), x_0.matchingGraph)
        self.rewards = rewards
        self.beta = beta
        self.incomplete_matchings = []
        # We can include holding costs in Stolyar algorithm by giving the costs as parameter and changing the virtual system update
        if costs is not None:
            self.costs = costs
            self.update_vs = self.update_virtual_system_wCosts
        else:
            self.update_vs = self.update_virtual_system

    def match(self, x):
        # update the virtual system: add previous arrivals, apply algo 1 and add the matching to the incomplete queue
        self.update_vs(x)
        # We scan the queue in FCFS order until we find a feasible match given state
        # We return the feasible if one was found or we return 0.
        for match in self.incomplete_matchings:
            if (match.data <= x.data).all():
                u = mm.Matching(x, match.data)
                self.incomplete_matchings.remove(match)
                self.previous_match = u
                return u
        u = mm.Matching.zeros(x)
        self.previous_match = u
        return u

    def update_virtual_system(self, x):
        # We add the previous arrivals to the virtual state
        arrivals = mm.State(x.data - self.previous_x.data + self.previous_match.data, x.matchingGraph)
        self.virtual_x += arrivals
        self.previous_x = x.copy()
        # We use Stolyar algorithm to get the matching based on the virtual system
        virtual_match = mm.Virtual_Matching.zeros(x)
        matchings_values = np.zeros(len(self.rewards))
        for i, edge in enumerate(x.matchingGraph.edges):
            matchings_values[i] = self.rewards[i] + self.beta * np.sum(self.virtual_x[edge])
        virtual_match[x.matchingGraph.edges[np.argmax(matchings_values)]] += 1.
        # We perform the matching in the virtual system
        self.virtual_x -= virtual_match
        # We add the matching to the list of incomplete matchings
        self.incomplete_matchings.append(virtual_match)

    def update_virtual_system_wCosts(self, x):
        # We add the previous arrivals to the virtual state
        arrivals = mm.State(x.data - self.previous_x.data + self.previous_match.data, x.matchingGraph)
        self.virtual_x += arrivals
        self.previous_x = x.copy()
        # We use Stolyar algorithm to get the matching based on the virtual system
        virtual_match = mm.Virtual_Matching.zeros(x)
        matchings_values = np.zeros(len(self.rewards))
        for i, edge in enumerate(x.matchingGraph.edges):
            matchings_values[i] = self.rewards[i] + self.beta * np.dot(self.costs[edge].reshape(1, -1),
                                                                       self.virtual_x[edge].reshape(-1, 1))
        virtual_match[x.matchingGraph.edges[np.argmax(matchings_values)]] += 1.
        # We perform the matching in the virtual system
        self.virtual_x -= virtual_match
        # We add the matching to the list of incomplete matchings
        self.incomplete_matchings.append(virtual_match)

    def reset_policy(self, x_0):
        # We reset the previous state, matching and virtual state
        self.previous_x = x_0.copy()
        self.previous_match = mm.Matching.zeros(x_0)
        self.virtual_x = mm.Virtual_State(x_0.data.copy(), x_0.matchingGraph)
        # We empty the list of incomplete matchings
        self.incomplete_matchings = []

    def __str__(self):
        if hasattr(self, 'costs'):
            return 'Stolyar policy with costs r={}, b={}'.format(self.rewards, self.beta)
        else:
            return 'Stolyar policy r={}, b={}'.format(self.rewards, self.beta)


class hMWT_policy(Policy):

    def __init__(self, matchingGraph, Workload_index, alpha, costs, beta, kappa, theta, delta_plus, NUmax):
        # The Workload_index must be a list with two element.
        # The first one is 'd' if we are looking at demand classes or 's' for supply classes
        # The second one is a tuple with the classes
        assert Workload_index[0] == 'd' or Workload_index[0] == 's'
        self.Workload_index = Workload_index
        self.NUmax = NUmax  # The maximal number of matchings that can be done at once
        self.alpha = alpha  # The mean arrival rate
        self.costs = costs  # The linear costs
        self.beta = beta  # The perturbation parameter to turn state (or w) in xtil (or wtil)
        self.kappa = kappa
        self.theta = theta
        self.delta_plus = delta_plus

        if self.Workload_index[0] == 'd':
            D = self.Workload_index[1]
            S_D = tuple(matchingGraph.demandToSupply[D])
            self.Idle_index = [matchingGraph.edgeIndex((d, s)) for s in S_D for d in matchingGraph.supplyToDemand[(s,)]
                               if d not in D]
            self.XiD = np.array(
                [1. if idx in D else 0. for idx in matchingGraph.demand_class_set] + [-1. if idx in S_D else 0. for idx
                                                                                      in
                                                                                      matchingGraph.supply_class_set])

            # We compute the optimal threshold tau_star
            Delta_plus = self.alpha.demand(np.array(D)).sum()
            Delta_minus = self.alpha.supply(np.array(S_D)).sum()
            delta = Delta_minus - Delta_plus
            self.barC_plus = self.costs.demand(np.array(D)).min() + self.costs.supply(
                np.array(matchingGraph.Scomplement(S_D))).min()
            self.barC_minus = self.costs.demand(np.array(matchingGraph.Dcomplement(D))).min() + self.costs.supply(
                np.array(S_D)).min()
            # sigmaSq_delta = (delta + 1.)**2 * Delta_plus*(1.-Delta_minus) + (delta - 1.)**2 * (1.-Delta_plus)*Delta_minus # Ana code version
            sigmaSq_delta = Delta_plus * (1. - Delta_minus) + (1. - Delta_plus) * Delta_minus - delta ** 2  # My version
            self.tau_star = 0.5 * (sigmaSq_delta / delta) * np.log(1. + (self.barC_plus / self.barC_minus))

            # We compute coefficients needed for hat_h
            hat_etaSS = self.tau_star * self.barC_minus
            self.Aplus = self.barC_plus / (2 * delta)
            self.Bplus = (sigmaSq_delta * self.Aplus - hat_etaSS) / delta
            self.THETA = 2 * delta / sigmaSq_delta
            self.Aminus = -self.barC_minus / (self.THETA * sigmaSq_delta)
            self.Bminus = 2 * self.Aminus / self.THETA - hat_etaSS / delta
            self.Dminus = (self.Bplus - self.Bminus) / self.THETA

            self.match = self.match_D
        else:
            S = self.Workload_index[1]
            D_S = tuple(matchingGraph.supplyToDemand[S])
            self.Idle_index = [matchingGraph.edgeIndex((d, s)) for d in D_S for s in matchingGraph.demandToSupply[(d,)]
                               if s not in S]
            self.XiS = np.array(
                [-1. if idx in D_S else 0. for idx in matchingGraph.demand_class_set] + [1. if idx in S else 0. for idx
                                                                                         in
                                                                                         matchingGraph.supply_class_set])

            # We compute the optimal threshold tau_star
            Delta_plus = self.alpha.supply(np.array(S)).sum()
            Delta_minus = self.alpha.demand(np.array(D_S)).sum()
            delta = Delta_minus - Delta_plus
            self.barC_plus = self.costs.supply(np.array(S)).min() + self.costs.demand(
                np.array(matchingGraph.Dcomplement(D_S))).min()
            self.barC_minus = self.costs.supply(np.array(matchingGraph.Scomplement(S))).min() + self.costs.demand(
                np.array(D_S)).min()
            # sigmaSq_delta = (delta + 1.)**2 * Delta_plus*(1.-Delta_minus) + (delta - 1.)**2 * (1.-Delta_plus)*Delta_minus # Ana code version
            sigmaSq_delta = Delta_plus * (1. - Delta_minus) + (1. - Delta_plus) * Delta_minus - delta ** 2  # My version
            self.tau_star = 0.5 * (sigmaSq_delta / delta) * np.log(1. + (self.barC_plus / self.barC_minus))

            # We compute coefficients need for hat_h
            hat_etaSS = self.tau_star * self.barC_minus
            self.Aplus = self.barC_plus / (2 * delta)
            self.Bplus = (sigmaSq_delta * self.Aplus - hat_etaSS) / delta
            self.THETA = 2 * delta / sigmaSq_delta
            self.Aminus = -self.barC_minus / (self.THETA * sigmaSq_delta)
            self.Bminus = 2 * self.Aminus / self.THETA - hat_etaSS / delta
            self.Dminus = (self.Bplus - self.Bminus) / self.THETA

            self.match = self.match_S

    def match_D(self, x):
        # We compute the workload w
        w = np.inner(x.data, self.XiD)
        # We computed the pertubated states and workload
        xtil = x.data + self.beta * (np.exp(-x.data / self.beta) - 1.)
        wtil = np.sign(w) * (np.abs(w) + self.beta * (np.exp(-np.abs(w) / self.beta) - 1.))

        # The function h is the sum of two function: \hat{h}(w) et h_c(state)
        # We compute the gradient of h_c(state)
        grad_ctil = np.multiply(self.costs.data, (1. - np.exp(-x.data / self.beta)))
        # print('grad_ctil',grad_ctil)
        if w >= 0:
            grad_barCtil = self.barC_plus * (1. - np.exp(-w / self.beta)) * self.XiD
        else:
            grad_barCtil = -self.barC_minus * (1. - np.exp(w / self.beta)) * self.XiD
        # print('xtil',xtil)
        # print('cxtil',np.inner(self.costs.data, xtil))
        grad_h_c = 2. * self.kappa * (
                    np.inner(self.costs.data, xtil) - np.maximum(self.barC_plus * wtil, -self.barC_minus * wtil)) * (
                               grad_ctil - grad_barCtil)
        # We compute the derivative of \hat{h}(w)
        if w >= 0:
            hat_hprime = 2 * self.Aplus * w + self.Bplus
        elif w < 0 and w >= -self.tau_star:
            # hat_hprime = 2*self.Aminus + self.Bminus + self.Dminus*self.THETA*np.exp(self.THETA*w) # Ana code version
            hat_hprime = 2 * self.Aminus * w + self.Bminus + self.Dminus * self.THETA * np.exp(
                self.THETA * w)  # My version
        else:
            ws = self.tau_star + w
            hat_hprime = (self.barC_minus / self.delta_plus) * (ws + (1 / self.theta) * (1 - np.exp(self.theta * ws)))
        # Finally, we compute the gradient of h(state)
        grad_h = mm.NodesData(hat_hprime * self.XiD + grad_h_c, x.matchingGraph)
        grad_h_index = np.array(
            [np.sum(grad_h[x.matchingGraph.edges[i]]) for i in np.arange(len(x.matchingGraph.edges))])

        prob = hMWT("hMWT")
        prob.model(x, grad_h_index, w, self.tau_star, self.Idle_index, self.NUmax)
        prob.optimize(False)
        if prob.is_solution == True:
            u_star = mm.Matching.zeros(x)
            for i in np.arange(len(x.matchingGraph.edges)):
                u_star[x.matchingGraph.edges[i]] += prob.u[i].val
            return u_star
        else:
            raise ValueError('The MIP optimizer has not found a solution')
        # prob.printSolution()

    def match_S(self, x):
        # We compute the workload w
        w = np.inner(x.data, self.XiS)
        # We computed the pertubated states and workload
        xtil = x.data + self.beta * (np.exp(-x.data / self.beta) - 1.)
        wtil = np.sign(w) * (np.abs(w) + self.beta * (np.exp(-np.abs(w) / self.beta) - 1.))

        # The function h is the sum of two function: \hat{h}(w) et h_c(state)
        # We compute the gradient of h_c(state)
        grad_ctil = np.multiply(self.costs.data, (1. - np.exp(-x.data / self.beta)))
        if w >= 0:
            grad_barCtil = self.barC_plus * (1. - np.exp(-w / self.beta)) * self.XiS
        else:
            grad_barCtil = -self.barC_minus * (1. - np.exp(w / self.beta)) * self.XiS
        grad_h_c = 2. * self.kappa * (
                    np.inner(self.costs.data, xtil) - np.maximum(self.barC_plus * wtil, -self.barC_minus * wtil)) * (
                               grad_ctil - grad_barCtil)
        # We compute the derivative of \hat{h}(w)
        if w >= 0:
            hat_hprime = 2 * self.Aplus * w + self.Bplus
        elif w < 0 and w >= -self.tau_star:
            # hat_hprime = 2*self.Aminus + self.Bminus + self.Dminus*self.THETA*np.exp(self.THETA*w) # Ana code version
            hat_hprime = 2 * self.Aminus * w + self.Bminus + self.Dminus * self.THETA * np.exp(
                self.THETA * w)  # My version
        else:
            ws = self.tau_star + w
            hat_hprime = (self.barC_minus / self.delta_plus) * (ws + (1 / self.theta) * (1 - np.exp(self.theta * ws)))
        # Finally, we compute the gradient of h(state)
        grad_h = hat_hprime * self.XiS + grad_h_c

        prob = hMWT("hMWT")
        prob.model(x, grad_h, w, self.tau_star, self.Idle_index, self.NUmax)
        prob.optimize(False)
        if prob.is_solution == True:
            u_star = mm.Matching.zeros(x)
            for i in np.arange(len(x.matchingGraph.edges)):
                u_star[x.matchingGraph.edges[i]] += prob.u[i].val
            return u_star
        else:
            raise ValueError('The MIP optimizer has not found a solution')
        # prob.printSolution()

    def reset_policy(self, x_0):
        pass

    def __str__(self):
        if self.Workload_index[0] == 'd':
            # return 'hMWT policy D={}'.format(self.Workload_index[1])
            return 'hMWT policy'
        else:
            return 'hMWT policy S={}'.format(self.Workload_index[1])


# We create a class MILP which maximizes f.u under the constraints Au <= b for u an array of integers between lb and ub
class hMWT(mip.Problem):
    def model(self, x, grad_h, w, tau_star, Idle_index, NUmax):
        nb_edges = len(x.matchingGraph.edges)
        # The variables are the number of matching in each edge
        # u[i] correspond to the number of matching in state.matching_graph.edges[i]
        self.u = u = mip.VarVector([nb_edges], "u", mip.INT, lb=0, ub=NUmax)
        # The goal is to maximize grad_h*u
        mip.maximize(mip.sum_(grad_h[i] * u[i] for i in range(nb_edges)))

        ### The inequalities constraints ###
        # The number of matchings can not be higher than NUmax
        mip.sum_(u[i] for i in range(nb_edges)) <= NUmax

        # The number of matchings can not be higher than the number of items in the system
        for i in x.matchingGraph.demand_class_set:
            linked_edges = [x.matchingGraph.edgeIndex((i, s)) for s in x.matchingGraph.demandToSupply[(i,)]]
            mip.sum_(u[k] for k in linked_edges) <= x.demand(i)
        for j in x.matchingGraph.supply_class_set:
            linked_edges = [x.matchingGraph.edgeIndex((d, j)) for d in x.matchingGraph.supplyToDemand[(j,)]]
            mip.sum_(u[k] for k in linked_edges) <= x.supply(j)

        # The workload process can not be higher than the threshold
        mip.sum_(u[i] for i in Idle_index) <= np.maximum(-tau_star - w, 0.)
        

