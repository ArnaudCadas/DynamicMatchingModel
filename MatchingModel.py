import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, chain, product
from typing import Tuple, List

# import po as po


class MatchingGraph:

    def __init__(self, edges, nb_demand_classes, nb_supply_classes):
        # Edges must be a list of tuples ('i','j') if demand class i can be matched with supply class j
        self.edges = edges
        self.nb_demand_classes = nb_demand_classes
        self.nb_supply_classes = nb_supply_classes
        # We compute the set and all subsets of demand classes
        self.demand_class_set = np.arange(1, self.nb_demand_classes + 1)
        self.demand_class_subsets = [tuple(c) for c in chain.from_iterable(
            combinations(self.demand_class_set, r) for r in self.demand_class_set)]
        # We create a dictionary which maps each subset of demand classes to the subset of supply classes to witch its linked
        self.build_demandToSupply()
        # We compute the set and all subsets of supply classes
        self.supply_class_set = np.arange(1, nb_supply_classes + 1)
        self.supply_class_subsets = [tuple(c) for c in chain.from_iterable(
            combinations(self.supply_class_set, r) for r in np.arange(1, nb_supply_classes + 1))]
        # We create a dictionary which maps each subset of supply classes to the subset of demand classes to witch its linked
        self.build_supplyToDemand()
        # We create a matrix to transform an EdgeData into a NodesData (if the values are scalars)
        self.edges_to_nodes = np.zeros((self.n, self.nb_edges))
        for i, edge in enumerate(self.edges):
            edge_as_nodes_data = NodesData.zeros(self)
            edge_as_nodes_data[edge] = 1.
            self.edges_to_nodes[:, i] = edge_as_nodes_data.data

    @property
    def n(self):
        return self.nb_demand_classes + self.nb_supply_classes

    @property
    def nb_edges(self) -> int:
        """
        :return: Number of edges in the graph.
        """
        return len(self.edges)

    @property
    def nodes(self):
        # We create a list of all nodes with first the demand classes and then the supply classes, both in increasing order
        return np.array(['d' + str(i) for i in self.demand_class_set] + ['s' + str(j) for j in self.supply_class_set])

    def build_demandToSupply(self):
        # We create a dictionary which maps each subset of demand classes to the subset of supply classes to witch its linked
        self.demandToSupply = {}
        for subset in self.demand_class_subsets:
            supply_subset = set()
            for edge in self.edges:
                if edge[0] in subset:
                    supply_subset.add(edge[1])
            self.demandToSupply[tuple(subset)] = list(supply_subset)

    def build_supplyToDemand(self):
        # We create a dictionary which maps each subset of supply classes to the subset of demand classes to witch its linked
        self.supplyToDemand = {}
        for subset in self.supply_class_subsets:
            demand_subset = set()
            for edge in self.edges:
                if edge[1] in subset:
                    demand_subset.add(edge[0])
            self.supplyToDemand[tuple(subset)] = list(demand_subset)

    def isEdge(self, e):
        return e in self.edges

    def Dcomplement(self, D):
        return tuple(i for i in self.demand_class_set if i not in D)

    def Scomplement(self, S):
        return tuple(j for j in self.supply_class_set if j not in S)

    def edgeIndex(self, e):
        if self.isEdge(e):
            return self.edges.index(e)
        else:
            raise ValueError('This value does not correspond to an edge of the matching graph')

    def degree(self):
        # We count the degree of each node
        d = NodesData(np.zeros(self.n), self)
        for edge in self.edges:
            d[edge] += 1
        return d

    def maximal_matchings(self):
        # We compute all the maximal matchings of the matching graph. This function only makes sense if it called on the MatchingGraph returned by available_matchings_subgraph()
        list_maximal_matchings = []
        deg = self.degree().data
        # We look at all nodes of degree superior than 2 (there can not be more than 2)
        if (deg < 2).all():
            # If the degree of each node is less than two then all the edges of the matching graph form the only maximal matching
            list_maximal_matchings.append(self.edges)
        elif np.sum(deg >= 2) == 1:
            # We get the index of the node
            node_index = int(np.where(deg >= 2)[0])
            # We test if it is a demand class or a supply class
            if node_index < self.nb_demand_classes:
                # We transform the node index to the demand class
                node = node_index + 1
                # We get all the supply classes that can be matched with arrival demand class
                S_i = self.demandToSupply[(node,)]
                # We test if there is still an edge we can matched after having matched the arrival demand class
                remaining_edge = [el for el in set(self.edges).difference((node, supply_class) for supply_class in S_i)]
                if remaining_edge:
                    for supply_class in S_i:
                        list_maximal_matchings.append([(node, supply_class), remaining_edge[0]])
                else:
                    for supply_class in S_i:
                        list_maximal_matchings.append([(node, supply_class)])
            else:
                # We transform the node index to the supply class
                node = node_index - self.nb_demand_classes + 1
                # We get all the demand classes that can be matched with arrival supply class
                D_j = self.supplyToDemand[(node,)]
                # We test if there is still an edge we can matched after having matched the arrival supply class
                remaining_edge = [el for el in set(self.edges).difference((demand_class, node) for demand_class in D_j)]
                if remaining_edge:
                    for demand_class in D_j:
                        list_maximal_matchings.append([(demand_class, node), remaining_edge[0]])
                else:
                    for demand_class in D_j:
                        list_maximal_matchings.append([(demand_class, node)])
        else:
            # We get the demand class and the supply class of the arrivals by selecting the only two nodes that have a degree greater than 2
            arrivals_classes_index = np.where(deg >= 2)[0]
            arrivals_classes = (arrivals_classes_index[0] + 1, arrivals_classes_index[1] - self.nb_demand_classes + 1)
            # We get all the supply classes that can be matched with arrival demand class
            S_i = self.demandToSupply[(arrivals_classes[0],)]
            # We get all the demand classes that can be matched with arrival supply class
            D_j = self.supplyToDemand[(arrivals_classes[1],)]

            if arrivals_classes[1] in S_i:
                # If both arrivals can be matched together, we add their matching as a maximal matching and remove them from S_i and D_j
                list_maximal_matchings.append([arrivals_classes])
                S_i.remove(arrivals_classes[1])
                D_j.remove(arrivals_classes[0])

            # We get all remaining maximal matchings by combining any possible matching of arrival demand class with any possible matching of arrival supply class
            for classes in product(S_i, D_j):
                list_maximal_matchings.append([(arrivals_classes[0], classes[0]), (classes[1], arrivals_classes[1])])

        return list_maximal_matchings

    def __eq__(self, other):
        if isinstance(other, MatchingGraph):
            return np.all(self.edges == other.edges) and self.nb_demand_classes == other.nb_demand_classes and \
                   self.nb_supply_classes == other.nb_supply_classes
        return NotImplemented


# We define a class NodesData which is a data structure for our system.
# It stores a value for each classes of demand and supply items.
# It is used for example to store the length of the queues, the holding costs or the arrival rates
class NodesData:
    """
    Stores data related to the edges of a MatchingGraph in an array.
    """
    # TODO: add assertion about equality of matching graph in operators

    def __init__(self, data: np.array, matching_graph: MatchingGraph):
        """
        :param data: Array which stores the data related to each node. It is organized as such: first the demand items,
            then the supply items and both sorted by classes in increasing order. This means that index i represent
            demand class i + 1 and index nb_demand_classes + j represent supply class j + 1.
        :param matching_graph: MatchingGraph to which this data is related to.
        """
        self.data = data
        self.matching_graph = matching_graph

    @classmethod
    def fromDict(cls, data, matching_graph):
        data_array = np.zeros(matching_graph.n)
        # The values must be stored in a dictionnary D where the keys are the nodes
        for node in data.keys():
            if node not in matching_graph.nodes:
                raise ValueError('A key from the dictionnary does not corespond to a node of the matching graph')
            elif node[0] == 'd':
                data_array[int(node[1]) - 1] = data[node]
            else:
                data_array[matching_graph.nb_demand_classes + int(node[1]) - 1] = data[node]
        return cls(data_array, matching_graph)

    @classmethod
    def zeros(cls, matching_graph):
        # We create an empty state
        return cls(np.zeros(matching_graph.n), matching_graph)

    @classmethod
    def items(cls, demand_items, supply_items, matching_graph):
        # We create a state by giving two separate array for demand and supply items
        return cls(np.hstack((demand_items, supply_items)), matching_graph)

    def demand(self, classes):
        return self.data[classes - 1]

    def supply(self, classes):
        return self.data[classes - 1 + self.matching_graph.nb_demand_classes]

    def __getitem__(self, edge: Tuple):
        demand_class, supply_class = edge
        return self.data[[demand_class - 1, self.matching_graph.nb_demand_classes + supply_class - 1]]

    def __setitem__(self, edge: Tuple, value):
        demand_class, supply_class = edge
        self.data[[demand_class - 1, self.matching_graph.nb_demand_classes + supply_class - 1]] = value

    def __add__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.data + other.data, self.matching_graph)
        return NotImplemented

    def __iadd__(self, other):
        if type(other) == self.__class__:
            self.data += other.data
            return self
        return NotImplemented

    def __eq__(self, other):
        if type(other) == self.__class__:
            return np.all(self.data == other.data) and self.matching_graph == other.matching_graph
        return NotImplemented

    def copy(self):
        return self.__class__(self.data.copy(), self.matching_graph)


# We define a class State which is a NodesData with the constraint that demand and supply items must be positives and their sum equal
# It is used for example to store the length of the queues, arrival items or matchings
class State(NodesData):

    def __init__(self, values: np.array, matching_graph: MatchingGraph, capacity=np.inf):
        """
        :param values: Array which stores the number of items in each node. It is organized as such: first the demand
            items, then the supply items and both sorted by classes in increasing order. This means that index i
            represent demand class i + 1 and index nb_demand_classes + j represent supply class j + 1.
        :param matching_graph: MatchingGraph to which this data is related to.
        :param capacity: Maximal number of items that can be held for each node. Default is infinite.
        """
        self.capacity = capacity
        # We use the NodesData initialization
        super(State, self).__init__(values, matching_graph)
        # We test that the number of demand items and the number of supply items are positives
        if np.any(self.data < 0):
            raise ValueError("The number of demand items and the number of supply items must be positives.")
        # We test that the number of demand items and the number of supply items are less than the capacity
        if np.any(self.data > self.capacity):
            raise ValueError("The number of demand items and the number of supply items must be less than capacity.")
        # We test that the sum of demand items is equal to the sum of supply items
        if self.demand(self.matching_graph.demand_class_set).sum() != self.supply(
                self.matching_graph.supply_class_set).sum():
            raise ValueError("The sum of demand items must be equal to the sum of supply items.")

    @classmethod
    def zeros(cls, matching_graph, capacity=np.inf):
        # We create an empty state
        return cls(np.zeros(matching_graph.n), matching_graph, capacity)

    def matchings_available(self):
        # We construct a list of all the edges which can be matched given the State
        list_edges = []
        for edge in self.matching_graph.edges:
            if (self[edge] >= 1).all():
                list_edges.append(edge)
        return list_edges

    def complete_matchings_available(self):
        matchings_list = []
        for matchings_numbers in product(*[np.arange(int(np.min(self[edge])) + 1) for edge in self.matching_graph.edges]):
            try:
                matching = Matching(state=self, values=np.array(matchings_numbers))
            except ValueError:
                pass
            else:
                matchings_list.append(matching)
        return matchings_list

    def matchings_available_subgraph(self):
        # We construct a subgraph composed of all the edges which can be matched given the State
        return MatchingGraph(self.matchings_available(), self.matching_graph.nb_demand_classes,
                             self.matching_graph.nb_supply_classes)

    def __setitem__(self, edge: Tuple, value):
        if np.any(value < 0):
            raise ValueError("The number of demand items and the number of supply items must be positives.")
        if np.any(value > self.capacity):
            raise ValueError("The number of demand items and the number of supply items must be less than capacity.")
        super(State, self).__setitem__(edge, value)

    def __add__(self, other):
        if type(other) == State:
            assert self.matching_graph == other.matching_graph and self.capacity == other.capacity
            _sum = State(values=self.data + other.data, matching_graph=self.matching_graph, capacity=self.capacity)
            if np.any(_sum.data > _sum.capacity):
                raise ValueError(
                    "The number of demand items and the number of supply items must be less than capacity.")
            return _sum
        return NotImplemented

    def __iadd__(self, other):
        if type(other) == State:
            assert self.matching_graph == other.matching_graph and self.capacity == other.capacity
            self.data += other.data
            if np.any(self.data > self.capacity):
                raise ValueError(
                    "The number of demand items and the number of supply items must be less than capacity.")
            return self
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matching):
            assert self.matching_graph == other.matching_graph
            return State(values=self.data - other.to_nodesdata(), matching_graph=self.matching_graph,
                         capacity=self.capacity)
        else:
            raise TypeError("Items from a State can only be substracted with a Matching")

    def __isub__(self, other):
        if isinstance(other, Matching):
            assert self.matching_graph == other.matching_graph
            self.data -= other.to_nodesdata()
            return self
        else:
            raise TypeError("Items from a State can only be substracted with a Matching")

    def __eq__(self, other):
        if type(other) == self.__class__:
            return np.all(self.data == other.data) and self.matching_graph == other.matching_graph and \
                   self.capacity == other.capacity
        return NotImplemented

    def copy(self):
        return State(values=self.data.copy(), matching_graph=self.matching_graph, capacity=self.capacity)


# We define a class Virtual State that acts as a State excepts that we allow negative values.
# This type of states are used in Stolyar policy
class Virtual_State(NodesData):

    def __init__(self, values, matchingGraph):
        # We use the NodesData initialization
        super(Virtual_State, self).__init__(values, matchingGraph)
        # We test that the sum of demand items is equal to the sum of supply items
        if self.demand(self.matching_graph.demand_class_set).sum() != self.supply(
                self.matching_graph.supply_class_set).sum():
            raise ValueError("The sum of demand items must be equal to the sum of supply items.")

    def __iadd__(self, other):
        if isinstance(other, State) or isinstance(other, Virtual_State):
            self.data += other.data
            return self
        else:
            raise TypeError("A Virtual_State can only be added with another Virtual_State or another State")

    def __sub__(self, other):
        if isinstance(other, Virtual_Matching):
            return Virtual_State(self.data - other.data, self.matching_graph)
        else:
            raise TypeError("Items from a State can only be substracted with a Virtual Matching")

    def __isub__(self, other):
        if isinstance(other, Virtual_Matching):
            self.data -= other.data
            return self
        else:
            raise TypeError("Items from a State can only be substracted with a Virtual Matching")


class EdgesData:
    """
    Stores data related to the edges of a MatchingGraph in an array.
    """

    # TODO: add assertion about equality of matching graph in operators

    def __init__(self, data: np.array, matching_graph: MatchingGraph):
        """
        :param data: Array which stores the data related to each edge. The value at a given index is related to the edge
            stored at the same index in the matching_graph.edges array.
        :param matching_graph: MatchingGraph to which this data is related to.
        """
        assert len(data) == matching_graph.nb_edges
        self.data = data
        self.matching_graph = matching_graph

    @classmethod
    def from_dict(cls, data: dict, matching_graph: MatchingGraph):
        """
        :param data: Dictionary which stores the data related to each edge. Each key should be an edge in the
            matching_graph.edges array. If an edge is not in the dictionary keys, then we put a default value of 0.
        :param matching_graph: MatchingGraph to which this data is related to.
        """
        data_array = np.zeros(matching_graph.nb_edges)
        for edge in data.keys():
            if matching_graph.isEdge(edge):
                data_array[matching_graph.edgeIndex(edge)] = data[edge]
            else:
                raise ValueError('A key from the dictionary does not correspond to an edge of the matching graph')
        return cls(data_array, matching_graph)

    @classmethod
    def zeros(cls, matching_graph: MatchingGraph):
        """ Creates an EdgeData with zeros for all edges.

        :param matching_graph: MatchingGraph to which this data is related to.
        """
        return cls(np.zeros(matching_graph.nb_edges), matching_graph)

    def __getitem__(self, edge: Tuple):
        if self.matching_graph.isEdge(edge):
            return self.data[self.matching_graph.edgeIndex(edge)]
        else:
            raise ValueError("The pair do not correspond to an edge in the matching graph")

    def __setitem__(self, edge: Tuple, value):
        if self.matching_graph.isEdge(edge):
            self.data[self.matching_graph.edgeIndex(edge)] = value
        else:
            raise ValueError("The pair do not correspond to an edge in the matching graph")

    def __eq__(self, other):
        if type(other) == self.__class__:
            return np.all(self.data == other.data) and self.matching_graph == other.matching_graph
        return NotImplemented

    def copy(self):
        return self.__class__(self.data.copy(), self.matching_graph)

    def __str__(self):
        return str(self.data)


# We create a Matching class which is a State with more restrictions.
# A matching can only add pairs of demand and supply items if they are associated to an edge in the matching graph.
# A matching has a reference to a State and can't have more items than the referenced State in any nodes.
class Matching(EdgesData):
    # TODO: add assertion about equality of matching graph in operators

    def __init__(self, state: State, values: np.array):
        """
        :param state: the State on which is performed the matching.
        :param values: Numpy Array which stores the number of matchings in each edge. The value at a given index is
            related to the edge stored at the same index in the State's MatchingGraph.
        """
        # We store a reference to the State on which we will perform matchings
        self.state = state
        # We use the EdgesData initialization
        super(Matching, self).__init__(values, state.matching_graph)
        # We test that the values for each edge is positive
        if (self.data < 0).any():
            raise ValueError("The number of matchings in each edge must be positive.")
        # We test that the number of items matched is lower than the number of items in the State
        if np.any(self.to_nodesdata() > state.data):
            raise ValueError(
                "The number of matched items can't be superior than the number of items in the State at any nodes")

    @classmethod
    def fromDict(cls, state: State, values: dict):
        """
        :param state: the State on which is performed the matching.
        :param values: Dictionary which stores the data related to each edge. Each key should be an edge in the
            matching_graph.edges array. If an edge is not in the dictionary keys, then we put a default value of 0.
        """
        values_array = np.zeros(state.matching_graph.nb_edges)
        for edge in values.keys():
            if state.matching_graph.isEdge(edge):
                values_array[state.matching_graph.edgeIndex(edge)] = values[edge]
            else:
                raise ValueError('A key from the dictionary does not correspond to an edge of the matching graph')
        return cls(state, values_array)

    @classmethod
    def zeros(cls, state: State):
        """
        :param state: the State on which is performed the matching.
        """
        # We create an empty state
        return cls(state, np.zeros(state.matching_graph.nb_edges))

    def to_nodesdata(self):
        return np.dot(self.matching_graph.edges_to_nodes, self.data)

    def __setitem__(self, edge: Tuple, value):
        super(Matching, self).__setitem__(edge, value)
        # We test if the number of matchings has exceed the number of items in the State
        if np.any(self.to_nodesdata() > self.state.data):
            raise ValueError(
                "The number of matched items can't be superior than the number of items in the State at any nodes")

    def __eq__(self, other):
        if type(other) == self.__class__:
            return np.all(self.data == other.data) and self.matching_graph == other.matching_graph and \
                   self.state == other.state
        return NotImplemented

    def copy(self):
        return Matching(self.state, self.data.copy())


# We define a Virtual Matching class which is the same as the Matching class excepts that we allow matchings to be made even if there is not enough items.
# This type of matching is used in Stolyar policy
class Virtual_Matching(State):

    def __init__(self, x, values):
        super(Virtual_Matching, self).__init__(values, x.matchingGraph)
        # We store a reference to the State on which we will perform matchings
        self.x = x
        # We test if the matching is feasible, i.e if it is a linear combination of edges from the matching graph
        if not self.feasible():
            raise ValueError("This matching is not feasible")

    def feasible(self):
        feasible_matching = True
        for subset in self.matching_graph.demand_class_subsets:
            if self.demand(np.array(subset)).sum() > self.supply(
                    np.array(self.matching_graph.demandToSupply[subset])).sum():
                feasible_matching = False
        for subset in self.matching_graph.supply_class_subsets:
            if self.supply(np.array(subset)).sum() > self.demand(
                    np.array(self.matching_graph.supplyToDemand[subset])).sum():
                feasible_matching = False
        return feasible_matching

    @classmethod
    def fromDict(cls, x, D):
        A = np.zeros(x.matchingGraph.n)
        # The values must be stored in a dictionnary D where the keys are the nodes
        for node in D.keys():
            if node not in x.matchingGraph.nodes:
                raise ValueError('A key from the dictionnary does not corespond to a node of the matching graph')
            elif node[0] == 'd':
                A[int(node[1]) - 1] = D[node]
            else:
                A[x.matchingGraph.nb_demand_classes + int(node[1]) - 1] = D[node]
        return cls(x, A)

    @classmethod
    def zeros(cls, x):
        # We create an empty state
        return cls(x, np.zeros(x.matchingGraph.n))

    @classmethod
    def items(cls, x, demand_items, supply_items):
        # We create a state by giving two separate array for demand and supply items
        return cls(x, np.hstack((demand_items, supply_items)))

    def __setitem__(self, index, value):
        # We test if the demand class i can be matched with the supply class j
        if self.matching_graph.isEdge(index):
            super(Virtual_Matching, self).__setitem__(index, value)
        else:
            raise ValueError("The pair do not correspond to an edge in the matching graph")

    def copy(self):
        return Virtual_Matching(self.x, self.data.copy())


class Model:

    def __init__(self, matching_graph: MatchingGraph, arrival_dist: NodesData, costs: NodesData, init_state: State,
                 state_space: str, init_arrival=None, discount=1., capacity=np.inf, penalty=0.):
        self.matching_graph = matching_graph
        # We initialize the class probabilities
        self.arrival_dist = arrival_dist
        # We stores the holding costs
        self.costs = costs
        # We initialise the discount factor
        self.discount = discount
        # We initialize the capacity of the system queues and the penalty for going beyond
        self.capacity = capacity
        self.penalty = penalty
        # We set up the functions and the initial state based on the type of state space
        assert state_space == "state" or state_space == "state_with_arrival"
        self.state_space = state_space
        if self.state_space == "state_with_arrival":
            if init_arrival is None:
                self.init_arrival = self.sample_arrivals()
            else:
                self.init_arrival = init_arrival
                assert self.matching_graph == init_arrival.matching_graph and self.capacity == init_arrival.capacity
        # We initialize the state of the system (the length of each queue)
        self.init_state = init_state
        # We assert that every NodesData has the same matching graph and that every State has the same capacity
        for nodes_data in [self.arrival_dist, self.costs, self.init_state]:
            assert self.matching_graph == nodes_data.matching_graph
            if type(nodes_data) == State:
                assert self.capacity == nodes_data.capacity

    def sample_arrivals(self):
        a = State.zeros(self.matching_graph, self.capacity)
        # We sample the class of the demand item
        d = np.random.choice(self.matching_graph.demand_class_set,
                             p=self.arrival_dist.demand(self.matching_graph.demand_class_set))
        # We sample the class of the supply item
        s = np.random.choice(self.matching_graph.supply_class_set,
                             p=self.arrival_dist.supply(self.matching_graph.supply_class_set))
        a[d, s] += 1
        return a

    def iterate_state(self, states_list, policies):
        costs_list = np.zeros(len(policies))
        # We sample new arrivals
        arrivals = self.sample_arrivals()
        for p, policy in enumerate(policies):
            # We compute costs
            costs_list[p] = np.dot(states_list[p].data, self.costs.data)
            # We apply the matchings
            states_list[p] -= policy.match(states_list[p])
            # We test if we get above capacity with new arrivals
            if np.any(states_list[p].data + arrivals.data > self.capacity):
                # If we do, we don't add the arrivals and induce a penalty
                costs_list[p] += self.penalty
            else:
                # If not, we add the arrivals and no penalty is induced
                states_list[p] += arrivals
        return states_list, costs_list

    def iterate_state_with_arrival(self, states_list: List[State], arrivals: State, policies):
        costs_list = np.zeros(len(policies))
        for p, policy in enumerate(policies):
            # We test if we get above capacity with new arrivals
            if np.any(states_list[p].data + arrivals.data > self.capacity):
                # If we do, we set the arrivals to zero and induce a penalty
                arrivals = State.zeros(matching_graph=self.matching_graph, capacity=self.capacity)
                costs_list[p] += self.penalty

            # We compute the matchings
            matchings = policy.match(state=states_list[p], arrivals=arrivals)
            # We add the arrivals
            states_list[p] += arrivals
            # We compute the costs
            costs_list[p] += np.dot(states_list[p].data, self.costs.data)
            # We apply the matchings
            states_list[p] -= matchings
        # We sample new arrivals
        arrivals = self.sample_arrivals()
        return states_list, arrivals, costs_list

    def run(self, nb_iter, policies, traj=False, plot=False):
        nb_policies = len(policies)
        # states_list stores the state of the system under each policy given by the list policies
        states_list = []
        costs_list = []
        # We initialize each state to the initial state of the model init_state and reset each policy
        for policy in policies:
            states_list.append(self.init_state.copy())
            policy.reset_policy(self.init_state.copy())
        arrivals = self.init_arrival.copy()

        if plot:
            traj = True
        if traj:
            # We keep the trajectory of the system under each policy
            state_size = self.matching_graph.n
            state_trajectories = np.zeros((nb_policies, state_size, nb_iter + 1))
            costs_trajectory = np.zeros((nb_policies, nb_iter + 1))
            state_trajectories[:, :, 0] = self.init_state.data
            if self.state_space == "state":
                for i in np.arange(nb_iter):
                    states_list, costs_list = self.iterate_state(states_list, policies)
                    state_trajectories[:, :, i + 1] = [state.data for state in states_list]
                    costs_trajectory[:, i + 1] = [costs for costs in costs_list]
            else:
                for i in np.arange(nb_iter):
                    states_list, arrivals, costs_list = self.iterate_state_with_arrival(states_list, arrivals,
                                                                                        policies)
                    state_trajectories[:, :, i + 1] = [state.data for state in states_list]
                    costs_trajectory[:, i + 1] = [costs for costs in costs_list]
            if plot:
                # plt.ion()
                # We plot the trajectories
                fig, axes = plt.subplots(nb_policies, 1, figsize=(15, nb_policies * 5), squeeze=0)
                for p, policy in enumerate(policies):
                    for e in np.arange(state_size):
                        lab = "d_" + str(e + 1) if e < self.matching_graph.nb_demand_classes else "s_" + str(
                            e - self.matching_graph.nb_demand_classes + 1)
                        axes[p, 0].plot(state_trajectories[p, e, :], label=lab)
                    axes[p, 0].legend(loc='best')
                    axes[p, 0].set_title(str(policy))
                fig.canvas.draw()
                plt.pause(0.1)
                fig.canvas.flush_events()
            return state_trajectories, costs_trajectory
        else:
            if self.state_space == "state":
                for _ in np.arange(nb_iter):
                    states_list, costs_list = self.iterate_state(states_list, policies)
            else:
                for _ in np.arange(nb_iter):
                    states_list, arrivals, costs_list = self.iterate_state_with_arrival(states_list, arrivals,
                                                                                        policies)
            return states_list, costs_list

    def average_cost(self, nb_iter, policies, plot=False):
        x_traj, costs_traj = self.run(nb_iter, policies, traj=True)
        if plot:
            # plt.ion()
            # We plot the costs trajectory
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            markers = ['.', 'o', '^', 'x', 's', 'D']
            for p, policy in enumerate(policies):
                ax.plot(np.cumsum(costs_traj[p]) / np.arange(1, nb_iter + 2), marker=markers[p], label=str(policy),
                        markevery=int(nb_iter / 10.))
            ax.legend(loc='best')
            ax.set_ylabel('Average cost')
            fig.canvas.draw()
            plt.pause(0.1)
            fig.canvas.flush_events()
        return costs_traj, x_traj

    def discounted_cost(self, nb_iter, policies, plot=False):
        x_traj, costs_traj = self.run(nb_iter, policies, traj=True)
        if plot:
            # plt.ion()
            # We plot the costs trajectory
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            linestyles = ['-', '--', '-^', ':']
            for p, policy in enumerate(policies):
                discounted_cost = np.cumsum(np.multiply(costs_traj[p], np.power(self.discount, np.arange(nb_iter + 1))))
                ax.plot(discounted_cost, linestyles[p], label=str(policy), markevery=int(nb_iter / 10.))
            ax.legend(loc='best')
            ax.set_ylabel('Discounted cost')
            fig.canvas.draw()
            plt.pause(0.1)
            fig.canvas.flush_events()
        return costs_traj, x_traj

    def __eq__(self, other):
        if type(other) == Model:
            return self.matching_graph == other.matching_graph and self.arrival_dist == other.arrival_dist and \
                   self.costs == other.costs and self.discount == other.discount and self.capacity == other.capacity \
                   and self.penalty == other.penalty and self.state_space == other.state_space and \
                   self.init_state == other.init_state and self.init_arrival == other.init_arrival
        return NotImplemented
        

