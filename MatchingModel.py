import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, chain, product


class MatchingGraph:
    
    def __init__(self, edges, nb_demand_classes, nb_supply_classes):
        # Edges must be a list of tuples ('i','j') if demand class i can be matched with supply class j
        self.edges = edges
        self.nb_demand_classes = nb_demand_classes
        self.nb_supply_classes = nb_supply_classes
        # We compute the set and all subsets of demand classes
        self.demand_class_set = np.arange(1,self.nb_demand_classes+1)
        self.demand_class_subsets = [tuple(c) for c in chain.from_iterable(combinations(self.demand_class_set, r) for r in self.demand_class_set)]
        # We create a dictionary which maps each subset of demand classes to the subset of supply classes to witch its linked
        self.build_demandToSupply()
        # We compute the set and all subsets of supply classes
        self.supply_class_set = np.arange(1,nb_supply_classes+1)
        self.supply_class_subsets = [tuple(c) for c in chain.from_iterable(combinations(self.supply_class_set, r) for r in np.arange(1,nb_supply_classes+1))]
        # We create a dictionary which maps each subset of supply classes to the subset of demand classes to witch its linked
        self.build_supplyToDemand()
        
    @property
    def n(self):
        return self.nb_demand_classes + self.nb_supply_classes
    
    @property
    def nodes(self):
        # We create a list of all nodes with first the demand classes and then the supply classes, both in increasing order
        return np.array(['d'+str(i) for i in self.demand_class_set] + ['s'+str(j) for j in self.supply_class_set])
        
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
            raise ValueError('This value does not correspond to an egde of the matching graph')
        
    def degree(self):
        # We count the degree of each node
        d = NodesData(np.zeros(self.n),self)
        for edge in self.edges:
            d[edge] += 1
        return d
    
    def maximal_matchings(self):
        # We compute all the maximal matchings of the matching graph. This function only makes sense if it called on the MatchingGraph returned by available_matchings_subgraph()
        list_maximal_matchings = []
        deg = self.degree().data
        # We look at all nodes of degree superior than 2 (there can not be more than 2)
        if (deg<2).all():
            # If the degree of each node is less than two then all the edges of the matching graph form the only maximal matching
            list_maximal_matchings.append(self.edges)
        elif np.sum(deg>=2)==1:
            # We get the index of the node
            node_index = int(np.where(deg>=2)[0])
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
                        list_maximal_matchings.append([(node, supply_class),remaining_edge[0]])
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
                        list_maximal_matchings.append([(demand_class, node),remaining_edge[0]])
                else:
                    for demand_class in D_j:
                        list_maximal_matchings.append([(demand_class, node)])
        else:       
            # We get the demand class and the supply class of the arrivals by selecting the only two nodes that have a degree greater than 2
            arrivals_classes_index = np.where(deg>=2)[0]
            arrivals_classes = (arrivals_classes_index[0]+1, arrivals_classes_index[1]-self.nb_demand_classes+1)
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
            for classes in product(S_i,D_j):
                list_maximal_matchings.append([(arrivals_classes[0],classes[0]),(classes[1],arrivals_classes[1])])
            
        return list_maximal_matchings
    
# We define a class NodesData which is a data structure for our system. 
# It stores a value for each classes of demand and supply items.
# It is used for example to store the length of the queues, the holding costs or the arrival rates
class NodesData:
        
    def __init__(self, A, matchingGraph):
        # The values must be stored, in a Numpy Array, organized as such: first the demand items, then the supply items and both sorted by classes in increasing order
        # This means that index i represent demand class i+1 and index nb_demand_classes+j represent supply class j+1
        self.data = A
        self.matchingGraph = matchingGraph
      
    @classmethod
    def fromDict(cls, D, matchingGraph):
        A = np.zeros(matchingGraph.n)
        # The values must be stored in a dictionnary D where the keys are the nodes 
        for node in D.keys():
            if node not in matchingGraph.nodes:
                raise ValueError('A key from the dictionnary does not corespond to a node of the matching graph')
            elif node[0]=='d':
                A[int(node[1])-1] = D[node]
            else:
                A[matchingGraph.nb_demand_classes+int(node[1])-1] = D[node]
        return cls(A,matchingGraph)
        
    @classmethod
    def zeros(cls, matchingGraph):
        # We create an empty state
        return cls(np.zeros(matchingGraph.n), matchingGraph)
    
    @classmethod
    def items(cls, demand_items, supply_items, matchingGraph):
        # We create a state by giving two separate array for demand and supply items
        return cls(np.hstack((demand_items,supply_items)), matchingGraph)
    
    def demand(self, classes):
        return self.data[classes-1]
    
    def supply(self, classes):
        return self.data[classes-1+self.matchingGraph.nb_demand_classes]
    
    def __getitem__(self, index):
        i, j = index
        return self.data[[i-1, self.matchingGraph.nb_demand_classes+j-1]]
    
    def __setitem__(self, index, value):
        i, j = index
        self.data[[i-1, self.matchingGraph.nb_demand_classes+j-1]] = value
    
    def __add__(self, other):
        return self.__class__(self.data + other.data, self.matchingGraph)
    
    def __iadd__(self, other):
        self.data += other.data
        return self   
    
    def copy(self):
        return self.__class__(self.data.copy(),self.matchingGraph)

# We define a class State which is a NodesData with the constraint that demand and supply items must be positives and their sum equal
# It is used for example to store the length of the queues, arrival items or matchings
class State(NodesData):
    
    def __init__(self, values, matchingGraph):
        # We use the NodesData initialization
        super(State,self).__init__(values, matchingGraph)     
        # We test that the number of demand items and the number of supply items are positives
        if (self.data < 0).any():
            raise ValueError("The number of demand items and the number of supply items must be positives.")
        # We test that the sum of demand items is equal to the sum of supply items
        if self.demand(self.matchingGraph.demand_class_set).sum() != self.supply(self.matchingGraph.supply_class_set).sum():
            raise ValueError("The sum of demand items must be equal to the sum of supply items.")
        
    def matchings_available(self):
        # We construct a list of all the edges which can be matched given the State
        list_edges = []
        for edge in self.matchingGraph.edges:
            if (self[edge]>=1).all():
                list_edges.append(edge)
        return list_edges
    
    def matchings_available_subgraph(self):
        # We construct a subgraph composed of all the edges which can be matched given the State
        return MatchingGraph(self.matchings_available(), self.matchingGraph.nb_demand_classes, self.matchingGraph.nb_supply_classes)
    
    def __iadd__(self, other):
        if isinstance(other, State):
            self.data += other.data
            return self
        else:
            raise TypeError("A State can only be added with another State")
    
    def __sub__(self, other):
        if isinstance(other, Matching):
            return State(self.data - other.data, self.matchingGraph) 
        else:
            raise TypeError("Items from a State can only be substracted with a Matching")
        
    def __isub__(self, other):
        if isinstance(other, Matching):
            self.data -= other.data
            return self 
        else:
            raise TypeError("Items from a State can only be substracted with a Matching")
    
# We define a class Virtual State that acts as a State excepts that we allow negative values. 
# This type of states are used in Stolyar policy
class Virtual_State(NodesData):
    
    def __init__(self, values, matchingGraph):
        # We use the NodesData initialization
        super(Virtual_State,self).__init__(values, matchingGraph)
        # We test that the sum of demand items is equal to the sum of supply items
        if self.demand(self.matchingGraph.demand_class_set).sum() != self.supply(self.matchingGraph.supply_class_set).sum():
            raise ValueError("The sum of demand items must be equal to the sum of supply items.")
        
    def __iadd__(self, other):
        if isinstance(other, State) or isinstance(other, Virtual_State):
            self.data += other.data
            return self
        else:
            raise TypeError("A Virtual_State can only be added with another Virtual_State or another State")
    
    def __sub__(self, other):
        if isinstance(other, Virtual_Matching):
            return Virtual_State(self.data - other.data, self.matchingGraph) 
        else:
            raise TypeError("Items from a State can only be substracted with a Virtual Matching")
        
    def __isub__(self, other):
        if isinstance(other, Virtual_Matching):
            self.data -= other.data
            return self 
        else:
            raise TypeError("Items from a State can only be substracted with a Virtual Matching")
            
    
# We create a Matching class which is a State with more restrictions.
# A matching can only add pairs of demand and supply items if they are associated to an edge in the matching graph.
# A matching has a reference to a State and can't have more items than the referenced State in any nodes.
class Matching(State):
    
    def __init__(self, x, values):
        super(Matching,self).__init__(values, x.matchingGraph)
        # We store a reference to the State on which we will perform matchings
        self.x = x
        # We test if the number of matchings is higher than the number of items in the State
        if (self.data > x.data).any():
            raise ValueError("The number of matched items can't be superior than the number of items in the State at any nodes")
        if not self.feasible():
            raise ValueError("This matching is not feasible")
    
    def feasible(self):
        feasible_matching = True
        for subset in self.matchingGraph.demand_class_subsets:
            if self.demand(np.array(subset)).sum() > self.supply(np.array(self.matchingGraph.demandToSupply[subset])).sum():
                feasible_matching = False
        for subset in self.matchingGraph.supply_class_subsets:
            if self.supply(np.array(subset)).sum() > self.demand(np.array(self.matchingGraph.supplyToDemand[subset])).sum():
                feasible_matching = False
        return feasible_matching
    
    @classmethod
    def fromDict(cls, x, D):
        A = np.zeros(x.matchingGraph.n)
        # The values must be stored in a dictionnary D where the keys are the nodes 
        for node in D.keys():
            if node not in x.matchingGraph.nodes:
                raise ValueError('A key from the dictionnary does not corespond to a node of the matching graph')
            elif node[0]=='d':
                A[int(node[1])-1] = D[node]
            else:
                A[x.matchingGraph.nb_demand_classes+int(node[1])-1] = D[node]
        return cls(x, A)
        
    @classmethod
    def zeros(cls, x):
        # We create an empty state
        return cls(x, np.zeros(x.matchingGraph.n))
    
    @classmethod
    def items(cls, x, demand_items, supply_items):
        # We create a state by giving two separate array for demand and supply items
        return cls(x, np.hstack((demand_items,supply_items)))
    
    def __setitem__(self, index, value):
        # We test if the demand class i can be matched with the supply class j
        if self.matchingGraph.isEdge(index):
            # We test if the number of matchings has exceed the number of items in the State
            if (value>self.x[index]).any():
                raise ValueError("The number of matched items can't be superior than the number of items in the State at any nodes")
            else:
                super(Matching,self).__setitem__(index,value)
        else:
            raise ValueError("The pair do not correspond to an edge in the matching graph")
            
    def copy(self):
        return Matching(self.x,self.data.copy())
            
# We define a Virtual Matching class which is the same as the Matching class excepts that we allow matchings to be made even if there is not enough items.
# This type of matching is used in Stolyar policy
class Virtual_Matching(State):
    
    def __init__(self, x, values):
        super(Virtual_Matching,self).__init__(values, x.matchingGraph)
        # We store a reference to the State on which we will perform matchings
        self.x = x
        # We test if the matching is feasible, i.e if it is a linear combination of edges from the matching graph
        if not self.feasible():
            raise ValueError("This matching is not feasible")
    
    def feasible(self):
        feasible_matching = True
        for subset in self.matchingGraph.demand_class_subsets:
            if self.demand(np.array(subset)).sum() > self.supply(np.array(self.matchingGraph.demandToSupply[subset])).sum():
                feasible_matching = False
        for subset in self.matchingGraph.supply_class_subsets:
            if self.supply(np.array(subset)).sum() > self.demand(np.array(self.matchingGraph.supplyToDemand[subset])).sum():
                feasible_matching = False
        return feasible_matching
    
    @classmethod
    def fromDict(cls, x, D):
        A = np.zeros(x.matchingGraph.n)
        # The values must be stored in a dictionnary D where the keys are the nodes 
        for node in D.keys():
            if node not in x.matchingGraph.nodes:
                raise ValueError('A key from the dictionnary does not corespond to a node of the matching graph')
            elif node[0]=='d':
                A[int(node[1])-1] = D[node]
            else:
                A[x.matchingGraph.nb_demand_classes+int(node[1])-1] = D[node]
        return cls(x, A)
        
    @classmethod
    def zeros(cls, x):
        # We create an empty state
        return cls(x, np.zeros(x.matchingGraph.n))
    
    @classmethod
    def items(cls, x, demand_items, supply_items):
        # We create a state by giving two separate array for demand and supply items
        return cls(x, np.hstack((demand_items,supply_items)))
    
    def __setitem__(self, index, value):
        # We test if the demand class i can be matched with the supply class j
        if self.matchingGraph.isEdge(index):
            super(Virtual_Matching,self).__setitem__(index,value)
        else:
            raise ValueError("The pair do not correspond to an edge in the matching graph")
            
    def copy(self):
        return Virtual_Matching(self.x,self.data.copy())
        
          
class Model:
    
    def __init__(self, matchingGraph, arrival_dist, costs, x_0):
        self.matchingGraph = matchingGraph
        # We initialize the class probabilities
        self.arrival_dist = arrival_dist
        # We stores the holding costs
        self.costs = costs
        # We initialize the state of the system (the length of each queue)
        self.x_0 = x_0
        
    def sample_arrivals(self):
        a = State.zeros(self.matchingGraph)
        # We sample the class of the demand item
        d = np.random.choice(self.matchingGraph.demand_class_set,p=self.arrival_dist.demand(self.matchingGraph.demand_class_set))
        # We sample the class of the supply item
        s = np.random.choice(self.matchingGraph.supply_class_set,p=self.arrival_dist.supply(self.matchingGraph.supply_class_set))
        a[d,s]+=1
        return a
    
    def iterate(self,states_list,policies):
        # We sample new arrivals
        arrivals = self.sample_arrivals()
        for p, policy in enumerate(policies):
            # We apply the matchings
            states_list[p] -= policy.match(states_list[p]) 
            # We add the arrivals
            states_list[p] += arrivals
        #return states_list
        
    def run(self, nb_iter, policies, traj=False, plot=False):
        nb_policies = len(policies)
        # states_list stores the state of the system under each policy given by the list policies
        states_list = []
        # We intialize each state to the initial state of the model x_0 and reset each policy
        for policy in policies:
            states_list.append(self.x_0.copy())
            policy.reset_policy(self.x_0)
            
        if plot:
            traj = True
        if traj:
            # We keep the trajectory of the system under each policy
            state_size = self.matchingGraph.n
            trajectories = np.zeros((nb_policies,state_size,nb_iter+1))
            trajectories[:,:,0] = self.x_0.data
            for i in np.arange(nb_iter):
                self.iterate(states_list,policies) 
                trajectories[:,:,i+1] = [state.data for state in states_list]
            
            if plot:
                # We plot the trajectories
                fig, axes = plt.subplots(nb_policies,1,figsize=(15,nb_policies*5),squeeze=0)
                for p, policy in enumerate(policies):
                    for e in np.arange(state_size):
                        lab = "d_"+str(e+1) if e<self.matchingGraph.nb_demand_classes else "s_"+str(e-self.matchingGraph.nb_demand_classes+1)
                        axes[p,0].plot(trajectories[p,e,:],label=lab)
                    axes[p,0].legend(loc='best')
                    axes[p,0].set_title(str(policy))
            return trajectories
        else:
            for _ in np.arange(nb_iter):
                self.iterate(states_list,policies)
            return states_list
        
    def average_cost(self, nb_iter, policies, plot=False):
        x_traj = self.run(nb_iter, policies, traj=True)
        costs_traj = [np.cumsum(np.dot(self.costs.data.reshape(1,-1),x_traj[i,:,:]))/np.arange(1.,nb_iter+2) for i in np.arange(len(policies))]
        if plot:
            # We plot the costs trajectory
            plt.figure(figsize=(15,5))
            linestyles = ['-', '--', '-^', ':']
            for p, policy in enumerate(policies):
                plt.plot(costs_traj[p],linestyles[p],label=str(policy),markevery=int(nb_iter/10.))
            plt.legend(loc='best')
            plt.ylabel('Average cost')
        return costs_traj, x_traj
        
