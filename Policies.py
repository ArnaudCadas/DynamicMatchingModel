import numpy as np
import mipcl_py.mipshell.mipshell as mip
from MatchingModel import *
import torch
from ddpg import Agent

# We define a class for policies
class Policy:

    def match(self,x):
        raise NotImplementedError
        
    def reset_policy(self, x_0):
        pass
    
    def __str__(self):
        pass

#######################
### Noise Metaclass ###
#######################

def add_noise_params(policy_init):
    def policy_noisy_init(self, *args, **kwargs):
        self.noise_params = kwargs.pop('noise_params')
        policy_init(self, *args, **kwargs)
    return policy_noisy_init

def add_noise(policy_match):
    def policy_noisy_match(self, x):
        u = policy_match(self, x)
        for i, edge in enumerate(x.matchingGraph.edges):
            u[edge] -= np.minimum(np.random.geometric(self.noise_params[i]),u[edge].min())
        return u
    return policy_noisy_match

class Noise_Geom(type):
    def __new__(cls, name, bases, attr):
        attr['__init__'] = add_noise_params(bases[0].__init__)
        attr['match'] = add_noise(bases[0].match)
        return type.__new__(cls, name, bases, attr)
    
# We define various policies by creating child class from Policy and implementing the function match()

# We define a random policy which choose a random possible (depending on the State) matching. 
# The policy has a parameter that gives the maximum number of times we repeat the last operation.
class Random_policy(Policy):
    
    def __init__(self,nb_matchings_max=np.inf):
        assert nb_matchings_max==np.inf or (type(nb_matchings_max)==int and nb_matchings_max>=1)
        self.nb_matchings_max = nb_matchings_max
    
    def match(self,x):
        nb_matchings = 0
        u = Matching.zeros(x)
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

class RL_policy(Policy):
    
    def __init__(self):
        # We initialise the agent
        self.agent = Agent(state_size=4, action_size=3, random_seed=2)
        # We set the weights to the trained model
        self.agent.actor_local.load_state_dict(torch.load('trained_actor.pth'))
        self.agent.critic_local.load_state_dict(torch.load('trained_critic.pth'))
        
    def match(self,x):
        action = self.agent.act(x.data.copy())

        # We transform the action in a matching
        u = Matching.zeros(x)
        tmp_x = x - u
        for matching_edge, nb_matches in zip(x.matchingGraph.edges,action):
            u[matching_edge] += np.minimum(np.floor(nb_matches), tmp_x[matching_edge].min())
            tmp_x = x - u
        return u
    
    def __str__(self):
        return 'RL policy'

class Threshold_policy_N(Policy):
    
    def __init__(self,threshold):
        # We store the threshold
        self.threshold = threshold
        
    def match(self,x):
        u = Matching.zeros(x)
        # We match all l1
        u[1,1] += x[1,1].min()
        # We match all l2
        u[2,2] += x[2,2].min()
        # We update the state with the matchings in l1 and l2 because they have priority and they influence the ones in l3
        #print('u after match l1 and l2',u.data)
        new_state = x - u
        #print('new_state after match l1 and l2',new_state.data)
        # We match all l3 above the threshold
        l3_matchings = np.maximum(new_state[1,2].min() - self.threshold, 0.)
        #print('threshold',self.threshold)
        #print('l3 remaining',new_state[1,2].min())
        #print('l3 remaining minus threshold',new_state[1,2].min() - self.threshold)
        #print('l3 matchings', l3_matchings)
        u[1,2] += l3_matchings
        #print('u after match l3',u.data)
        return u
    
    def __str__(self):
        return 'Threshold policy t={}'.format(self.threshold)

    def __call__(self,x):
        return self.match(x)

    def grad(self,x):
        g = np.zeros(4)
        if np.minimum(x.demand(1)-x[1,1].min(),x.supply(2)-x[2,2].min()) - self.threshold > 0.:
            g[0] = -1.
            g[3] = -1.
        return g.reshape(1,-1)

    @property
    def theta(self):
        return np.array([[self.threshold]])

    @theta.setter
    def theta(self, value):
        print('value after update', value)
        self.threshold = np.floor(np.maximum(value.item(),0.))
        print('new threshold after projection', self.threshold)

class Threshold_policy_N_noisy(Threshold_policy_N, metaclass=Noise_Geom):
    pass
    
class Threshold_policy(Policy):
    
    def __init__(self,thresholds):
        # We store the thresholds as [s_1, s_2]
        self.thresholds = thresholds
        
    def match(self,x):
        u = Matching.zeros(x)
        # We match all l4
        u[3,2] += x[3,2].min()
        # We match all l1
        u[1,1] += x[1,1].min()
        # We update the state with the matchings in l4 and l1 because they have priority and they influence the ones in l2 and l3
        new_state = x - u
        # We match all l2 above the s_1 threshold
        l2_matchings = np.maximum(new_state[2,1].min() - self.thresholds[0], 0.)
        u[2,1] += l2_matchings
        # We match all l3 above the s_2 threshold
        l3_matchings = np.maximum(new_state[2,2].min() - self.thresholds[1], 0.)
        u[2,2] += l3_matchings
        return u
    
    def __str__(self):
        return 'Threshold policy t={}'.format(self.thresholds)

class TwP_policy(Policy):
    
    def __init__(self,thresholds):
        # We store the thresholds as [d_1, ds_2, s_3]
        self.thresholds = thresholds
        
    def match(self,x):
        u = Matching.zeros(x)
        # We match all (1,1) and all (3,3)
        u[1,1] += x[1,1].min()
        u[3,3] += x[3,3].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (1,2) above the threshold in d_1 and we match all (2,3) above the threshold in s_3
        u[1,2] += np.minimum(np.maximum(new_state.demand(1) - self.thresholds[0], 0.), new_state.supply(2))
        u[2,3] += np.minimum(np.maximum(new_state.supply(3) - self.thresholds[2], 0.), new_state.demand(2))
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (2,2) above the threshold
        u[2,2] += np.maximum(new_state[2,2].min() - self.thresholds[1], 0.)
        
        return u
    
    def __str__(self):
        #return 'TwP policy t={}'.format(self.thresholds)
        return 'Thresholds with Priority policy'
    
class TwPbis_policy(Policy):
    
    def __init__(self,thresholds):
        # We store the thresholds as [d_1, ds_2, s_3]
        self.thresholds = thresholds
        
    def match(self,x):
        u = Matching.zeros(x)
        # We match all (1,1) and all (3,3)
        u[1,1] += x[1,1].min()
        u[3,3] += x[3,3].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (1,2) above the threshold in d_1 and we match all (2,3) above the threshold in s_3
        u[1,2] += np.maximum(new_state[1,2].min() - self.thresholds[0], 0.)
        u[2,3] += np.maximum(new_state[2,3].min() - self.thresholds[2], 0.)
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We match all (2,2) above the threshold
        u[2,2] += np.maximum(new_state[2,2].min() - self.thresholds[1], 0.)
        
        return u
    
    def __str__(self):
        #return 'TwP policy t={}'.format(self.thresholds)
        return 'Thresholds with Priority policy'

class TwMW_policy(Policy):
    
    def __init__(self,thresholds,costs):
        # We store the thresholds as [d_1, s_3]
        self.thresholds = thresholds
        self.costs = costs
        
    def match(self,x):
        u = Matching.zeros(x)
        # We match all (1,1) and all (3,3)
        u[1,1] += x[1,1].min()
        u[3,3] += x[3,3].min()
        # We update the state with the matchings because they have priority and they influence the future ones
        new_state = x - u
        # We remove in d_1 and s_3
        uprime = Matching.zeros(x)
        uprime[1,2] = np.minimum(new_state[1,2].min(),self.thresholds[0])
        uprime[2,3] = np.minimum(new_state[2,3].min(),self.thresholds[1])
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
    
    def match(self,x):
        # We suppose that x is a stable state (i.e, before the arrivals, no more matching could have been done)
        u = Matching.zeros(x)
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
    
class Stolyar_policy(Policy):
    
    def __init__(self, x_0, rewards, beta, costs=None):
        self.previous_x = x_0.copy()
        self.previous_match = Matching.zeros(x_0)
        self.virtual_x = Virtual_State(x_0.data.copy(), x_0.matchingGraph)
        self.rewards = rewards
        self.beta = beta
        self.incomplete_matchings = []
        # We can include holding costs in Stolyar algorithm by giving the costs as parameter and changing the virtual system update
        if costs is not None:
            self.costs = costs
            self.update_vs = self.update_virtual_system_wCosts
        else:
            self.update_vs = self.update_virtual_system
    
    def match(self,x):
        # update the virtual system: add previous arrivals, apply algo 1 and add the matching to the incomplete queue
        self.update_vs(x)
        # We scan the queue in FCFS order until we find a feasible match given x
        # We return the feasible if one was found or we return 0.
        for match in self.incomplete_matchings:
            if (match.data <= x.data).all():
                u = Matching(x, match.data)
                self.incomplete_matchings.remove(match)
                self.previous_match = u
                return u
        u = Matching.zeros(x)
        self.previous_match = u
        return u
    
    def update_virtual_system(self,x):
        # We add the previous arrivals to the virtual state
        arrivals = State(x.data - self.previous_x.data + self.previous_match.data, x.matchingGraph)
        self.virtual_x += arrivals
        self.previous_x = x.copy()
        # We use Stolyar algorithm to get the matching based on the virtual system
        virtual_match = Virtual_Matching.zeros(x)
        matchings_values = np.zeros(len(self.rewards))
        for i, edge in enumerate(x.matchingGraph.edges):
            matchings_values[i] = self.rewards[i] + self.beta*np.sum(self.virtual_x[edge])
        virtual_match[x.matchingGraph.edges[np.argmax(matchings_values)]]+=1.
        # We perform the matching in the virtual system
        self.virtual_x -= virtual_match
        # We add the matching to the list of incomplete matchings
        self.incomplete_matchings.append(virtual_match)
        
    def update_virtual_system_wCosts(self,x):
        # We add the previous arrivals to the virtual state
        arrivals = State(x.data - self.previous_x.data + self.previous_match.data, x.matchingGraph)
        self.virtual_x += arrivals
        self.previous_x = x.copy()
        # We use Stolyar algorithm to get the matching based on the virtual system
        virtual_match = Virtual_Matching.zeros(x)
        matchings_values = np.zeros(len(self.rewards))
        for i, edge in enumerate(x.matchingGraph.edges):
            matchings_values[i] = self.rewards[i] + self.beta*np.dot(self.costs[edge].reshape(1,-1),self.virtual_x[edge].reshape(-1,1))
        virtual_match[x.matchingGraph.edges[np.argmax(matchings_values)]]+=1.
        # We perform the matching in the virtual system
        self.virtual_x -= virtual_match
        # We add the matching to the list of incomplete matchings
        self.incomplete_matchings.append(virtual_match)
        
    def reset_policy(self,x_0):
        # We reset the previous state, matching and virtual state
        self.previous_x = x_0.copy()
        self.previous_match = Matching.zeros(x_0)
        self.virtual_x = Virtual_State(x_0.data.copy(), x_0.matchingGraph)
        # We empty the list of incomplete matchings
        self.incomplete_matchings = []
        
    def __str__(self):
        if hasattr(self, 'costs'):
            return 'Stolyar policy with costs r={}, b={}'.format(self.rewards,self.beta)
        else:
            return 'Stolyar policy r={}, b={}'.format(self.rewards,self.beta)
        
class hMWT_policy(Policy):
    
    def __init__(self, matchingGraph, Workload_index, alpha, costs, beta, kappa, theta, delta_plus, NUmax):
        # The Workload_index must be a list with two element. 
        # The first one is 'd' if we are looking at demand classes or 's' for supply classes
        # The second one is a tuple with the classes
        assert Workload_index[0] == 'd' or Workload_index[0] == 's'
        self.Workload_index = Workload_index
        self.NUmax = NUmax # The maximal number of matchings that can be done at once
        self.alpha = alpha # The mean arrival rate
        self.costs = costs # The linear costs
        self.beta = beta # The perturbation parameter to turn x (or w) in xtil (or wtil)
        self.kappa = kappa
        self.theta = theta
        self.delta_plus = delta_plus
        
        if self.Workload_index[0] == 'd':
            D = self.Workload_index[1]
            S_D = tuple(matchingGraph.demandToSupply[D])
            self.Idle_index = [matchingGraph.edgeIndex((d,s)) for s in S_D for d in matchingGraph.supplyToDemand[(s,)] if d not in D]
            self.XiD = np.array([1. if idx in D else 0. for idx in matchingGraph.demand_class_set]+[-1. if idx in S_D else 0. for idx in matchingGraph.supply_class_set])
            
            # We compute the optimal threshold tau_star
            Delta_plus = self.alpha.demand(np.array(D)).sum()
            Delta_minus = self.alpha.supply(np.array(S_D)).sum()
            delta = Delta_minus - Delta_plus
            self.barC_plus = self.costs.demand(np.array(D)).min() + self.costs.supply(np.array(matchingGraph.Scomplement(S_D))).min()
            self.barC_minus = self.costs.demand(np.array(matchingGraph.Dcomplement(D))).min() + self.costs.supply(np.array(S_D)).min()
            #sigmaSq_delta = (delta + 1.)**2 * Delta_plus*(1.-Delta_minus) + (delta - 1.)**2 * (1.-Delta_plus)*Delta_minus # Ana code version
            sigmaSq_delta = Delta_plus*(1.-Delta_minus) + (1.-Delta_plus)*Delta_minus - delta**2 # My version
            self.tau_star = 0.5*(sigmaSq_delta/delta)*np.log(1.+(self.barC_plus/self.barC_minus))
            
            # We compute coefficients needed for hat_h
            hat_etaSS = self.tau_star*self.barC_minus
            self.Aplus = self.barC_plus/(2*delta)
            self.Bplus = (sigmaSq_delta*self.Aplus-hat_etaSS)/delta
            self.THETA = 2*delta/sigmaSq_delta
            self.Aminus = -self.barC_minus/(self.THETA*sigmaSq_delta)
            self.Bminus = 2*self.Aminus/self.THETA - hat_etaSS/delta
            self.Dminus = (self.Bplus-self.Bminus)/self.THETA
            
            self.match = self.match_D
        else:
            S = self.Workload_index[1]
            D_S = tuple(matchingGraph.supplyToDemand[S])
            self.Idle_index = [matchingGraph.edgeIndex((d,s)) for d in D_S for s in matchingGraph.demandToSupply[(d,)] if s not in S]
            self.XiS = np.array([-1. if idx in D_S else 0. for idx in matchingGraph.demand_class_set]+[1. if idx in S else 0. for idx in matchingGraph.supply_class_set])
        
            # We compute the optimal threshold tau_star
            Delta_plus = self.alpha.supply(np.array(S)).sum()
            Delta_minus = self.alpha.demand(np.array(D_S)).sum()
            delta = Delta_minus - Delta_plus
            self.barC_plus = self.costs.supply(np.array(S)).min() + self.costs.demand(np.array(matchingGraph.Dcomplement(D_S))).min()
            self.barC_minus = self.costs.supply(np.array(matchingGraph.Scomplement(S))).min() + self.costs.demand(np.array(D_S)).min()
            #sigmaSq_delta = (delta + 1.)**2 * Delta_plus*(1.-Delta_minus) + (delta - 1.)**2 * (1.-Delta_plus)*Delta_minus # Ana code version
            sigmaSq_delta = Delta_plus*(1.-Delta_minus) + (1.-Delta_plus)*Delta_minus - delta**2 # My version
            self.tau_star = 0.5*(sigmaSq_delta/delta)*np.log(1.+(self.barC_plus/self.barC_minus))
            
            # We compute coefficients need for hat_h
            hat_etaSS = self.tau_star*self.barC_minus
            self.Aplus = self.barC_plus/(2*delta)
            self.Bplus = (sigmaSq_delta*self.Aplus-hat_etaSS)/delta
            self.THETA = 2*delta/sigmaSq_delta
            self.Aminus = -self.barC_minus/(self.THETA*sigmaSq_delta)
            self.Bminus = 2*self.Aminus/self.THETA - hat_etaSS/delta
            self.Dminus = (self.Bplus-self.Bminus)/self.THETA
        
            self.match = self.match_S
    
    def match_D(self, x):
        # We compute the workload w
        w = np.inner(x.data, self.XiD)
        # We computed the pertubated states and workload
        xtil = x.data + self.beta*(np.exp(-x.data/self.beta)-1.)
        wtil = np.sign(w)*(np.abs(w) + self.beta*(np.exp(-np.abs(w)/self.beta)-1.))
        
        # The function h is the sum of two function: \hat{h}(w) et h_c(x)
        # We compute the gradient of h_c(x)
        grad_ctil = np.multiply(self.costs.data,(1. - np.exp(-x.data/self.beta)))
        #print('grad_ctil',grad_ctil)
        if w>=0:
            grad_barCtil = self.barC_plus*(1.-np.exp(-w/self.beta))*self.XiD
        else:
            grad_barCtil = -self.barC_minus*(1.-np.exp(w/self.beta))*self.XiD
        #print('xtil',xtil)
        #print('cxtil',np.inner(self.costs.data, xtil))
        grad_h_c = 2.*self.kappa*(np.inner(self.costs.data, xtil) - np.maximum(self.barC_plus*wtil, -self.barC_minus*wtil) )*(grad_ctil - grad_barCtil)
        # We compute the derivative of \hat{h}(w)
        if w>=0:
            hat_hprime = 2*self.Aplus*w+self.Bplus
        elif w<0 and w>=-self.tau_star:
            #hat_hprime = 2*self.Aminus + self.Bminus + self.Dminus*self.THETA*np.exp(self.THETA*w) # Ana code version
            hat_hprime = 2*self.Aminus*w + self.Bminus + self.Dminus*self.THETA*np.exp(self.THETA*w) # My version
        else:
            ws = self.tau_star + w
            hat_hprime = (self.barC_minus/self.delta_plus)*(ws+(1/self.theta)*(1-np.exp(self.theta*ws)))
        # Finally, we compute the gradient of h(x)
        grad_h = NodesData(hat_hprime*self.XiD + grad_h_c, x.matchingGraph)
        grad_h_index = np.array([np.sum(grad_h[x.matchingGraph.edges[i]]) for i in np.arange(len(x.matchingGraph.edges))])
        
        prob = hMWT("hMWT")
        prob.model(x, grad_h_index, w, self.tau_star, self.Idle_index, self.NUmax)
        prob.optimize(False)
        if prob.is_solution == True:
            u_star = Matching.zeros(x)
            for i in np.arange(len(x.matchingGraph.edges)):
                u_star[x.matchingGraph.edges[i]] += prob.u[i].val
            return u_star
        else:
            raise ValueError('The MIP optimizer has not found a solution')
        #prob.printSolution()
        
    def match_S(self, x):
        # We compute the workload w
        w = np.inner(x.data, self.XiS)
        # We computed the pertubated states and workload
        xtil = x.data + self.beta*(np.exp(-x.data/self.beta)-1.)
        wtil = np.sign(w)*(np.abs(w) + self.beta*(np.exp(-np.abs(w)/self.beta)-1.))
        
        # The function h is the sum of two function: \hat{h}(w) et h_c(x)
        # We compute the gradient of h_c(x)
        grad_ctil = np.multiply(self.costs.data,(1. - np.exp(-x.data/self.beta)))
        if w>=0:
            grad_barCtil = self.barC_plus*(1.-np.exp(-w/self.beta))*self.XiS
        else:
            grad_barCtil = -self.barC_minus*(1.-np.exp(w/self.beta))*self.XiS
        grad_h_c = 2.*self.kappa*(np.inner(self.costs.data, xtil) - np.maximum(self.barC_plus*wtil, -self.barC_minus*wtil) )*(grad_ctil - grad_barCtil)
        # We compute the derivative of \hat{h}(w)
        if w>=0:
            hat_hprime = 2*self.Aplus*w+self.Bplus
        elif w<0 and w>=-self.tau_star:
            #hat_hprime = 2*self.Aminus + self.Bminus + self.Dminus*self.THETA*np.exp(self.THETA*w) # Ana code version
            hat_hprime = 2*self.Aminus*w + self.Bminus + self.Dminus*self.THETA*np.exp(self.THETA*w) # My version
        else:
            ws = self.tau_star + w
            hat_hprime = (self.barC_minus/self.delta_plus)*(ws+(1/self.theta)*(1-np.exp(self.theta*ws)))
        # Finally, we compute the gradient of h(x)
        grad_h = hat_hprime*self.XiS + grad_h_c
        
        prob = hMWT("hMWT")
        prob.model(x, grad_h, w, self.tau_star, self.Idle_index, self.NUmax)
        prob.optimize(False)
        if prob.is_solution == True:
            u_star = Matching.zeros(x)
            for i in np.arange(len(x.matchingGraph.edges)):
                u_star[x.matchingGraph.edges[i]] += prob.u[i].val
            return u_star
        else:
            raise ValueError('The MIP optimizer has not found a solution')
        #prob.printSolution()
        
    def reset_policy(self, x_0):
        pass
    
    def __str__(self):
        if self.Workload_index[0]=='d':
            #return 'hMWT policy D={}'.format(self.Workload_index[1])
            return 'hMWT policy'
        else:
            return 'hMWT policy S={}'.format(self.Workload_index[1])
    
# We create a class MILP which maximizes f.u under the constraints Au <= b for u an array of integers between lb and ub
class hMWT(mip.Problem):
    def model(self, x, grad_h, w, tau_star, Idle_index, NUmax):
        nb_edges = len(x.matchingGraph.edges)
        # The variables are the number of matching in each edge
        # u[i] correspond to the number of matching in x.matchingGraph.edges[i]
        self.u = u = mip.VarVector([nb_edges],"u",mip.INT,lb=0,ub=NUmax)
        # The goal is to maximize grad_h*u
        mip.maximize(mip.sum_(grad_h[i]*u[i] for i in range(nb_edges)))

        ### The inequalities constraints ###
        # The number of matchings can not be higher than NUmax
        mip.sum_(u[i] for i in range(nb_edges)) <= NUmax
        
        # The number of matchings can not be higher than the number of items in the system
        for i in x.matchingGraph.demand_class_set:
            linked_edges = [x.matchingGraph.edgeIndex((i,s)) for s in x.matchingGraph.demandToSupply[(i,)]]
            mip.sum_(u[k] for k in linked_edges) <= x.demand(i)
        for j in x.matchingGraph.supply_class_set:
            linked_edges = [x.matchingGraph.edgeIndex((d,j)) for d in x.matchingGraph.supplyToDemand[(j,)]]
            mip.sum_(u[k] for k in linked_edges) <= x.supply(j)
            
        # The workload process can not be higher than the threshold
        mip.sum_(u[i] for i in Idle_index) <= np.maximum(-tau_star-w, 0.)
        

