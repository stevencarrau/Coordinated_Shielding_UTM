import numpy as np
from mdp import MDP
import math


class UTM_Network():
    def __init__(self, n_agents=1, n_towers=2, tower_locs=[], initial=[], tasks=[], prob_success=0.9):
        self.n_agents = n_agents
        self.n_towers = n_towers
        # State space and transtions for single agent
        self.states = []
        self.state_labels = []
        self.transitions = set()
        self.rewards = dict()
        self.actlist = set()
        id_nos = 0
        for loc in tower_locs:
            self.states.append(Tower(loc, 1, 2, id_nos))
            self.state_labels.append(id_nos)
            id_nos += 1
        for i in self.states:
            for j in self.states:
                if i.DistanceTo(j) < i.VLOS_radius + j.VLOS_radius:
                    if i.id_no is j.id_no:
                        i.AddAction(j.id_no, 1)
                        self.actlist = self.actlist.union(set(i.actlist.keys()))
                        self.transitions.add((i.id_no, j.id_no, j.id_no, 1))
                        self.rewards.update({(i.id_no, j.id_no,j.id_no): -1.5})
                    else:
                        i.AddAction(j.id_no, prob_success)
                        self.actlist = self.actlist.union(set(i.actlist.keys()))
                        self.transitions.add((i.id_no, j.id_no, j.id_no, prob_success))
                        self.transitions.add((i.id_no, j.id_no, i.id_no, 1 - prob_success))
                        self.rewards.update({(i.id_no, j.id_no,j.id_no): -i.DistanceTo(j)})
                        self.rewards.update({(i.id_no, j.id_no,i.id_no): -1.5})
        for s in self.states:
            co_ord_sector = set()
            co_ord_sector = co_ord_sector.union(set(s.actlist.keys()))
            for k in co_ord_sector:
                co_ord_sector = co_ord_sector.union(set(self.states[k].actlist.keys()))
            s.AddCoordinationZones(co_ord_sector)

        # Agents
        self.n_UAVs = len(initial)
        self.UAV = []
        for init, tar in zip(initial, tasks):
            self.UAV.append(UAV(init, tar))
            self.UAV[-1].AddMDP(self.state_labels, self.actlist, self.transitions)
            self.UAV[-1].rewards = self.UAV[-1].AddRewards(self.rewards,self.UAV[-1].target,self.states)
            _,self.UAV[-1].policy,self.UAV[-1].Q = self.CalculateRoute(self.UAV[-1].mdp, self.UAV[-1].rewards,tar)
        self.UTM_Goal = self.UAVsAtGoal(tasks)

    def UAVsAtGoal(self,tasks):
        flag = True
        for uav,tar in zip(self.UAV,tasks):
            flag = flag and (uav.current_state is tar)
        return flag

    def CalculateRoute(self, mdp, R,target):
        return mdp.E_step_value_iteration(R,set(),set([target]))

    def Propagate(self,tasks):
        for u_av in self.UAV:
            u_av.PropagateUAV(u_av.policy)
        self.UTM_Goal = self.UAVsAtGoal(tasks)

    def DisplayStates(self):
        print([u_av.current_state for u_av in self.UAV])


    def Run(self,tasks):
        while not self.UTM_Goal:
            self.Propagate(tasks)
            self.DisplayStates()

    def Coordinate(self):
        

class Tower():
    def __init__(self, location, n_operators, VLOS_radius, id_no):
        self.location = location
        self.n_operators = n_operators
        self.VLOS_radius = VLOS_radius
        self.actlist = dict()
        self.id_no = id_no

    def DistanceTo(self, state):
        return np.linalg.norm(np.array(self.location) - np.array(state.location))

    def AddAction(self, state, prob):
        self.actlist.update({state: prob})

    def AddCoordinationZones(self,co_ord):
        self.co_ord_sector = co_ord


class UAV():
    def __init__(self, initial, target):
        self.initial_loc = initial
        self.current_state = initial
        self.target = target
        self.mdp = []
        self.rewards = []
        self.policy = dict()

    def AddMDP(self, states,actlist, transitions):
        self.mdp = MDP(states,actlist, transitions)

    def AddRewards(self,R,target,states):
        rewards = R.copy()
        for s in states:
            for a in s.actlist:
                if a is target:
                    s_next = a
                    rewards[s.id_no,a,s_next] = 100
        return rewards


    def PropagateUAV(self,policy):
        s = self.current_state
        a = policy[s]
        a = list(a)[0]
        next_s,target = self.mdp.sample(s,a)
        self.current_state = next_s

locations = [0, 7, 8]
targets = [5, 4, 7]
tower_positions = [[0, 0], [1, 1], [1, 2.8], [2, 2], [1.3, -0.2], [4, 3], [2.8, 3.4], [3.1, 1.8], [2, 4]]

network = UTM_Network(3, 9, tower_positions, locations, targets)
network.DisplayStates()
network.Run(targets)

