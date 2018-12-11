import numpy as np
from mdp import MDP
import math


class UTM_Network():
    def __init__(self, n_agents=1, n_towers=2, tower_locs=[], initial=[], tasks=[], prob_success=0.9):
        self.n_agents = n_agents
        self.n_towers = n_towers
        # State space and transtions for single agent
        self.states = []
        self.transitions = set()
        self.rewards = dict()
        self.actlist = set()
        id_nos = 0
        for loc in tower_locs:
            self.states.append(Tower(loc, 1, 2, id_nos))
            id_nos += 1
        for i in self.states:
            for j in self.states:
                if i.DistanceTo(j) < i.VLOS_radius + j.VLOS_radius:
                    if i.id_no is j.id_no:
                        i.AddAction(j.id_no, 1)
                        self.actlist.union(set(i.actlist.keys()))
                        self.transitions.add((i.id_no, j.id_no, j.id_no, 1))
                        self.rewards.update({(i.id_no, j.id_no): 1.5})
                    else:
                        i.AddAction(j.id_no, prob_success)
                        self.actlist.union(set(i.actlist.keys()))
                        self.transitions.add((i.id_no, j.id_no, j.id_no, prob_success))
                        self.transitions.add((i.id_no, j.id_no, i.id_no, 1 - prob_success))
                        self.rewards.update({(i.id_no, j.id_no): i.DistanceTo(j)})
        for s in self.states:
            co_ord_sector = set()
            co_ord_sector.union(set(s.actlist.keys()))
            for k in co_ord_sector:
                co_ord_sector.union(set(k.actlist.keys()))

        # Agents
        self.n_UAVs = len(initial)
        self.UAV = []
        for init, tar in zip(initial, tasks):
            self.UAV.append(UAV(init, tar))
            self.UAV[-1].AddMDP(self.states, self.actlist, self.transitions)
            self.UAV[-1].Policy = self.CalculateRoute(self.UAV[-1].mdp, self.rewards, self.UAV[-1].target)

    def CalculateRoute(self, mdp, R, target):
        return mdp.E_step_value_iteration(R, set(), target, 0.1, 0.8)


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


class UAV():
    def __init__(self, initial, target):
        self.initial_loc = initial
        self.current_state = initial
        self.target = target

    def AddMDP(self, states, actlist, transitions):
        self.mdp = MDP(states, actlist, transitions)


locations = [1, 8, 9]
targets = [6, 5, 8]
tower_positions = [[0, 0], [1, 1], [1, 2.8], [2, 2], [1.3, -0.2], [4, 3], [2.8, 3.4], [3.1, 1.8], [2, 4]]

network = UTM_Network(3, 9, tower_positions, locations, targets)
