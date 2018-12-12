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
        return mdp.E_step_value_iteration(R,set(),set())

    def Propagate(self,tasks):
        for u_av in self.UAV:
            u_av.PropagateUAV(u_av.policy)
        self.UTM_Goal = self.UAVsAtGoal(tasks)

    def DisplayStates(self):
        print([u_av.current_state for u_av in self.UAV])


    def Run(self,tasks):
        while not self.UTM_Goal:
            self.Propagate(tasks)
            self.Coordinate()
            self.DisplayStates()

    def Coordinate(self):
        locations = []
        for uav in self.UAV:
            locations.append(uav.current_state)
        co_ord_agents = set()
        for uav_id in range(self.n_agents):
            loc_copy = locations.copy()
            loc_copy[uav_id] = -1
            st_dict = dict((k,i) for i,k in enumerate(loc_copy))
            inter = set(st_dict.keys()).intersection(self.states[self.UAV[uav_id].current_state].co_ord_sector)
            index = [st_dict[x] for x in inter]
            co_ord_agents = co_ord_agents.union(index)
        if co_ord_agents:
            Q_all = []
            Q_allvalues = []
            state_locs = []
            for agent in co_ord_agents:
                Q_a = [k for k in self.UAV[agent].Q.keys() if k[0] is self.UAV[agent].current_state]
                Q_all.append(Q_a)
                Q_avalues = [self.UAV[agent].Q[e] for e in Q_a]
                Q_allvalues.append(Q_avalues)
                state_locs.append(self.UAV[agent].current_state)
            self.max_DBN(Q_all,Q_allvalues,state_locs,co_ord_agents)

    def max_DBN(self,Q,Qa,state_locs,co_ord_agents):
        no_agents = len(co_ord_agents)
        i=0
        for n_a in co_ord_agents:
            state_locs_hold = state_locs.copy()
            state_locs_hold[n_a] = -1
            st_dict = dict((k,i) for i,k in enumerate(state_locs_hold))
            act_s = [q[1] for q in Q[i]]
            inter = set(st_dict.keys()).intersection(act_s)
            index = [st_dict[x] for x in inter]
            for i_nd in index:
                Qa[i][i_nd] -= 100
            i+=1
        i=0
        if no_agents is 2:
            Qx,Qy = np.meshgrid(Qa[0],Qa[1])
            comp_Q = Qx+Qy
            s_acts = [q[1] for q in Q[i]]
            for j in s_acts:
                []
        else:
            Qx,Qy,Qz = np.meshgrid(Qa[0],Qa[1],Qa[2])
            comp_Q = Qx + Qy + Qz

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

