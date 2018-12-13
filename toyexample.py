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
            self.states.append(Tower(loc, 1, 1, id_nos))
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

    def WritePolicy(self,time):
        t_f = open("policyplan.out", "a")
        t_f.write("\nTime: "+str(time))
        i = 0
        WP = "WP"
        for uav in self.UAV:
            t_f.write("\nkp00"+str(i)+" "+WP)
            s = uav.current_state
            t_f.write(str(s))
            while s is not uav.target:
                s_new = uav.policy[s]
                s_new = list(s_new)[0]
                if s_new is s:
                    break
                s = s_new
                t_f.write(" "+WP+str(s))
            i+=1

    def Run(self,tasks):
        time = 0
        while not self.UTM_Goal:
            self.Coordinate()
            self.WritePolicy(time)
            self.Propagate(tasks)
            self.DisplayStates()
            time += 1

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
        co_ord_agents = list(co_ord_agents)
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
        for i in range(no_agents):
            state_locs_hold = state_locs.copy()
            state_locs_hold[i] = -1
            act_s = [q[1] for q in Q[i]]
            st_dict = dict((k,i) for i,k in enumerate(act_s))
            inter = set(st_dict.keys()).intersection(state_locs_hold)
            index = [st_dict[x] for x in inter]
            for i_nd in index:
                Qa[i][i_nd] -= 500
        if no_agents is 2:
            Qx,Qy = np.meshgrid(Qa[0],Qa[1],indexing='ij')
            comp_Q = Qx+Qy
            for i in range(no_agents):
                s_acts = [(j,q[1]) for j,q in enumerate(Q[i])]
                rem_agents = list(range(no_agents))
                rem_agents.pop(i)
                for j,q_j in s_acts:
                    for k in rem_agents:
                        s_next_acts = [k_i for k_i,q in enumerate(Q[k]) if q[1] is q_j]
                        if i is 0:
                            for r_i in s_next_acts:
                                comp_Q[j,r_i] -= 500
                        elif i is 1:
                            for r_i in s_next_acts:
                                comp_Q[r_i,j] -= 500
        else:
            Qx,Qy,Qz = np.meshgrid(Qa[0],Qa[1],Qa[2],indexing='ij')
            comp_Q = Qx + Qy + Qz
            for i in range(no_agents):
                s_acts = [(j,q[1]) for j,q in enumerate(Q[i])]
                rem_agents = list(range(no_agents))
                rem_agents.pop(i)
                for j,q_j in s_acts:
                    for k in rem_agents:
                        s_next_acts = [k_i for k_i,q in enumerate(Q[k]) if q[1] is q_j]
                        if i is 0 and k is 1:
                            for r_i in s_next_acts:
                                comp_Q[j,r_i,:] -= 500
                        elif i is 0 and k is 2:
                            for r_i in s_next_acts:
                                comp_Q[j,:,r_i] -= 500
                        elif i is 1 and k is 0:
                            for r_i in s_next_acts:
                                comp_Q[r_i,j,:] -= 500
                        elif i is 1 and k is 2:
                            for r_i in s_next_acts:
                                comp_Q[:,j,r_i] -= 500
                        elif i is 2 and k is 0:
                            for r_i in s_next_acts:
                                comp_Q[r_i,:, j] -= 500
                        elif i is 2 and k is 1:
                            for r_i in s_next_acts:
                                comp_Q[:, r_i,j] -= 500
        act = np.unravel_index(np.argmax(comp_Q),comp_Q.shape)
        for act_i,agent_i,loc_i,q_i in zip(act,co_ord_agents,state_locs,Q):
            s_acts = [ q[1] for q in q_i]
            self.UAV[agent_i].policy[loc_i] = set([s_acts[act_i]])


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

