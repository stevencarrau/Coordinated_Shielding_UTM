
from nfa import NFA
import numpy as np
import random
from tqdm import tqdm

class MDP(NFA):
    def __init__(self, states, alphabet, transitions=[]):
        # we call the underlying NFA constructor but drop the probabilities
        trans = [(s, a, t) for s, a, t, p in transitions]
        super(MDP, self).__init__(states, alphabet, trans)
        # in addition to the NFA we need a probabilistic transition
        # function
        self._prob_cache = dict()
        for s, a, t, p in transitions:
            self._prob_cache[(s, a, t)] = p
        self._prepare_post_cache()

    def prob_delta(self, s, a, t):
        return self._prob_cache[(s, a, t)]

    def sample(self, state, action):
        """Sample the next state according to the current state, the action,  and
        the transition probability. """
        in_target=False
        if action not in self.available(state):
            return None
        # N = len(self.post(state, action))
        prob = []
        for t in self.post(state, action):
            prob.append(self.prob_delta(state, action, t))
        #
        # rand_val = random.random()
        # total = 0
        # for key in self.post(state,action):
        #     total +=self.prob_delta(state,action,key)
        #
        #     if rand_val <= total:
        #
        #         next_state=key
        #         break
        # (x,y,t)=state
        # ballpos = (-200, 0)
        # if (abs(x) > 1000 or abs(y) > 1000) or (y >= ballpos[1] + 100 and abs(x) <= 400) or (t < 25 or t > 155):
        #
        #     in_target=True
        #
        #
        # if x==0 and y==0 and t==90:
        #
        #     in_target=True


        next_state = list(self.post(state, action))[np.random.choice(range(len(self.post(state, action))),1,True,prob)[0]]
        # Note that only one element is chosen from the array, which is the
        # output by random.choice
        return next_state,in_target

    def set_prob_delta(self, s, a, t, p):
        self._prob_cache[(s, a, t)] = p

    def evaluate_policy_E(self,policy,R, epsilon = 0.001, gamma = 0.9):
        V1 = dict.fromkeys(self.states,0)
        while True:
            e = 0
            V = V1.copy()
            for s in self.states:
                if type(policy[s]) == set:
                    a= random.choice(list(policy[s]))
                else:
                    a=policy[s]
                V1[s]= sum([self.prob_delta(s,a,next_s)*(gamma*V[next_s] + R[s,a]) for next_s in self.post(s,a)])
                e = max(e, abs(V1[s] - V[s]))
            if e < epsilon:
                return V

    def expected_utility(self,a, s, U):
        "The expected utility of doing a in state s, according to the MDP and U."
        return sum([self.prob_delta(s,a,next_s) * U[next_s] for next_s in self.post(s,a)])

    def best_policy(self, U):
        """Given an MDP and a utility function U, determine the best policy,
        as a mapping from state to action."""
        pi = {}
        utility = {s:dict() for s in self.states}
        for s in self.states:
            for a in self.available(s):
                utility[s][a] = self.expected_utility(a,s,U)
            pi[s] = utility[s].keys()[utility[s].values().index(max(utility[s].values()))]
        return pi

    def T_step_value_iteration(self,R, T):
        """Solving an MDP by value iteration for T-step horizon"""
        U1 = dict([(s, 0) for s in self.states])
        self._prepare_post_cache()
        policy = dict([(s, set()) for s in self.states])
        t = T
        while t > 0:
            U = U1.copy()
            delta = 0
            for s in self.states:
                U1[s] = max([sum([self.prob_delta(s,a,s1) * (U[s1] + R[s, a,s1])
                                  for s1 in self.post(s, a)])]
                            for a in self.available(s))[0]
                delta = max(delta, abs(U1[s] - U[s]))
            t = t - 1
            print(t)
        for s in self.states:
            Vmax = dict()
            for a in self.available(s):
                Vmax[a] = [sum([self.prob_delta(s,a,s1) * (U[s1] + R[s, a,s1])
                                for s1 in self.post(s, a)])][0]
            maxV = max(Vmax.values())
            for a in Vmax.keys():
                if Vmax[a] == maxV:
                    policy[s].add(a)
        return U, policy

    # def E_step_value_iteration(self,R,
    #                     epsilon=0.1, gamma=0.9):
    #     U1 = dict([(s, 0) for s in self.states])
    #     while True:
    #         U = U1.copy()
    #         delta = 0
    #         for s in self.states:
    #             U1[s] = max([sum([self.prob_delta(s,a,next_s) * (gamma*U[next_s] + R[s,a,next_s]) for next_s in self.post(s,a)])
    #                                         for a in self.available(s)])
    #             delta = max(delta, abs(U1[s] - U[s]))
    #             print(delta)
    #         if delta < epsilon * (1 - gamma) / gamma:
    #              break
    #     policy = self.best_policy(U)
    #     return policy

    def write_to_file(self,filename,initial,targets=set()):
        file = open(filename, 'w')
        self._prepare_post_cache()
        file.write('|S| = {}\n'.format(len(self.states)))
        file.write('|A| = {}\n'.format(len(self.alphabet)))
        file.write('s0 = {}\n'.format(initial))
        if len(targets)>0:
            stri = 'targets = ('
            for t in targets:
                stri += '{} '.format(t)
            stri = stri[:-1]
            stri+=')\n'
            file.write(stri)

        file.write('s a t p\n')
        for s in self.states:
            for a in self.available(s):
                for t in self.post(s,a):
                    file.write('{} {} {} {}\n'.format(s,a,t,self.prob_delta(s,a,t)))

    def E_step_value_iteration(self,R,sink,targstates,
                        epsilon=0.1, gamma=0.8):
        policyT = dict([])
        Vstate1 = dict([])
        Vstate1.update({s: 0 for s in self.states})
        e = 1
        Q = dict([])
        while e > epsilon:
            Vstate = Vstate1.copy()
            for s in tqdm(self.states - sink- targstates):
                acts = self.available(s)
                optimal = -1000
                act = None
                for a in self.available(s):
                    Q[(s, a)] = sum([self.prob_delta(s, a, next_s) *
                                     (gamma*Vstate[next_s] + R[(s,a,next_s)])
                                     for next_s in self.post(s, a)])
                    if Q[(s, a)] >= optimal:
                        optimal = Q[(s, a)]
                        act = a
                    else:
                        pass
                acts = set([])
                for act in self.available(s):
                    if Q[(s, act)] == optimal:
                        acts.add(act)
                Vstate1[s] = optimal
                policyT[s] = acts
            e = max(np.abs([Vstate1[s] -
                         Vstate[s] for s in self.states]))  # the abs error
            print(e)
        # policyT[list(targstates)[0]] = targstates # Remain in target set
        return Vstate1, policyT,Q

    def max_reach_prob(self, target,sinks=set(),epsilon=0.1):
        """
        infinite time horizon
        Value iteration: Vstate[s] the maximal probability of hitting the
        target AEC within infinite steps.
        """
        policyT = dict([])
        Vstate1 = dict([])
        R = dict()
        Win = target
        NAEC = set(self.states) - Win

        Vstate1.update({s: 1 for s in list(Win)})
        Vstate1.update({s: 0 for s in list(NAEC)})
        policyT.update({s: self.available(s) for s in list(Win)})
        e = 1
        Q = dict([])
        while e > epsilon:
            Vstate = Vstate1.copy()
            for s in tqdm(set(self.states) - Win - sinks):
                acts = self.available(s)
                optimal = 0
                act = None
                for a in self.available(s):
                    Q[(s, a)] = sum([self.prob_delta(s, a, next_s) *
                                     Vstate[next_s]
                                     for next_s in self.post(s, a)])
                    if Q[(s, a)] >= optimal:
                        optimal = Q[(s, a)]
                        act = a
                    else:
                        pass
                acts = set([])
                for act in self.available(s):
                    if Q[(s, act)] == optimal:
                        acts.add(act)
                Vstate1[s] = optimal
                policyT[s] = acts
            e = abs(max([Vstate1[s] -
                         Vstate[s] for s in self.states]))  # the abs error
            print(e)
                # print "iteration: {} and the state
                # value is {}".format(t, Vstate1)
        for s in sinks:
            policyT[s] = {'stop'}
        return Vstate1, policyT

    def policyTofile(self,policy,outfile):
        file = open(outfile, 'w')
        file.write('policy = dict()\n')
        for s in self.states:
            x = -s[1]
            y = s[0]
            t = (s[2]-270)%360
            s2 = (x,y,t)
            if s not in policy.keys():
                file.write('policy[' + str(s2) + '] = \'stop\' '+'\n')
            else:
                if 'stop' not in policy[s]:
                    file.write('policy[' + str(s2) + '] = \'' + str(policy[s]) + '\'\n')
                else:
                    file.write('policy['+str(s2)+'] = \'stop\' '+'\n')
        file.close()





    def computeTrace(self,init,policy,T,targ = None):
        s = init
        trace = dict()
        t = 0
        trace[t] = s
        while t < T:
            #print 't = ', t, 'state = ', s
            act = policy
            #print ' act = ', act
            ns,target = self.sample(s,act)
            #print(ns)
            t += 1
            s = ns
            trace[t] = ns
            return ns,target
            #if ns == targ:
             #   return trace



