import numpy as np
from mdp import MDP
import math

class UTM_Network():
	def __init__(self,n_agents,n_towers,tower_locs=[],initial,tasks):
		self.n_agents = n_agents
		self.n_towers = n_towers
		self.states = []
		for loc in tower_locs:
			self.states.append(Tower(loc,1,2))

		self.n_UAVs  = len(initial)
		self.UAV = []
		for init,tar in zip(initial,tasks) :
			self.UAV.append(UAV(init,tar))


class Tower():
	def __init__(self,location,n_operators,VLOS_radius):
		self.location = location
		self.n_operators = n_operators
		self.VLOS_radius = VLOS_radius

class UAV():
	def __init__(self,initial,target):
		self.initial_loc = initial
		self.current_state = initial



locations = [1,8,9]
targets = [6,5,8]
tower_positions = [[0,0],[1,1],[1,2.8],[2,2],[1.3,-0.2],[4,3],[2.8,3.4],[3.1,1.8],[2,4]]

network = UTM_Network(3,9,tower_positions,locations,targets)

