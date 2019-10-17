import random as rnd
import numpy as np

n = 4000
ns = n
m = 4
k = 1
s0 = rnd.randint(0,n)
ng = 1
horizon = 250
gamma = 0.9

f = open("mdp"+str(n)+".raw","w+")

line1 = "{},{},{},{},{},{},{},{}\n".format(n,ns,m,k,s0,ng,horizon,gamma)
f.write(line1)

goals = np.random.randint(n,size=ng)
goals_char = np.char.mod('%d', goals)
goals_char = ",".join(goals_char)+'\n'
f.write(goals_char)

for i in range(n*m):
	state_trans = np.random.choice(n, ns, replace=False)
	state_trans_char = np.char.mod('%d', state_trans)
	state_trans_char = ",".join(state_trans_char)+'\n'
	f.write(state_trans_char)

for i in range(n*m):
	prob = np.random.rand(ns)
	prob /= prob.sum()
	prob_char = np.char.mod('%f', prob)
	prob_char = ",".join(prob_char)+'\n'
	f.write(prob_char)

for i in range(m*k):
	reward = np.random.uniform(-1,1,n)
	reward_char = np.char.mod('%f', reward)
	reward_char = ",".join(reward_char)+'\n'
	f.write(reward_char)

f.close()
