import gym
import numpy as np
import math
from matplotlib import pyplot as plt
from random import randint
from statistics import median, mean


env = gym.make('CartPole-v2')
#action_set =[]
award_set =[]

test_run = 15


def sig(x):
	return 1/(1+np.exp(-x))

def reLu(x):
	return np.maximum(0, x)

def intial_gen(test_run):
	input_weight = []
	input_bias = []
	hidden_weight = []
	n_hid_node = 4
	for i in range(test_run):
		in_w = np.random.rand(4,n_hid_node)
		input_weight.append(in_w)

		in_b = np.random.rand(n_hid_node)
		input_bias.append(in_b)

		hi_w = np.random.rand(n_hid_node, 1)
		hidden_weight.append(hi_w)
	
	generation = [input_weight, input_bias, hidden_weight]
	return generation

def nn(obs,in_w,in_b,hi_w):
	obs = np.reshape(obs,(1,4))
	  
	A1 = reLu(np.dot(obs,in_w)+in_b.T)

	hid_layer = np.dot(A1,hi_w)
	out_put = reLu(hid_layer)

	if out_put < 0.99: #(this should be close to 1 for better results)
		out_put = 0
	else:
		out_put = 1

	return out_put

def run_env(env,in_w,in_b,hi_w):
	obs = env.reset()
	award = 0
	for t in range(5000):
		#env.render() thia slows the process
		action = nn(obs,in_w,in_b,hi_w)
		obs, reward, done, info = env.step(action)
		award += reward 
		if done:
			break
	return award


def rand_run(env,test_run):
	award_set = []
	generations = intial_gen(test_run)

	for episode in range(test_run):# run env 10 time
		in_w  = generations[0][episode]
		in_b = generations[1][episode]
		hi_w =  generations[2][episode]
		award = run_env(env,in_w,in_b,hi_w)
		award_set = np.append(award_set,award)
	gen_award = [generations, award_set]
	return gen_award  


def mutation(new_dna):

	j = np.random.randint(0,len(new_dna))
	if ( 0 <j < 7): # controlling rate of amount mutation
		for ix in range(j):
			n = np.random.randint(0,len(new_dna)) #random postion for mutation
			new_dna[n] = new_dna[n] + np.random.rand()

	mut_dna = new_dna

	return mut_dna

def crossover(Dna_list):
	newDNA_list = []
	newDNA_list.append(Dna_list[0])
	newDNA_list.append(Dna_list[1]) 
	
	for l in range(13):  # generation after crassover
		j = np.random.randint(0,len(Dna_list[0]))
		new_dna = np.append(Dna_list[0][:j], Dna_list[1][j:])

		mut_dna = mutation(new_dna)
		newDNA_list.append(mut_dna)

	return newDNA_list


def reproduce(award_set, generations):

	good_award_idx = award_set.argsort()[-2:][::-1] # here only best 2 are selected 
	good_generation = []
	DNA_list = []

	new_input_weight = []
	new_input_bias = []
	new_hidden_weight = []
	new_award_set = []
	for index in good_award_idx:
		
		w1 = generations[0][index]
		dna_in_w = w1.reshape(w1.shape[1],-1)
	
		b1 = generations[1][index]
		dna_b1 = np.append(dna_in_w, b1)

		wh = generations[2][index]
		dna = np.append(dna_b1, wh)


		#wh = generations[1][index]
		#dna = np.append(dna_in_w, wh) # single DNA

		DNA_list.append(dna) # make 2 dna for good gerneration

	newDNA_list = crossover(DNA_list)

	for newdna in newDNA_list: # collection of weights from dna info
		
	 	newdna_in_w1 = np.array(newdna[:generations[0][0].size]) 
	 	new_in_w = np.reshape(newdna_in_w1, (-1,generations[0][0].shape[1]))
	 	new_input_weight.append(new_in_w)

		new_in_b = np.array([newdna[newdna_in_w1.size:newdna_in_w1.size+generations[1][0].size]]).T #bias
	 	new_input_bias.append(new_in_b)

	 	sl = newdna_in_w1.size + new_in_b.size
	 	new_hi_w   = np.array([newdna[sl:]]).T
	 	new_hidden_weight.append(new_hi_w)

	 	new_award = run_env(env, new_in_w, new_hi_w, new_in_b) #bias
	 	new_award_set = np.append(new_award_set,new_award)

	new_generation = [new_input_weight,new_input_bias,new_hidden_weight]

	return new_generation, new_award_set


def evolution(env,test_run,n_of_generations):
	gen_award = rand_run(env, test_run)

	current_generations = gen_award[0] 
	current_award_set = gen_award[1]
	
	A =[]
	for n in range(n_of_generations):
		new_generation, new_award_set = reproduce(current_award_set, current_generations)
		current_gen = new_generation
		current_award_set = new_award_set
		a = np.amax(current_award_set)
		print("generation: {}, score: {}".format(n, a))
		A = np.append(A, a)
	Best_award = np.amax(A)

	
	plt.plot(A)
	plt.xlabel('generations')
	plt.ylabel('score')

	print('Average accepted score:',mean(A))
	print('Median score for accepted scores:',median(A))
	return plt.show()


n_of_generations = 500
evolution(env, test_run, n_of_generations)

