import numpy as np 
import mlrose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, datasets
import time
from random import randint
import warnings


# Define 3 optimization problems 

# 1. 4-Peaks Problem
def four_peaks_input_size():
	# TO DO: 
	# - Try different t_pct values --> Threshold Parameter
	# - Try different decay schedules 
	# The fitness function should be as big as possible
	fitness_sa_arr = []
	fitness_rhc_arr = []
	fitness_ga_arr = []
	fitness_mimic_arr = []

	time_sa_arr = []
	time_rhc_arr = []
	time_ga_arr = []
	time_mimic_arr = []
	for n in range(5,120,20):
		fitness = mlrose.FourPeaks(t_pct=0.15)
		problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize=True, max_val=2)
		init_state = np.random.randint(2,size=n)
		schedule = mlrose.ExpDecay()
		st = time.time()
		best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 1000, max_iters=100000, init_state = init_state, curve=True)
		end = time.time()
		sa_time = end-st

		st = time.time()
		best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts = 1000, max_iters=10000, init_state = init_state, curve=True)
		end = time.time()
		rhc_time = end-st

		st = time.time()
		best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts = 1000, max_iters=10000, curve=True)
		end = time.time()
		ga_time = end-st

		st = time.time()
		best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,pop_size=500, max_attempts = 100, max_iters=10000, curve=True)
		end = time.time()
		mimic_time = end-st
		print(mimic_time,n)

		fitness_sa_arr.append(best_fitness_sa)
		fitness_rhc_arr.append(best_fitness_rhc)
		fitness_ga_arr.append(best_fitness_ga)
		fitness_mimic_arr.append(best_fitness_mimic)

		time_sa_arr.append(sa_time)
		time_rhc_arr.append(rhc_time)
		time_ga_arr.append(ga_time)
		time_mimic_arr.append(mimic_time)

	fitness_sa_arr = np.array(fitness_sa_arr)
	fitness_rhc_arr = np.array(fitness_rhc_arr)
	fitness_ga_arr = np.array(fitness_ga_arr)
	fitness_mimic_arr = np.array(fitness_mimic_arr)

	time_sa_arr = np.array(time_sa_arr)
	time_rhc_arr = np.array(time_rhc_arr)
	time_ga_arr = np.array(time_ga_arr)
	time_mimic_arr = np.array(time_mimic_arr)

	plt.figure()
	plt.plot(np.arange(5,120,20),fitness_sa_arr,label='SA')
	plt.plot(np.arange(5,120,20),fitness_rhc_arr,label = 'RHC')
	plt.plot(np.arange(5,120,20),fitness_ga_arr, label = 'GA')
	plt.plot(np.arange(5,120,20),fitness_mimic_arr, label = 'MIMIC')
	plt.xlabel('Input Size')
	plt.ylabel('Fitness Vaue')
	plt.legend()
	plt.title('Fitness Value vs. Input Size (4 Peaks)')
	plt.savefig('4_peaks_input_size_fitness.png')

	plt.figure()
	plt.plot(np.arange(5,120,20),time_sa_arr,label='SA')
	plt.plot(np.arange(5,120,20),time_rhc_arr,label='RHC')
	plt.plot(np.arange(5,120,20),time_ga_arr,label='GA')
	plt.plot(np.arange(5,120,20),time_mimic_arr,label='MIMIC')
	plt.legend()
	plt.xlabel('Input Size')
	plt.ylabel('Computation Time (s)')
	plt.title('Computation Time vs. Input Size (4 Peaks)')
	plt.savefig('4_peaks_input_size_computation.png')


def four_peaks_iterations():
	# TO DO: 
	# - Try different t_pct values --> Threshold Parameter
	# - Try different decay schedules 
	# The fitness function should be as big as possible
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size = 100)
	schedule = mlrose.ExpDecay()
	best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts = 1000, max_iters=10000, init_state = init_state, curve=True)
	best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts = 1000, max_iters=10000, curve=True)
	best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,pop_size=500,max_attempts = 1000, max_iters=10000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa,label='SA')
	plt.plot(fitness_curve_rhc,label='RHC')
	plt.plot(fitness_curve_ga,label='GA')
	plt.plot(fitness_curve_mimic,label='MIMIC')
	plt.legend()
	plt.ylabel('Fitness Value')
	plt.xlabel('Number of Iterations')
	plt.title('Fitness Value vs. Number of Iterations (4 Peaks)')
	plt.savefig('4_peaks_iterations.png')
	return
	
def four_peaks_t(): 
	fitness = mlrose.FourPeaks(t_pct=0.5)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)


	fitness = mlrose.FourPeaks(t_pct=0.25)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	fitness = mlrose.FourPeaks(t_pct=0.35)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.45)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	plt.figure()
	plt.plot(fitness_curve_sa_1,label='t_pct = 0.5')
	plt.plot(fitness_curve_sa_2,label='t_pct = 0.15')
	plt.plot(fitness_curve_sa_3,label='t_pct = 0.25')
	plt.plot(fitness_curve_sa_4,label='t_pct = 0.35')
	plt.plot(fitness_curve_sa_5,label='t_pct = 0.45')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.title('Fitness Value vs. Number of Iterations (4 Peaks - SA)')
	plt.savefig('4_peaks_t_pct_SA.png')

	fitness = mlrose.FourPeaks(t_pct=0.5)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	fitness = mlrose.FourPeaks(t_pct=0.25)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	fitness = mlrose.FourPeaks(t_pct=0.35)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.45)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	plt.figure()
	plt.plot(fitness_curve_sa_1,label='t_pct = 0.5')
	plt.plot(fitness_curve_sa_2,label='t_pct = 0.15')
	plt.plot(fitness_curve_sa_3,label='t_pct = 0.25')
	plt.plot(fitness_curve_sa_4,label='t_pct = 0.35')
	plt.plot(fitness_curve_sa_5,label='t_pct = 0.45')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.title('Fitness Value vs. Number of Iterations (4 Peaks - RHC)')
	plt.savefig('4_peaks_t_pct_RHC.png')

	fitness = mlrose.FourPeaks(t_pct=0.5)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)

	fitness = mlrose.FourPeaks(t_pct=0.25)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)

	fitness = mlrose.FourPeaks(t_pct=0.35)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.45)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)

	plt.figure()
	plt.plot(fitness_curve_sa_1,label='t_pct = 0.5')
	plt.plot(fitness_curve_sa_2,label='t_pct = 0.15')
	plt.plot(fitness_curve_sa_3,label='t_pct = 0.25')
	plt.plot(fitness_curve_sa_4,label='t_pct = 0.35')
	plt.plot(fitness_curve_sa_5,label='t_pct = 0.45')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.title('Fitness Value vs. Number of Iterations (4 Peaks - GA)')
	plt.savefig('4_peaks_t_pct_GA.png')

def four_peaks_rhc():
	fitness = mlrose.FourPeaks(t_pct = 0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_rhc_1, best_fitness_rhc_1, fitness_curve_rhc_1 = mlrose.random_hill_climb(problem,restarts = 0, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_2, best_fitness_rhc_2, fitness_curve_rhc_2 = mlrose.random_hill_climb(problem,restarts = 2, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_3, best_fitness_rhc_3, fitness_curve_rhc_3 = mlrose.random_hill_climb(problem,restarts = 4, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_4, best_fitness_rhc_4, fitness_curve_rhc_4 = mlrose.random_hill_climb(problem,restarts = 6, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_5, best_fitness_rhc_5, fitness_curve_rhc_5 = mlrose.random_hill_climb(problem,restarts = 8, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_6, best_fitness_rhc_6, fitness_curve_rhc_6 = mlrose.random_hill_climb(problem,restarts = 10, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)

	plt.figure()
	plt.plot(fitness_curve_rhc_1,label='restarts = 0')
	plt.plot(fitness_curve_rhc_2,label='restarts = 2')
	plt.plot(fitness_curve_rhc_3,label='restarts = 4')
	plt.plot(fitness_curve_rhc_4,label='restarts = 6')
	plt.plot(fitness_curve_rhc_5,label='restarts = 8')
	plt.plot(fitness_curve_rhc_6,label='restarts = 10')
	plt.title('4 Peaks RHC Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('4_peaks_rhc_analysis.png')

def four_peaks_mimic():
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.mimic(problem,keep_pct=0.1,pop_size=100,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.mimic(problem,keep_pct=0.2,pop_size=100 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.mimic(problem,keep_pct=0.5,pop_size=100 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.mimic(problem,keep_pct=0.1,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.mimic(problem,keep_pct=0.2,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_6, best_fitness_sa_6, fitness_curve_sa_6 = mlrose.mimic(problem,keep_pct=0.5,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlrose.mimic(problem,keep_pct=0.1,pop_size=500 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_8, best_fitness_sa_8, fitness_curve_sa_8 = mlrose.mimic(problem,keep_pct=0.5,pop_size=500 ,max_attempts = 100, max_iters=10000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa_1,label='0.1 / 100')
	plt.plot(fitness_curve_sa_2,label='0.2 / 100')
	plt.plot(fitness_curve_sa_3,label='0.5 / 100')
	plt.plot(fitness_curve_sa_4,label='0.1 / 200')
	plt.plot(fitness_curve_sa_5,label='0.2 / 200')
	plt.plot(fitness_curve_sa_6,label='0.5 / 200')
	plt.plot(fitness_curve_sa_7,label='0.1 / 500')
	plt.plot(fitness_curve_sa_8,label='0.5 / 500')
	plt.title('4 Peaks MIMIC Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('4_peaks_mimic_analysis.png')

def four_peaks_ga():
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=100,max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.genetic_alg(problem,mutation_prob=0.2,pop_size=100 ,max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=100 ,max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=200 ,max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.genetic_alg(problem,mutation_prob=0.2,pop_size=200 ,max_attempts = 10000, max_iters=10000, curve=True)
	

	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_6, best_fitness_sa_6, fitness_curve_sa_6 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=200 ,max_attempts = 10000, max_iters=10000, curve=True)
	

	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=500 ,max_attempts = 10000, max_iters=10000, curve=True)

	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_8, best_fitness_sa_8, fitness_curve_sa_8 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=500 ,max_attempts = 10000, max_iters=10000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa_1,label='0.1 / 100')
	plt.plot(fitness_curve_sa_2,label='0.2 / 100')
	plt.plot(fitness_curve_sa_3,label='0.5 / 100')
	plt.plot(fitness_curve_sa_4,label='0.1 / 200')
	plt.plot(fitness_curve_sa_5,label='0.2 / 200')
	plt.plot(fitness_curve_sa_6,label='0.5 / 200')
	plt.plot(fitness_curve_sa_7,label='0.1 / 500')
	plt.plot(fitness_curve_sa_8,label='0.5 / 500')
	plt.title('4 Peaks GA Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('4_peaks_ga_analysis.png')


# 2. k-Color Problem 
def k_color_input_size():
	# TO DO: 
	# - Try different edge values
	# - Try different decay schedules values 
	# The fitness function should be as big as possible
	fitness_sa_arr = []
	fitness_rhc_arr = []
	fitness_ga_arr = []
	fitness_mimic_arr = []

	time_sa_arr = []
	time_rhc_arr = []
	time_ga_arr = []
	time_mimic_arr = []
	for n in range(5,28,2):
		edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5), (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)]
		fitness = mlrose.MaxKColor(edges[:n])
		problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize=True, max_val=2)
		init_state = np.random.randint(2,size=n)
		schedule = mlrose.ExpDecay()
		st = time.time()
		best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 1000, max_iters=5000, init_state = init_state, curve=True)
		end = time.time()
		sa_time = end-st

		st = time.time()
		best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts = 1000, max_iters=5000, init_state = init_state, curve=True)
		end = time.time()
		rhc_time = end-st

		st = time.time()
		best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts = 1000, max_iters=5000, curve=True)
		end = time.time()
		ga_time = end-st

		st = time.time()
		best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, pop_size=500,max_attempts = 100, max_iters=5000, curve=True)
		end = time.time()
		mimic_time = end-st

		fitness_sa_arr.append(best_fitness_sa)
		fitness_rhc_arr.append(best_fitness_rhc)
		fitness_ga_arr.append(best_fitness_ga)
		fitness_mimic_arr.append(best_fitness_mimic)

		time_sa_arr.append(sa_time)
		time_rhc_arr.append(rhc_time)
		time_ga_arr.append(ga_time)
		time_mimic_arr.append(mimic_time)

	fitness_sa_arr = np.array(fitness_sa_arr)
	fitness_rhc_arr = np.array(fitness_rhc_arr)
	fitness_ga_arr = np.array(fitness_ga_arr)
	fitness_mimic_arr = np.array(fitness_mimic_arr)

	time_sa_arr = np.array(time_sa_arr)
	time_rhc_arr = np.array(time_rhc_arr)
	time_ga_arr = np.array(time_ga_arr)
	time_mimic_arr = np.array(time_mimic_arr)

	plt.figure()
	plt.plot(np.arange(5,28,2),fitness_sa_arr,label='SA')
	plt.plot(np.arange(5,28,2),fitness_rhc_arr,label='RHC')
	plt.plot(np.arange(5,28,2),fitness_ga_arr,label='GA')
	plt.plot(np.arange(5,28,2),fitness_mimic_arr,label='MIMIC')
	plt.legend()
	plt.ylabel('Fitness Value')
	plt.xlabel('Input Size')
	plt.title('Fitness Value vs. Input Size (Max k-Coloring)')
	plt.savefig('k_color_input_size_fitness.png')

	plt.figure()
	plt.plot(np.arange(5,28,2),time_sa_arr,label='SA')
	plt.plot(np.arange(5,28,2),time_rhc_arr,label='RHC')
	plt.plot(np.arange(5,28,2),time_ga_arr,label='GA')
	plt.plot(np.arange(5,28,2),time_mimic_arr,label='MIMIC')
	plt.ylabel('Computation Time')
	plt.xlabel('Input Size')
	plt.legend()
	plt.title('Computation Time vs. Input Size (Max k-Coloring)')
	plt.savefig('k_color_input_size_computation.png')

def k_color_iterations(): 
	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()
	best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 100, max_iters = 4000, init_state = init_state, curve=True)
	best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts = 100, max_iters = 4000, init_state = init_state, curve=True)
	best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts = 100, max_iters = 4000, curve=True)
	best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,pop_size=500, max_attempts = 100, max_iters = 4000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa,label='SA')
	plt.plot(fitness_curve_rhc,label='RHC')
	plt.plot(fitness_curve_ga,label='GA')
	plt.plot(fitness_curve_mimic,label='MIMIC')
	plt.ylabel('Fitness Value')
	plt.xlabel('Number of Iterations')
	plt.legend()
	plt.title('Fitness Value vs. Number of Iterations (Max K-Color)')
	plt.savefig('k_color_iterations.png')
	return

def k_color_rhc():
	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()

	best_state_rhc_1, best_fitness_rhc_1, fitness_curve_rhc_1 = mlrose.random_hill_climb(problem,restarts = 0, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_2, best_fitness_rhc_2, fitness_curve_rhc_2 = mlrose.random_hill_climb(problem,restarts = 2, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_3, best_fitness_rhc_3, fitness_curve_rhc_3 = mlrose.random_hill_climb(problem,restarts = 4, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_4, best_fitness_rhc_4, fitness_curve_rhc_4 = mlrose.random_hill_climb(problem,restarts = 6, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_5, best_fitness_rhc_5, fitness_curve_rhc_5 = mlrose.random_hill_climb(problem,restarts = 8, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_6, best_fitness_rhc_6, fitness_curve_rhc_6 = mlrose.random_hill_climb(problem,restarts = 10, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)

	plt.figure()
	plt.plot(fitness_curve_rhc_1,label='restarts = 0')
	plt.plot(fitness_curve_rhc_2,label='restarts = 2')
	plt.plot(fitness_curve_rhc_3,label='restarts = 4')
	plt.plot(fitness_curve_rhc_4,label='restarts = 6')
	plt.plot(fitness_curve_rhc_5,label='restarts = 8')
	plt.plot(fitness_curve_rhc_6,label='restarts = 10')
	plt.title('k-Color RHC Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('k_color_rhc_analysis.png')

def k_color_ga():
	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()

	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=100,max_attempts = 10000, max_iters=4000, curve=True)
	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.genetic_alg(problem,mutation_prob=0.2,pop_size=100 ,max_attempts = 10000, max_iters=4000, curve=True)
	
	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=100 ,max_attempts = 10000, max_iters=40000, curve=True)
	
	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=200 ,max_attempts = 10000, max_iters=4000, curve=True)
	
	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.genetic_alg(problem,mutation_prob=0.2,pop_size=200 ,max_attempts = 10000, max_iters=4000, curve=True)
	

	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()
	best_state_sa_6, best_fitness_sa_6, fitness_curve_sa_6 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=200 ,max_attempts = 10000, max_iters=4000, curve=True)
	

	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()
	best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=500 ,max_attempts = 10000, max_iters=4000, curve=True)

	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	schedule = mlrose.ExpDecay()
	best_state_sa_8, best_fitness_sa_8, fitness_curve_sa_8 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=500 ,max_attempts = 10000, max_iters=4000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa_1,label='0.1 / 100')
	plt.plot(fitness_curve_sa_2,label='0.2 / 100')
	plt.plot(fitness_curve_sa_3,label='0.5 / 100')
	plt.plot(fitness_curve_sa_4,label='0.1 / 200')
	plt.plot(fitness_curve_sa_5,label='0.2 / 200')
	plt.plot(fitness_curve_sa_6,label='0.5 / 200')
	plt.plot(fitness_curve_sa_7,label='0.1 / 500')
	plt.plot(fitness_curve_sa_8,label='0.5 / 500')
	plt.legend()
	plt.title('k-color GA Analysis')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('k_color_ga_analysis.png')

def k_color_mimic():
	edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5)
			, (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (8, 9), (7, 6), (3, 9), (2, 7), (4, 1), (3, 8), (4, 1)
			, (1, 2), (1, 3), (1, 5), (2, 4), (3, 1), (3, 4), (4, 5), (2, 3), (3, 5), (2, 6), (2, 7), (2, 5), (4, 5), (5, 6)
			, (3, 4), (3, 5), (3, 7), (4, 6), (5, 3), (5, 6), (6, 7), (9, 10), (8, 7), (4, 10), (3, 8), (5, 2), (4, 9), (5, 2)
			,(2, 4), (2, 5), (2, 7), (3, 6), (4, 3), (4, 6), (5, 7), (3, 5), (4, 7), (3, 8), (3, 9), (3, 7), (5, 7), (6, 8)
			, (4, 6), (4, 7), (4, 9), (5, 8), (6, 5), (6, 8), (7, 9), (10, 12), (9, 9), (5, 12), (4, 10), (6, 4), (5, 11)
			, (6, 4), (3, 5), (3, 6), (3, 8), (4, 7), (5, 4), (5, 7), (6, 8), (4, 6), (5, 8), (4, 9), (4, 10), (4, 8), (6, 8)
			, (7, 9), (5, 7), (5, 8), (5, 10), (6, 9), (7, 6), (7, 9), (8, 10), (11, 13), (10, 10), (6, 13), (5, 11)
			, (7, 5), (6, 12), (7, 5)]
	fitness = mlrose.MaxKColor(edges)
	problem = mlrose.DiscreteOpt(length = 112, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2, size=112)
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.mimic(problem,keep_pct=0.1,pop_size=100,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.mimic(problem,keep_pct=0.2,pop_size=100 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.mimic(problem,keep_pct=0.5,pop_size=100 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.mimic(problem,keep_pct=0.1,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.mimic(problem,keep_pct=0.2,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_6, best_fitness_sa_6, fitness_curve_sa_6 = mlrose.mimic(problem,keep_pct=0.5,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlrose.mimic(problem,keep_pct=0.1,pop_size=500 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_8, best_fitness_sa_8, fitness_curve_sa_8 = mlrose.mimic(problem,keep_pct=0.5,pop_size=500 ,max_attempts = 100, max_iters=10000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa_1,label='0.1 / 100')
	plt.plot(fitness_curve_sa_2,label='0.2 / 100')
	plt.plot(fitness_curve_sa_3,label='0.5 / 100')
	plt.plot(fitness_curve_sa_4,label='0.1 / 200')
	plt.plot(fitness_curve_sa_5,label='0.2 / 200')
	plt.plot(fitness_curve_sa_6,label='0.5 / 200')
	plt.plot(fitness_curve_sa_7,label='0.1 / 500')
	plt.plot(fitness_curve_sa_8,label='0.5 / 500')
	plt.title('k-Color MIMIC Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('k_color_mimic_analysis.png')



# 3. Continous-Peaks Problem
def continuous_peaks_input_size():
	# TO DO: 
	# - Try different t_pct values --> Threshold Parameter
	# - Try different decay schedules 
	# The fitness function should be as big as possible
	fitness_sa_arr = []
	fitness_rhc_arr = []
	fitness_ga_arr = []
	fitness_mimic_arr = []

	time_sa_arr = []
	time_rhc_arr = []
	time_ga_arr = []
	time_mimic_arr = []
	for n in range(5,120,20):
		fitness = mlrose.ContinuousPeaks(t_pct=0.15)
		problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize=True, max_val=2)
		init_state = np.random.randint(2,size=n)
		schedule = mlrose.ExpDecay()
		st = time.time()
		best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 1000, max_iters=10000, init_state = init_state, curve=True)
		end = time.time()
		sa_time = end-st
		fitness_sa_arr.append(best_fitness_sa)
		time_sa_arr.append(sa_time)

	for n in range(5,120,20):
		fitness = mlrose.ContinuousPeaks(t_pct=0.15)
		problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize=True, max_val=2)
		init_state = np.random.randint(2,size=n)
		schedule = mlrose.ExpDecay()
		st = time.time()
		best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts = 1000, max_iters=10000, init_state = init_state, curve=True)
		end = time.time()
		rhc_time = end-st
		fitness_rhc_arr.append(best_fitness_rhc)
		time_rhc_arr.append(rhc_time)


	for n in range(5,120,20):
		fitness = mlrose.ContinuousPeaks(t_pct=0.15)
		problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize=True, max_val=2)
		init_state = np.random.randint(2,size=n)
		schedule = mlrose.ExpDecay()
		st = time.time()
		best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts = 1000, max_iters=10000, curve=True)
		end = time.time()
		ga_time = end-st
		fitness_ga_arr.append(best_fitness_ga)
		time_ga_arr.append(ga_time)

	for n in range(5,120,20):
		fitness = mlrose.ContinuousPeaks(t_pct=0.15)
		problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize=True, max_val=2)
		init_state = np.random.randint(2,size=n)
		schedule = mlrose.ExpDecay()
		st = time.time()
		best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,pop_size=500, max_attempts = 100, max_iters=10000, curve=True)
		end = time.time()
		mimic_time = end-st
		fitness_mimic_arr.append(best_fitness_mimic)
		time_mimic_arr.append(mimic_time)

	fitness_sa_arr = np.array(fitness_sa_arr)
	fitness_rhc_arr = np.array(fitness_rhc_arr)
	fitness_ga_arr = np.array(fitness_ga_arr)
	fitness_mimic_arr = np.array(fitness_mimic_arr)

	time_sa_arr = np.array(time_sa_arr)
	time_rhc_arr = np.array(time_rhc_arr)
	time_ga_arr = np.array(time_ga_arr)
	time_mimic_arr = np.array(time_mimic_arr)
	plt.figure()
	plt.plot(np.arange(5,120,20),fitness_sa_arr,label='SA')
	plt.plot(np.arange(5,120,20),fitness_rhc_arr,label='RHC')
	plt.plot(np.arange(5,120,20),fitness_ga_arr,label='GA')
	plt.plot(np.arange(5,120,20),fitness_mimic_arr,label='MIMIC')
	plt.ylabel('Fitness Value')
	plt.xlabel('Input Size')
	plt.legend()
	plt.title('Fitness Value vs. Input Size (Continuous Peaks)')
	plt.savefig('cont_peaks_input_size_fitness.png')

	plt.figure()
	plt.plot(np.arange(5,120,20),time_sa_arr,label='SA')
	plt.plot(np.arange(5,120,20),time_rhc_arr,label='RHC')
	plt.plot(np.arange(5,120,20),time_ga_arr,label='GA')
	plt.plot(np.arange(5,120,20),time_mimic_arr,label='MIMIC')
	plt.legend()
	plt.ylabel('Computation Time')
	plt.xlabel('Input Size')
	plt.title('Computation Time vs. Input Size (Continuous Peaks)')
	plt.savefig('cont_peaks_input_size_computation.png')

def continuous_peaks_iterations():
	# TO DO: 
	# - Try different t_pct values --> Threshold Parameter
	# - Try different decay schedules 
	# The fitness function should be as big as possible
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size = 100)
	schedule = mlrose.ExpDecay()

	best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts = 1000, max_iters=10000, init_state = init_state, curve=True)
	best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts = 1000, max_iters=10000, curve=True)
	best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,pop_size=500, max_attempts = 100, max_iters=10000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa,label='SA')
	plt.plot(fitness_curve_rhc,label='RHC')
	plt.plot(fitness_curve_ga,label='GA')
	plt.plot(fitness_curve_mimic,label='MIMIC')
	plt.legend()
	plt.ylabel('Fitness Value')
	plt.xlabel('Number of Iterations')
	plt.title('Fitness Value vs. Number of Iterations (Continuous Peaks)')
	plt.savefig('cont_peaks_iterations.png')
	return

def continuous_peaks_mimic():
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.mimic(problem,keep_pct=0.1,pop_size=100,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.mimic(problem,keep_pct=0.2,pop_size=100 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.mimic(problem,keep_pct=0.5,pop_size=100 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.mimic(problem,keep_pct=0.1,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.mimic(problem,keep_pct=0.2,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_6, best_fitness_sa_6, fitness_curve_sa_6 = mlrose.mimic(problem,keep_pct=0.5,pop_size=200 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlrose.mimic(problem,keep_pct=0.1,pop_size=500 ,max_attempts = 100, max_iters=10000, curve=True)
	best_state_sa_8, best_fitness_sa_8, fitness_curve_sa_8 = mlrose.mimic(problem,keep_pct=0.5,pop_size=500 ,max_attempts = 100, max_iters=10000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa_1,label='0.1 / 100')
	plt.plot(fitness_curve_sa_2,label='0.2 / 100')
	plt.plot(fitness_curve_sa_3,label='0.5 / 100')
	plt.plot(fitness_curve_sa_4,label='0.1 / 200')
	plt.plot(fitness_curve_sa_5,label='0.2 / 200')
	plt.plot(fitness_curve_sa_6,label='0.5 / 200')
	plt.plot(fitness_curve_sa_7,label='0.1 / 500')
	plt.plot(fitness_curve_sa_8,label='0.5 / 500')
	plt.title('Continuous Peaks MIMIC Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('cont_peaks_mimic_analysis.png')

def continuous_peaks_t(): 
	fitness = mlrose.ContinuousPeaks(t_pct=0.5)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)


	fitness = mlrose.ContinuousPeaks(t_pct=0.25)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	fitness = mlrose.ContinuousPeaks(t_pct=0.35)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.45)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	plt.figure()
	plt.plot(fitness_curve_sa_1,label='t_pct = 0.5')
	plt.plot(fitness_curve_sa_2,label='t_pct = 0.15')
	plt.plot(fitness_curve_sa_3,label='t_pct = 0.25')
	plt.plot(fitness_curve_sa_4,label='t_pct = 0.35')
	plt.plot(fitness_curve_sa_5,label='t_pct = 0.45')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.title('Fitness Value vs. Number of Iterations (Continuous Peaks - SA)')
	plt.savefig('cont_peaks_t_pct_SA.png')

	fitness = mlrose.ContinuousPeaks(t_pct=0.5)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	fitness = mlrose.ContinuousPeaks(t_pct=0.25)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	fitness = mlrose.ContinuousPeaks(t_pct=0.35)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.45)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.random_hill_climb(problem, max_attempts = 10000, max_iters=10000, init_state = init_state, curve=True)

	plt.figure()
	plt.plot(fitness_curve_sa_1,label='t_pct = 0.5')
	plt.plot(fitness_curve_sa_2,label='t_pct = 0.15')
	plt.plot(fitness_curve_sa_3,label='t_pct = 0.25')
	plt.plot(fitness_curve_sa_4,label='t_pct = 0.35')
	plt.plot(fitness_curve_sa_5,label='t_pct = 0.45')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.title('Fitness Value vs. Number of Iterations (Continuous Peaks - RHC)')
	plt.savefig('cont_peaks_t_pct_RHC.png')

	fitness = mlrose.ContinuousPeaks(t_pct=0.5)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)

	fitness = mlrose.ContinuousPeaks(t_pct=0.25)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)

	fitness = mlrose.ContinuousPeaks(t_pct=0.35)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.45)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.genetic_alg(problem, max_attempts = 10000, max_iters=10000, curve=True)

	plt.figure()
	plt.plot(fitness_curve_sa_1,label='t_pct = 0.5')
	plt.plot(fitness_curve_sa_2,label='t_pct = 0.15')
	plt.plot(fitness_curve_sa_3,label='t_pct = 0.25')
	plt.plot(fitness_curve_sa_4,label='t_pct = 0.35')
	plt.plot(fitness_curve_sa_5,label='t_pct = 0.45')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.title('Fitness Value vs. Number of Iterations (Continuous Peaks - GA)')
	plt.savefig('cont_peaks_t_pct_GA.png')

def continuous_peaks_rhc():
	fitness = mlrose.ContinuousPeaks(t_pct = 0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_rhc_1, best_fitness_rhc_1, fitness_curve_rhc_1 = mlrose.random_hill_climb(problem,restarts = 0, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_2, best_fitness_rhc_2, fitness_curve_rhc_2 = mlrose.random_hill_climb(problem,restarts = 2, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_3, best_fitness_rhc_3, fitness_curve_rhc_3 = mlrose.random_hill_climb(problem,restarts = 4, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_4, best_fitness_rhc_4, fitness_curve_rhc_4 = mlrose.random_hill_climb(problem,restarts = 6, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_5, best_fitness_rhc_5, fitness_curve_rhc_5 = mlrose.random_hill_climb(problem,restarts = 8, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)
	best_state_rhc_6, best_fitness_rhc_6, fitness_curve_rhc_6 = mlrose.random_hill_climb(problem,restarts = 10, max_attempts = 1000, max_iters=1000, init_state = init_state, curve=True)

	plt.figure()
	plt.plot(fitness_curve_rhc_1,label='restarts = 0')
	plt.plot(fitness_curve_rhc_2,label='restarts = 2')
	plt.plot(fitness_curve_rhc_3,label='restarts = 4')
	plt.plot(fitness_curve_rhc_4,label='restarts = 6')
	plt.plot(fitness_curve_rhc_5,label='restarts = 8')
	plt.plot(fitness_curve_rhc_6,label='restarts = 10')
	plt.title('Continuous Peaks RHC Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('cont_peaks_rhc_analysis.png')

def continuous_peaks_ga():
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_1, best_fitness_sa_1, fitness_curve_sa_1 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=100,max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_2, best_fitness_sa_2, fitness_curve_sa_2 = mlrose.genetic_alg(problem,mutation_prob=0.2,pop_size=100 ,max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_3, best_fitness_sa_3, fitness_curve_sa_3 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=100 ,max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_4, best_fitness_sa_4, fitness_curve_sa_4 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=200 ,max_attempts = 10000, max_iters=10000, curve=True)
	
	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_5, best_fitness_sa_5, fitness_curve_sa_5 = mlrose.genetic_alg(problem,mutation_prob=0.2,pop_size=200 ,max_attempts = 10000, max_iters=10000, curve=True)

	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_6, best_fitness_sa_6, fitness_curve_sa_6 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=200 ,max_attempts = 10000, max_iters=10000, curve=True)
	

	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_7, best_fitness_sa_7, fitness_curve_sa_7 = mlrose.genetic_alg(problem,mutation_prob=0.1,pop_size=500 ,max_attempts = 10000, max_iters=10000, curve=True)

	fitness = mlrose.ContinuousPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
	init_state = np.random.randint(2,size=100)
	schedule = mlrose.ExpDecay()
	best_state_sa_8, best_fitness_sa_8, fitness_curve_sa_8 = mlrose.genetic_alg(problem,mutation_prob=0.5,pop_size=500 ,max_attempts = 10000, max_iters=10000, curve=True)
	
	plt.figure()
	plt.plot(fitness_curve_sa_1,label='0.1 / 100')
	plt.plot(fitness_curve_sa_2,label='0.2 / 100')
	plt.plot(fitness_curve_sa_3,label='0.5 / 100')
	plt.plot(fitness_curve_sa_4,label='0.1 / 200')
	plt.plot(fitness_curve_sa_5,label='0.2 / 200')
	plt.plot(fitness_curve_sa_6,label='0.5 / 200')
	plt.plot(fitness_curve_sa_7,label='0.1 / 500')
	plt.plot(fitness_curve_sa_8,label='0.5 / 500')
	plt.legend()
	plt.title('Continuous Peaks GA Analysis')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Fitness Value')
	plt.savefig('cont_peaks_ga_analysis.png')

# Use Randomized Optimization for Neural Network Weights
def neural_net_optimization():
	sklearn_data = datasets.load_breast_cancer()
	x, y = sklearn_data.data, sklearn_data.target
	# Normalize the data
	x = preprocessing.scale(x)
	# Split the data into training and testing 
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=3169)
	
	# Data is ready, so begin the neural network analysis for each optimization algorithm
	# First do an analysis of iterations 
	train_acc_rhc = []
	test_acc_rhc = []
	train_acc_sa = []
	test_acc_sa = []
	train_acc_ga = []
	test_acc_ga = []
	time_rhc = []
	time_sa = []
	time_ga = []
	for i in range(1000,120000,5000):
		print(i)
		nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='random_hill_climb', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169)

		st = time.time()
		nn_model_rhc.fit(X_train, y_train)
		end = time.time()
		ft = end - st

		# Predict labels for train set and assess accuracy
		y_train_pred_rhc = nn_model_rhc.predict(X_train)
		y_train_accuracy_rhc = accuracy_score(y_train, y_train_pred_rhc)
		train_acc_rhc.append(y_train_accuracy_rhc)

		# Predict labels for test set and assess accuracy
		y_test_pred_rhc = nn_model_rhc.predict(X_test)
		y_test_accuracy_rhc = accuracy_score(y_test, y_test_pred_rhc)
		test_acc_rhc.append(y_test_accuracy_rhc)
		time_rhc.append(ft)
		print('RHC Completed!')

		nn_model_sa = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='simulated_annealing', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169)

		st = time.time()
		nn_model_sa.fit(X_train, y_train)
		end = time.time()
		ft = end - st

		# Predict labels for train set and assess accuracy
		y_train_pred_sa = nn_model_sa.predict(X_train)
		y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
		train_acc_sa.append(y_train_accuracy_sa)

		# Predict labels for test set and assess accuracy
		y_test_pred_sa = nn_model_sa.predict(X_test)
		y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
		test_acc_sa.append(y_test_accuracy_sa)
		time_sa.append(ft)
		print('SA completed!')

		nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='genetic_alg', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169)

		st = time.time()
		nn_model_ga.fit(X_train, y_train)
		end = time.time()
		ft = end - st

		# Predict labels for train set and assess accuracy
		y_train_pred_ga = nn_model_ga.predict(X_train)
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		train_acc_ga.append(y_train_accuracy_ga)

		# Predict labels for test set and assess accuracy
		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_ga.append(y_test_accuracy_ga)
		time_ga.append(ft)
		print('GA completed!')

	plt.figure()
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_rhc),label='RHC')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_sa),label='SA')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_ga),label='GA')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Test Accuracy')
	plt.title('Test Accuracy vs. Iterations for Optimization Algorithms')
	plt.legend()
	plt.savefig('nn_test_iterations.png')

	plt.figure()
	plt.plot(np.arange(1000,120000,5000),np.array(train_acc_rhc),label='RHC')
	plt.plot(np.arange(1000,120000,5000),np.array(train_acc_sa),label='SA')
	plt.plot(np.arange(1000,120000,5000),np.array(train_acc_ga),label='GA')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Training Accuracy')
	plt.title('Training Accuracy vs. Iterations for Optimization Algorithms')
	plt.legend()
	plt.savefig('nn_train_iterations.png')

	plt.figure()
	plt.plot(np.arange(1000,120000,5000),np.array(time_rhc),label='RHC')
	plt.plot(np.arange(1000,120000,5000),np.array(time_sa),label='SA')
	plt.plot(np.arange(1000,120000,5000),np.array(time_ga),label='GA')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Training Time')
	plt.title('Computation Time vs. Iterations for Optimization Algorithms')
	plt.legend()
	plt.savefig('nn_computation.png')
	return

	
	# Now, do an analysis of restarts (RHC), Schedule (SA), and Mutation Probability (GA)
	train_acc_rhc = []
	test_acc_rhc = []
	for i in range(0,10):
		nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='random_hill_climb', 
                                 	max_iters = 100000, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000,  restarts=i)

		nn_model_rhc.fit(X_train, y_train)

		# Predict labels for train set and assess accuracy
		y_train_pred_rhc = nn_model_rhc.predict(X_train)
		y_train_accuracy_rhc = accuracy_score(y_train, y_train_pred_rhc)
		train_acc_rhc.append(y_train_accuracy_rhc)

		# Predict labels for test set and assess accuracy
		y_test_pred_rhc = nn_model_rhc.predict(X_test)
		y_test_accuracy_rhc = accuracy_score(y_test, y_test_pred_rhc)
		test_acc_rhc.append(y_test_accuracy_rhc)

	plt.figure()
	plt.plot(np.arange(0,10),np.array(test_acc_rhc),label='RHC')
	plt.xlabel('Number of Random Restarts')
	plt.ylabel('Test Accuracy')
	plt.title('Test Accuracy vs. Restarts on RHC')
	plt.legend()
	plt.savefig('nn_test_restarts.png')


	plt.figure()
	plt.plot(np.arange(0,10),np.array(train_acc_rhc),label='RHC')
	plt.xlabel('Number of Random Restarts')
	plt.ylabel('Training Accuracy')
	plt.title('Training Accuracy vs. Restarts on RHC')
	plt.legend()
	plt.savefig('nn_train_restarts.png')

	train_acc_sa = []
	test_acc_sa = []
	for schedule in [mlrose.GeomDecay(),mlrose.ExpDecay(),mlrose.ArithDecay()]:
		nn_model_sa = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='simulated_annealing', 
                                 	max_iters = 100000, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000,  schedule=schedule)

		nn_model_sa.fit(X_train, y_train)

		# Predict labels for train set and assess accuracy
		y_train_pred_sa = nn_model_sa.predict(X_train)
		y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
		train_acc_sa.append(y_train_accuracy_sa)

		# Predict labels for test set and assess accuracy
		y_test_pred_sa = nn_model_sa.predict(X_test)
		y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
		test_acc_sa.append(y_test_accuracy_sa)

	plt.figure()
	plt.plot(['GeomDecay','ExpDecay','ArithDecay'],np.array(test_acc_sa),label='SA')
	plt.xlabel('Different Schedule Functions')
	plt.ylabel('Test Accuracy')
	plt.title('Test Accuracy vs. Scheduling on SA')
	plt.legend()
	plt.savefig('nn_test_scheduling.png')


	plt.figure()
	plt.plot(['GeomDecay','ExpDecay','ArithDecay'],np.array(train_acc_sa),label='SA')
	plt.xlabel('Different Schedule Functions')
	plt.ylabel('Training Accuracy')
	plt.title('Training Accuracy vs. Scheduling on SA')
	plt.legend()
	plt.savefig('nn_train_scheduling.png')


	train_acc_ga = []
	test_acc_ga = []

	nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                             	algorithm ='genetic_alg', 
                             	max_iters = 100000, bias = True, is_classifier = True, 
                             	learning_rate = 1e-04, early_stopping = True, 
                             	max_attempts = 100,  mutation_prob=0.1, pop_size=100)
	nn_model_ga.fit(X_train, y_train)
	y_train_pred_ga = nn_model_ga.predict(X_train)
	y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
	train_acc_ga.append(y_train_accuracy_ga)
	y_test_pred_ga = nn_model_ga.predict(X_test)
	y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
	test_acc_ga.append(y_test_accuracy_ga)

	nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                             	algorithm ='genetic_alg', 
                             	max_iters = 100000, bias = True, is_classifier = True, 
                             	learning_rate = 1e-04, early_stopping = True, 
                             	max_attempts = 100,  mutation_prob=0.2, pop_size=100)
	nn_model_ga.fit(X_train, y_train)
	y_train_pred_ga = nn_model_ga.predict(X_train)
	y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
	train_acc_ga.append(y_train_accuracy_ga)
	y_test_pred_ga = nn_model_ga.predict(X_test)
	y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
	test_acc_ga.append(y_test_accuracy_ga)

	nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                             	algorithm ='genetic_alg', 
                             	max_iters = 100000, bias = True, is_classifier = True, 
                             	learning_rate = 1e-04, early_stopping = True, 
                             	max_attempts = 100,  mutation_prob=0.3, pop_size=100)
	nn_model_ga.fit(X_train, y_train)
	y_train_pred_ga = nn_model_ga.predict(X_train)
	y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
	train_acc_ga.append(y_train_accuracy_ga)
	y_test_pred_ga = nn_model_ga.predict(X_test)
	y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
	test_acc_ga.append(y_test_accuracy_ga)

	nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                             	algorithm ='genetic_alg', 
                             	max_iters = 100000, bias = True, is_classifier = True, 
                             	learning_rate = 1e-04, early_stopping = True, 
                             	max_attempts = 100,  mutation_prob=0.1, pop_size=200)
	nn_model_ga.fit(X_train, y_train)
	y_train_pred_ga = nn_model_ga.predict(X_train)
	y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
	train_acc_ga.append(y_train_accuracy_ga)
	y_test_pred_ga = nn_model_ga.predict(X_test)
	y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
	test_acc_ga.append(y_test_accuracy_ga)

	nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                             	algorithm ='genetic_alg', 
                             	max_iters = 100000, bias = True, is_classifier = True, 
                             	learning_rate = 1e-04, early_stopping = True, 
                             	max_attempts = 100,  mutation_prob=0.2, pop_size=200)
	nn_model_ga.fit(X_train, y_train)
	y_train_pred_ga = nn_model_ga.predict(X_train)
	y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
	train_acc_ga.append(y_train_accuracy_ga)
	y_test_pred_ga = nn_model_ga.predict(X_test)
	y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
	test_acc_ga.append(y_test_accuracy_ga)

	nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                             	algorithm ='genetic_alg', 
                             	max_iters = 100000, bias = True, is_classifier = True, 
                             	learning_rate = 1e-04, early_stopping = True, 
                             	max_attempts = 100,  mutation_prob=0.3, pop_size=200)
	nn_model_ga.fit(X_train, y_train)
	y_train_pred_ga = nn_model_ga.predict(X_train)
	y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
	train_acc_ga.append(y_train_accuracy_ga)
	y_test_pred_ga = nn_model_ga.predict(X_test)
	y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
	test_acc_ga.append(y_test_accuracy_ga)

	plt.figure()
	plt.plot(np.array([0.1,0.2,0.3,0.1,0.2,0.3]),np.array(test_acc_ga),label='GA')
	plt.xlabel('Different Mutation Probabilities')
	plt.ylabel('Test Accuracy')
	plt.title('Test Accuracy vs. Mutation Prob and Population Size (100,200)')
	plt.legend()
	plt.savefig('nn_test_ga_analysis.png')


	plt.figure()
	plt.plot(np.array(train_acc_ga),label='GA')
	plt.xlabel('Different Mutation Probabilities and Population Size')
	plt.ylabel('Training Accuracy')
	plt.title('Training Accuracy vs. Mutation Prob and Population Size')
	plt.legend()
	plt.savefig('nn_train_ga_analysis.png')

def nn_analysis():
	sklearn_data = datasets.load_breast_cancer()
	x, y = sklearn_data.data, sklearn_data.target
	# Normalize the data
	x = preprocessing.scale(x)
	# Split the data into training and testing 
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=3169)

	
	test_acc_sa_1 = []
	for i in range(1000,120000,5000):
		print(i)
		nn_model_sa = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='simulated_annealing', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	schedule = mlrose.GeomDecay())

		nn_model_sa.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_sa = nn_model_sa.predict(X_train)
		y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
		# Predict labels for test set and assess accuracy
		y_test_pred_sa = nn_model_sa.predict(X_test)
		y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
		test_acc_sa_1.append(y_test_accuracy_sa)

	test_acc_sa_2 = []
	for i in range(1000,120000,5000):
		nn_model_sa = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='simulated_annealing', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	schedule = mlrose.ExpDecay())

		nn_model_sa.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_sa = nn_model_sa.predict(X_train)
		y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
		# Predict labels for test set and assess accuracy
		y_test_pred_sa = nn_model_sa.predict(X_test)
		y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
		test_acc_sa_2.append(y_test_accuracy_sa)

	test_acc_sa_3 = []
	for i in range(1000,120000,5000):
		nn_model_sa = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='simulated_annealing', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	schedule = mlrose.ArithDecay())

		nn_model_sa.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_sa = nn_model_sa.predict(X_train)
		y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
		# Predict labels for test set and assess accuracy
		y_test_pred_sa = nn_model_sa.predict(X_test)
		y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
		test_acc_sa_3.append(y_test_accuracy_sa)

	plt.figure()
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_sa_1),label='Geometric Decay')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_sa_2),label='Exponential Decay')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_sa_3),label='Arithmetic Decay')
	plt.title('Neural Network SA Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Testing Accuracy')
	plt.savefig('NN_SA_analysis.png')
	

	test_acc_backprop = []
	for i in range(10,1000,2):
		nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='gradient_descent', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	pop_size=500,mutation_prob=0.5)

		nn_model_ga.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_ga = nn_model_ga.predict(X_train)
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		# Predict labels for test set and assess accuracy
		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_backprop.append(y_test_accuracy_ga)
	
	plt.figure()
	plt.plot(range(10,1000,2),np.array(test_acc_backprop))
	plt.title('Neural Network Backpropagation')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Testing Accuracy')
	plt.savefig('nn_backprop.png')
	
	# Data is ready, so begin the neural network analysis for each optimization algorithm
	# First do an analysis of iterations 
	test_acc_ga_1 = []
	for i in range(1000,120000,5000):
		print(i)
		nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='genetic_alg', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	pop_size=100,mutation_prob=0.1)

		nn_model_ga.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_ga = nn_model_ga.predict(X_train)
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		# Predict labels for test set and assess accuracy
		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_ga_1.append(y_test_accuracy_ga)

	test_acc_ga_2 = []
	for i in range(1000,120000,5000):
		nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='genetic_alg', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	pop_size=200,mutation_prob=0.1)

		nn_model_ga.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_ga = nn_model_ga.predict(X_train)
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		# Predict labels for test set and assess accuracy
		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_ga_2.append(y_test_accuracy_ga)

	test_acc_ga_3 = []
	for i in range(1000,120000,5000):
		nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='genetic_alg', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	pop_size=500,mutation_prob=0.1)

		nn_model_ga.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_ga = nn_model_ga.predict(X_train)
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		# Predict labels for test set and assess accuracy
		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_ga_3.append(y_test_accuracy_ga)

	test_acc_ga_4 = []
	for i in range(1000,120000,5000):
		nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='genetic_alg', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	pop_size=100,mutation_prob=0.5)

		nn_model_ga.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_ga = nn_model_ga.predict(X_train)
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		# Predict labels for test set and assess accuracy
		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_ga_4.append(y_test_accuracy_ga)

	test_acc_ga_5 = []
	for i in range(1000,120000,5000):
		nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='genetic_alg', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	pop_size=200,mutation_prob=0.5)

		nn_model_ga.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_ga = nn_model_ga.predict(X_train)
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		# Predict labels for test set and assess accuracy
		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_ga_5.append(y_test_accuracy_ga)

	test_acc_ga_6 = []
	for i in range(1000,120000,5000):
		nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 	algorithm ='genetic_alg', 
                                 	max_iters = i, bias = True, is_classifier = True, 
                                 	learning_rate = 1e-03, early_stopping = True, 
                                 	max_attempts = 1000, random_state = 3169, 
                                 	pop_size=500,mutation_prob=0.5)

		nn_model_ga.fit(X_train, y_train)
		# Predict labels for train set and assess accuracy
		y_train_pred_ga = nn_model_ga.predict(X_train)
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		# Predict labels for test set and assess accuracy
		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_ga_6.append(y_test_accuracy_ga)

	plt.figure()
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_ga_1),label='0.1 / 100')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_ga_2),label='0.2 / 100')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_ga_3),label='0.5 / 100')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_ga_4),label='0.1 / 500')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_ga_5),label='0.2 / 500')
	plt.plot(np.arange(1000,120000,5000),np.array(test_acc_ga_6),label='0.5 / 500')
	plt.title('Neural Network GA Analysis')
	plt.legend()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Testing Accuracy')
	plt.savefig('NN_GA_analysis.png')


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### MAIN FUNCTION FOR ASSIGNMENT 2 ########

### FOUR PEAKS ######
four_peaks_input_size()
four_peaks_iterations()
four_peaks_t()
four_peaks_ga()
four_peaks_rhc()
four_peaks_mimic()

#neural_net_optimization()

### CONTINUOUS PEAKS #####
continuous_peaks_input_size()
continuous_peaks_iterations()
continuous_peaks_t()
continuous_peaks_ga()
continuous_peaks_rhc()
continuous_peaks_mimic()

### K-COLORING #####
k_color_input_size()
k_color_iterations()
k_color_ga()
k_color_rhc()
k_color_mimic()

### NEURAL NETWORK WEIGHT OPTIMIZATION #####
neural_net_optimization()
nn_analysis()

