import os
import csv
import numpy as np
from multiprocessing import Pool
import random as rand
from collections import defaultdict
import scipy.spatial.distance as sci
import math
import sys
import copy
import json
import random
from geopy.distance import geodesic
import pandas as pd
import datetime



def get_graph_data(data_file, directed = False, q = 6):
	#loads the graphs for the graph application
    edges = defaultdict(set)
    edge_weights = dict()
    vertex_weights = dict()
    elements = set()
    with open(data_file, 'r') as f:
        for line in f:
            [e_0, e_1] = [int(e) for e in line.replace("\n","").split()]
            elements.add(e_0)
            elements.add(e_1)
            edges[e_0].add(e_1)
            ew = random.uniform(1,20)
            edge_weights[(e_0,e_1)] = ew
            if not directed:
                edges[e_1].add(e_0)
                edge_weights[(e_1,e_0)] = ew
    costs = dict()
    for e in elements:
        #costs[e] = random.uniform(1, 30)
        #vertex_weights[e] = random.uniform(1, 20)
        vertex_weights[e] = 1
        #costs[e] = vertex_weights[e] + random.uniform(len(edges[e]) * 1, len(edges[e]) * 20)
        costs[e] = 1 + max(len(edges[e]) - q , 0)


    elements = list(elements)
    #random.shuffle(elements)

    data_dict = {'edge_weights':edge_weights, 'vertex_weights': vertex_weights, 'edges': edges, 'elements': elements, 'nb_elements':len(elements), 'costs': costs}
    return data_dict


class VertexCover():
    def __init__(self, dataset):
        self.dataset = dataset
        self.oracle_calls = 0
    def evaluate(self, S):
        if isinstance(S, list):
            nodes = set(S)
            for e in S:
                for nei in self.dataset['edges'][e]:
                    nodes.add(nei)
        else:
            nodes = set()
            nodes.add(S)
            for nei in self.dataset['edges'][S]:
                nodes.add(nei)

        val = 0
        for e in nodes:
            val += self.dataset['vertex_weights'][e]
        self.oracle_calls += 1
        return val, nodes


    def marginal_gain(self, e, S, current_val = None, auxiliary = None):
        if current_val == None or auxiliary == None:
            current_val, auxiliary = self.evaluate(S)
        nodes = set()
        nodes.add(e)
        gain = 0
        new_auxiliary = set()
        if e not in auxiliary:
            gain += self.dataset['vertex_weights'][e]
            new_auxiliary.add(e)

        for nei in self.dataset['edges'][e]:
            if nei not in auxiliary:
                gain += self.dataset['vertex_weights'][nei]
                new_auxiliary.add(nei)
        self.oracle_calls += 1
        return gain, new_auxiliary


    def add_one_element(self, e, S, current_val = None, auxiliary = None):
        if e in S:
            return 0
        if current_val == None or auxiliary == None:
            current_val, auxiliary = self.evaluate(S)
        nodes = set()
        nodes.add(e)
        val = current_val
        if e not in auxiliary:
            val += self.dataset['vertex_weights'][e]
            auxiliary.add(e)

        for nei in self.dataset['edges'][e]:
            if nei not in auxiliary:
                val += self.dataset['vertex_weights'][nei]
                auxiliary.add(nei)
        self.oracle_calls += 1
        return val, auxiliary




def node_cost(dataset, S):
    if isinstance(S, list):
        val = 0
        for e in set(S):
            val += dataset['costs'][e]
        return val
    else:
        return dataset['costs'][S]

        

def vertex_cover(dataset, S):
    if isinstance(S, list):
        nodes = set(S)
        for e in S:
            for nei in dataset['edges'][e]:
                nodes.add(nei)
    else:
        nodes = set()
        nodes.add(S)
        for nei in dataset['edges'][S]:
            nodes.add(nei)

    val = 0
    for e in nodes:
        val += dataset['vertex_weights'][e]

    return val



def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power) 


def average_distance(dataset, S):
	#
	# calculated the average cost of a set of elements
	#
	if isinstance(S, int):
		S = [S]
	val = 0
	for e in S:
		val += dataset['distance'][e]
	return val / len(S)


def yelp_distance_cost(dataset, S):
	#
	# it calculates the cost of each element in the yelp dataset
	#
	if isinstance(S, list):
		val = 0
		for e in set(S):
			val += dataset['distance'][e]
		return val * dataset['reg_coef']
	else:
		return dataset['distance'][S] * dataset['reg_coef']

def get_movie_data(data_folder, lambdaa = 1.0, reg_coef = 1.0, rating_base = 10, 
	year_shift = 1990, recommended_genres=None, movie_lim=None, application = 1):
	#
	# it loads the movie-lens dataset
	#
	print('Importing data...', end='')

	data_mat = []
	genres = []
	titles = []
	years = []
	ratings = []

	movie_year_rating_file = os.path.join(data_folder, 'year_rating')
	year_rating_dict = json.load(open(movie_year_rating_file))

	if recommended_genres is None:
		recommended_genres = ['Adventure', 'Animation', 'Fantasy']
	if recommended_genres is "all":
		recommended_genres = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
							  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
							  'Thriller', 'War', 'Western']
	recommended_genres = set(recommended_genres)

	# import meta data from movies.csv
	movie_info_dict = {}
	movie_info_file = os.path.join(data_folder, 'movies.csv')
	with open(movie_info_file, 'r') as movie_info_csv:
		reader = csv.reader(movie_info_csv)
		header = next(reader)
		# go through all rows
		for row in reader:
			# put info into dictionary
			id = int(row[0])
			title = row[1]
			genre_list = row[2].split('|')
			movie_info_dict[id] = {'title': title, 'genres': genre_list}

	# import feature matrix from mvs.csv
	movie_feat_file = os.path.join(data_folder, 'mvs.csv')
	with open(movie_feat_file, 'r') as movie_feat_csv:

		# create reader
		reader = csv.reader(movie_feat_csv)
		header = next(reader)
		# go through all rows
		for row in reader:
			# put info into
			id = int(row[0])
			feat_vec = [float(x) for x in row[1:]]
			# check if this movie has at least one of the desired genres
			if recommended_genres.isdisjoint(movie_info_dict[id]['genres']):
				continue
			else:
				data_mat.append(feat_vec)
				titles.append(movie_info_dict[id]['title'])
				genres.append(list( set(movie_info_dict[id]['genres']) &  recommended_genres)) 
				years.append(year_rating_dict[str(id)][2])
				ratings.append(year_rating_dict[str(id)][3])
	# format movie feature matrix into

	data_mat = np.array(data_mat)
	if movie_lim is not None:
		if movie_lim < data_mat.shape[0]:
			data_mat = data_mat[:movie_lim, :]
			titles = titles[:movie_lim]
			genres = genres[:movie_lim]
			years = years[:movie_lim]
			ratings = ratings[:movie_lim]
	orig_rating = copy.copy(ratings)
	orig_years = copy.copy(years)

	M = sci.squareform(sci.pdist(data_mat, 'euclidean'))
	M = np.exp(-1 * lambdaa * M )

	print('data contains %d movies' % data_mat.shape[0])

	elements = [i for i in range(len(titles))]

	coef = len(elements) / 10.0
	tot = 0
	for e in elements:
		tot += abs(ratings[e] - rating_base)
	print(tot)
	for e in elements:
		ratings[e] = coef * abs(ratings[e] - rating_base) / tot

	tot = 0
	for e in elements:
		tot += abs(years[e] - year_shift)
	print(tot)
	for e in elements:
		years[e] = coef * abs(years[e] - year_shift) / tot

	if application == 1:
		costs = copy.copy(ratings)
		orig_cost = copy.copy(orig_rating)
	if application == 2:
		costs = copy.copy(years)
		orig_cost = copy.copy(orig_years)
	

	# return collected data
	data_dict = {'matrix': data_mat, 'genres': genres, 'titles': titles, 
	 'recommended_genres':recommended_genres, 
	'lambda': lambdaa,'sim_mat': M,
	'elements':elements, 'rating_base':rating_base, 'year_shift':year_shift,
	'years':years, 'ratings':ratings, 'reg_coef': reg_coef,
	'orig_rating':orig_rating, 'orig_years':orig_years, 
	'costs':costs, 'orig_cost':orig_cost}
	return data_dict


class LogDet():
	#
	# a monotone submodular function 
	#
	def __init__(self, dataset, alpha = 10):
		self.dataset = dataset
		self.alpha = alpha
		self.oracle_calls = 0

	def evaluate(self, S):
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if isinstance(S, list):
			S = list(set(S))

		n = len(S) 
		if n == 0:
			return 0, None
		M = np.identity(n) + self.alpha * self.dataset['sim_mat'][np.ix_(S,S)]
		self.oracle_calls += 1
		return math.log(np.linalg.det(M)), None

	def marginal_gain(self, e, S, current_val = None, auxiliary = None):
		if e in S:
			return 0, None
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if e in S:
			return 0.0, None
		if current_val == None:
			current_val, __ = self.evaluate(S)

		if isinstance(e, int):
			val, __ = self.evaluate(S + [e])
		else:
			val, __ = self.evaluate(S + e)
		return val - current_val, None


	def add_one_element(self, e, S, current_val = None, auxiliary = None):
		if e in S:
			return 0, None
		if isinstance(S, int):
			S = [S]
		if isinstance(e, list):
			return self.evaluate(S + e)
		else:
			return self.evaluate(S + [e])



def movie_ratings_cost(dataset, S):
	if isinstance(S, list) or isinstance(S, set):
		val = 0
		for e in set(S):
			val += dataset['ratings'][e]
		return val * dataset['reg_coef']
	else:
		return dataset['ratings'][S] * dataset['reg_coef']

def movie_years_cost(dataset, S):
	if isinstance(S, list) or isinstance(S, set):
		val = 0
		for e in set(S):
			val +=  dataset['years'][e]
		return val * dataset['reg_coef']
	else:
		return  dataset['years'][S] * dataset['reg_coef']
		

		
def average_rating(dataset, S):
	if isinstance(S, int):
		S = [S]
	val = 0
	for e in S:
		val += dataset['ratings'][e]
	return val / len(S)


def average_year(dataset, S):
	if isinstance(S, int):
		S = [S]
	val = 0
	for e in S:
		val += dataset['years'][e]
	return val / len(S)


class CandidateSol():
	#
	# This class is use to keep track of a solution
	# it keeps th order of elements added 
	#
	def __init__(self):
		self.S_list = []
		self.S_set = set()
		self.auxiliary = None

	def add_elements(self, e):
		if isinstance(e, int):
			if e not in self.S_set:
				self.S_set.add(e)
				self.S_list.append(e)
		else:
			set_e = set(e)
			for u in e:
				if u not in self.S_set:
					self.S_list.append(u)
			self.S_set.update(set_e)

	def remove_elements(self, e):
		if isinstance(e, int):
			if e in self.S_set:
				self.S_set.remove(e)
				self.S_list.remove(e)
		else:
			if len(e) > 0:
				set_e = set(e)
				self.S_set -= set_e
				current_S = copy.copy(self.S_list)
				self.S_list = [u for u in current_S if u not in set_e]

	def find_index(self, e):
		if e  not in self.S_set:
			return -1
		return self.S_list.index(e)


class KnapsackConstraint():
	def __init__(self, dataset, costs, max_elements):
		self.dataset = dataset
		self.costs = costs
		self.nb_elements = len(dataset['elements'])
		self.max_movies = max_elements
		self.nb_knapsacks = 1
		self.nb_matroids = 1
		self.max_cardinality = max_elements

	def is_feasible(self, S):
		if len(S) == 0:
			return True
		if not isinstance(S, set):
			S = set(S)
		if len(S) > self.max_movies:
			return False
		if round( sum([self.costs[e][0] for e in S]), 4) > 1:
			return False
		return True

	def is_add_feasible(self, e, S):
		if isinstance(S, int):
			S = [S]
		if not isinstance(e, list):
			return self.is_feasible(S + [e])
		else:
			return self.is_feasible(S + e)

	def check_costs(self, S):
		if len(S) == 0:
			return True
		if round( sum([self.costs[e][0] for e in S]), 4) > 1:
			return False
		return True

	def is_psystem_add_feasible(self, e, S):
		if isinstance(S, int):
			S = [S]
		if isinstance(e, int):
			e = [e]
		S_e = S + e
		S = set(S_e)
		if len(S) > self.max_movies:
			return False
		return True


def get_yelp_data(yelp_info_file, lambdaa = 1.0, reg_coef = 1.0, max_datapoints = 1000, nb_sample = 100):
	#
	# This function load the yelp data set
	#
	recommended_genres = ['Pittsburgh', 'Charlotte', 'Phoenix', 'Madison', 'Las Vegas', 'Edinburgh']
	print("lambdaaaa: ", lambdaa)

	distance = []
	state = []
	city = []
	address = []
	data_mat = []
	elements = []
	locations = []
	cnt = 0
	costs = defaultdict(list)
	with open(yelp_info_file, 'r') as yelp_info_csv:
		reader = csv.reader(yelp_info_csv)
		header = next(reader)
		for row in reader:
			if cnt < max_datapoints:
				city.append([row[0]])
				elements.append(cnt)
				distance.append([float(row[1]), float(row[2]), float(row[3])])
				costs[cnt] = [float(row[1]), float(row[2]), float(row[3])]
				state.append(row[7])
				address.append(row[8])
				vals = [float(val) for val in row[9:]]
				data_mat.append(vals)
				locations.append((float(row[4]), float(row[5])))
				cnt+=1


	data_mat = np.array(data_mat)
	#np.shape()

	M = sci.squareform(sci.pdist(data_mat, 'euclidean'))
	M = np.exp(-1 * lambdaa * M )

	nb_knapsacks = 0
	elements = elements[nb_sample:]

	M = M[:,:nb_sample]

	vals = [0,0,0]
	for e in elements:
		for i in range(3):
			vals[i] += costs[e][i]
	print(vals)
	rate = len(elements) / 20

	for e in elements:
		cost = costs[e]
		costs[e] = [rate * cost[i] / vals[i] for i in range(3)] 

	print(len(city))
	
	dataset ={}
	data_dict = {'matrix': data_mat, 'genres': city, 'titles': address, 
	 'recommended_genres':recommended_genres, 'nb_sample':nb_sample, 
	 'nb_elements': len(elements),
	'lambda': lambdaa,'sim_mat': M, 'nb_knapsacks':nb_knapsacks,
	'elements':elements, 'reg_coef': reg_coef, 'costs':costs, 'orig_distance':distance}
	return data_dict


class FacilityLocation():

	def __init__(self, dataset):
		self.dataset = dataset
		self.sim_mat = dataset['sim_mat']
		self.nb_sample = dataset['nb_sample']
		self.oracle_calls = 0

	def evaluate(self, S):
		self.oracle_calls += 1
		if isinstance(S, int):
			S = [S]

		if isinstance(S, set):
			S = list(S)

		if isinstance(S, list):
			S = list(set(S))

		if len(S) == 0:
			return 0, None
		#print(S, np.max(self.sim_mat[np.ix_(S, )], axis = 0))
		val  = np.sum(np.max(self.sim_mat[np.ix_(S, )], axis = 0))
		return val, None

	def marginal_gain(self, e, S, current_val = None, auxiliary = None):
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if e in S:
			return 0.0, None
		if current_val == None:
			current_val, __ = self.evaluate(S)

		if isinstance(e, int):
			val, __ = self.evaluate(S + [e])
		else:
			val, __ = self.evaluate(S + e)
		return val - current_val, None

	def add_one_element(self, e, S, current_val = None, auxiliary = None):
		if e in S:
			return 0, None
		if isinstance(S, int):
			S = [S]
		if isinstance(e, list):
			return self.evaluate(S + e)
		else:
			return self.evaluate(S + [e])


def distance_cost(dataset, S):
	if isinstance(S, list) or isinstance(S, set):
		val = 0
		for e in set(S):
			val += dataset['dist_cost'][e]
		return val * dataset['reg_coef']
	else:
		return dataset['dist_cost'][S] * dataset['reg_coef']

def twitter_cost(dataset, S):
	if isinstance(S, list) or isinstance(S, set):
		val = 0
		for e in set(S):
			val += dataset['costs'][e]
		return val * dataset['reg_coef']
	else:
		return dataset['costs'][S] * dataset['reg_coef']


class TweetDiversity():
	def __init__(self, dataset, power=0.5):
		self.dataset = dataset
		self.tweets = dataset['tweets']
		self.power = power
		self.oracle_calls = 0

	def evaluate(self, S):
		self.oracle_calls += 1
		if isinstance(S, int):
			S = [S]

		if isinstance(S, set):
			S = list(S)

		if isinstance(S, list):
			S = list(set(S))

		if len(S) == 0:
			return 0, None

		counts = dict()
		for e in S:
			tweet = self.tweets[e]
			user = tweet[0]
			words = tweet[1]
			retweets = tweet[2]
			for word in words:
				if word not in counts:
					counts[word] = 0
				counts[word] += retweets
		score = 0.0 
		for word in counts:
			if counts[word] > 0:
				points = counts[word] ** (self.power)
			else:
				points = 0
			score += points
		return score, None

	def marginal_gain(self, e, S, current_val = None, auxiliary = None):
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if e in S:
			return 0.0, None
		if current_val == None:
			current_val, __ = self.evaluate(S)

		if isinstance(e, int):
			val, __ = self.evaluate(S + [e])
		else:
			val, __ = self.evaluate(S + e)
		return val - current_val, None

	def add_one_element(self, e, S, current_val = None, auxiliary = None):
		if e in S:
			return 0, None
		if isinstance(S, int):
			S = [S]
		if isinstance(e, list):
			return self.evaluate(S + e)
		else:
			return self.evaluate(S + [e])



def loadTwitterData(filename, nb_tweets=1000,startDate='2000-01-01',endDate='2020-01-01',reg_coef=1.0):
	count = 0
	tweets = [] #List of tweets.
	M = {} #List of handles (i.e. unique streams we scraped from).
	with open(filename,'r') as f:
		for line in f:
			arr = line.split(',,') #Value are split up by double commas ',,'.
			temp = []
			
			handle = arr[0]
			temp.append(handle) #handle.
			
			cleanText = arr[3]
			cleanText = cleanText.split(' ')
			temp.append(cleanText) #list of (cleaned) words in this tweet.
			
			averageRetweets = int(arr[5])/float(len(cleanText))
			temp.append(averageRetweets) #Number of retweets divided by number of words
			
			timestamp = arr[7].strip()
			temp.append(timestamp) #Date/time of tweet.
			temp.append(arr[2]) #Raw text of tweet.
			
			if timestamp[:10] >= startDate and timestamp[:10] <= endDate and averageRetweets > 0:
				if handle not in M:
					M[handle] = 0
				M[handle] += 1
				count += 1
				tweets.append(temp)	

	stream = []
	for tweet in tweets:
		timestamp = tweet[3]
		stream.append([tweet,timestamp]) 
		
	cleanStream = []
	for tweet in stream:
		cleanStream.append(tweet[0])

	cleanStream = cleanStream[:nb_tweets]
	elements = [i for i in range(nb_tweets)]

	tweets_len = dict()

	for e in elements:
		tweets_len[e] = len(cleanStream[e][1])

	base=datetime.date(2019,1,1)
	tweets_time = dict()
	for e in elements:
		date_time_str=cleanStream[e][3]
		date=datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S').date()
		delta = date - base
		tweets_time[e] = abs(delta.days/30)

	dataset = {'tweets':cleanStream, 'handles':M, 'nb_elements':nb_tweets,
	'elements':elements, 'tweets_len':tweets_len, 'reg_coef':reg_coef, 'tweets_time':tweets_time}

	return dataset


def log_submodular_cost(dataset,S):
	if isinstance(S, int):
		S = [S]
	linear_part = 0
	for e in S:
		linear_part += dataset['costs'][e]
	return linear_part

class RootLogDet():
	def __init__(self, dataset, k = 40, gamma = 1):
		self.dataset = dataset
		self.A = dataset['A']
		self.costs = dataset['costs']
		self.gamma = gamma
		self.k = k
		self.oracle_calls = 0
		self.n = len(dataset['elements'])

	def evaluate_rho(self, S):
		if isinstance(S, int):
				S = [S]
		if isinstance(S, set):
			S = list(S)
		if isinstance(S, list):
			S = list(set(S))

		#if len(S) > self.k:
			#return 0, None

		n = len(S) 
		if n == 0:
			return 0, None
		M = self.A[np.ix_(S,S)]
		return math.log(math.sqrt(np.linalg.det(M))), None


	def evaluate(self, S):
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if isinstance(S, list):
			S = list(set(S))

		if len(S) > self.k:
			return 0, None

		n = len(S) 
		if n == 0:
			return 0, None
		M = self.A[np.ix_(S,S)]
		self.oracle_calls += 1
		linear_part = 0
		for e in S:
			linear_part += self.costs[e]
		return math.log(math.sqrt(np.linalg.det(M))) + linear_part, None

	def marginal_gain(self, e, S, current_val = None, auxiliary = None):
		if e in S:
			return 0, None
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if e in S:
			return 0.0, None
		if current_val == None:
			current_val, __ = self.evaluate(S)

		if isinstance(e, int):
			val, __ = self.evaluate(S + [e])
		else:
			val, __ = self.evaluate(S + e)
		return val - current_val, None


	def add_one_element(self, e, S, current_val = None, auxiliary = None):
		if e in S:
			return 0, None
		if isinstance(S, int):
			S = [S]
		if isinstance(e, list):
			return self.evaluate(S + e)
		else:
			return self.evaluate(S + [e])

def get_video_frames_data( data_file, Ent, reg_coefa, norm_k=2):  ## data_file has the features for all images, Nxd
    n= data_file.shape[0]
    D= np.zeros((n,n))   ## distance matrix
    sim= np.zeros((n,n)) ## similarity matrix
    z= np.zeros((n,n))   ## selection matrix
    
    #### forming the similarity and distance matrices
    for i in range(n):
        for j in range(n):
            cur_diff= data_file[i,:]-data_file[j,:]
            cur_dist= LA.norm(cur_diff, norm_k)
            D[i,j]= cur_dist
            sim[i,j]= np.exp(-1*cur_dist)

    #elements= np.arange(n)
    #elements = list(elements)
    elements = [i for i in range(n)]
    #random.shuffle(elements)
    
    #### defining the cost function
    costs = dict()
    for e in elements:
        costs[e] = 1
        
    entropy= {}
    for e in elements:
        entropy[e]= Ent[e][0]
        #entropy[e]= 1


    data_dict = {'similarity':sim, 'sim_mat':sim, 'entropy': entropy,'distance':D, 
    'reg_coef':reg_coefa, 'elements': elements, 'nb_elements':len(elements), 
    'costs': entropy, 'orig_cost': entropy}
    return data_dict


    
def entropy_cost(dataset, S):
    if isinstance(S, list) or isinstance(S, set):
        val = 0
        for e in set(S):
            val += dataset['reg_coef']*dataset['entropy'][e]
        return val
    else:
        return dataset['reg_coef']*dataset['entropy'][S]