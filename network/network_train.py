import pickle
import pandas as pd
import numpy as np

train = pickle.load(open('../data/train_centrality.p', 'rb'))

degree_mean = train['degree'].mean()
degree_sd = train['degree'].std()

btw_mean = train['betweenness'].mean()
btw_sd = train['betweenness'].std()
