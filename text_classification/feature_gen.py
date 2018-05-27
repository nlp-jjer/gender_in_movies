# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:22:25 2018

Series of processing steps, after initial preprocessing, to create features 
and prepare df for machine learning pipeline.

@author: JoanWang
"""
import pickle
import pandas as pd
import numpy as np


# load pre-processed df
df = pickle.load(open("../movies.p", 'rb'))

# Treat gender_from and gender_to columns: remove unknown gender, 0 is male and 1 is female
df = df[df.gender_from != '?']
df['gender_from'] = np.where(df.gender_from == 'f', 1, 0)

# Treat gender_to column: remove unknown gender, 0 is male and 1 is female
df = df[df.gender_to != '?']
df['gender_to'] = np.where(df.gender_to == 'f', 1, 0)
    
# Create gender pair column
# MM = 0, MF = 1, FM = 2, FF = 3
conditions = [(df.gender_from == 0) & (df.gender_to == 0),
     (df.gender_from == 0) & (df.gender_to == 1),
     (df.gender_from == 1) & (df.gender_to == 0),
     (df.gender_from == 1) & (df.gender_to == 1)]
choices = [0,1,2,3]
df['gender_pair'] = np.select(conditions, choices)

print(df.shape)

# load df with power/agency verb columns and join with main
df_verbs = pickle.load(open("../data/movies_verbs.p", 'rb'))
df = pd.merge(df, 
              df_verbs[['line_id', 'agency_pos_prop','power_pos_prop', 'agency_neg_prop','power_neg_prop']], 
              how='left', 
              on='line_id')

print(df.shape)

# save to new pickle
pickle.dump(df, open('../data/movies_features.p', 'wb'))