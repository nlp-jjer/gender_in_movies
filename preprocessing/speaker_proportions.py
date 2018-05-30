# -*- coding: utf-8 -*-
"""
Created on Wed May 30 00:23:36 2018

@author: JoanWang
"""
import pickle
import numpy as np

def get_normalization_prop(movies_lines_filename):
    '''
    Get average line proportions per gender for each movie in training set 
    (movies used to test and validate the classifer) to use as normalization factor for final scoring.
    '''
    movies_lines_train = pickle.load(open(movies_lines_filename, 'rb'))
    movies_lines_train['gender_from'] = np.where(movies_lines_train.gender_from == 'f', 1, 0)
    
    groupby_movie = movies_lines_train[['movie_id', 'gender_from']].groupby(by = 'movie_id')
    female_props = groupby_movie.sum()/groupby_movie.count()
    male_props = 1 - female_props
    
    prop_diff = male_props - female_props

    return prop_diff, male_props, female_props

# compute normalization metrics for line proportions using training data    
prop_diff, male_props, female_props = get_normalization_prop("../data/movies_lines_train.p")
prop_diff_mean = np.nanmean(prop_diff)
prop_diff_sd = np.nanstd(prop_diff)