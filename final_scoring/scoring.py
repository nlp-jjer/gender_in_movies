# -*- coding: utf-8 -*-
"""
Created on Sat May 26 19:12:59 2018

@author: JoanWang
"""
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#import our modules containing functions to be called from within the class
sys.path.insert(0, '../preprocessing')
from speaker_proportions import prop_diff_mean, prop_diff_sd

sys.path.insert(0, '../text_classification')
import classification_pipeline as pipeline
import classification_main as clf_main

sys.path.insert(0, '../network')
from network_train import degree_mean, degree_sd, btw_mean, btw_sd
import network_functions as nt

sys.path.insert(0, '../word_embeddings')
from cosim_score import movie_cosine

##########################################################

class Movie():
    def __init__(self, lines_df, movie_id):
        """
        lines_df: dataframe containing lines of movie(s) to be tested
        """
        self.lines = lines_df
        self.id = movie_id
        self.counts = nt.create_movie_df(self.lines, self.id)
        
        # for line proportions
        self.female_prop = None
        self.male_prop = None
        self.prop_diff = None
        
        # for text classification probs
        self.X_test = None
        self.preds = None
        self.pred_probs = None
        self.clf_object = None
        self.male_class_avg = None
        self.female_class_avg = None

        # for cosine similarity
        self.raw_cosine = None
        self.cosine_sim = None

        # from network connectedness
        self.network_degree = None
        self.m_degree_norm = None
        self.f_degree_norm = None
        
        self.network_btw = None
        self.m_btw_norm = None
        self.f_btw_norm = None
        
        # for final score        
        self.final_score = None


    def get_proportions(self):
        print("\n########################################\nCalculating line proportions...")
        tot_lines = self.lines.shape[0]
        female_lines = len(self.lines[self.lines.gender_from == 'f'])
        
        female_prop = female_lines/tot_lines
        male_prop = 1-female_prop
        prop_diff = abs(male_prop - female_prop)
        
        print('Female proportion: ', round(female_prop, 2))
        print('Male proportion: ', round(male_prop, 2))
        
        # Normalize
        prop_diff = (prop_diff - prop_diff_mean) / prop_diff_sd        
        print('Diff in proportions (normed): ', round(prop_diff, 2))
        
        return female_prop, male_prop, prop_diff


    def get_cosine_sim(self):
        print("\n########################################\nCalculating cosine similarity...")

        # Calculate & save similarity score before normalization 
        self.raw_cosine = movie_cosine.similarity(self.lines)
        cosine_sim = movie_cosine.norm_similarity(self.lines)

        print("Cosine similarity (normed): ", round(cosine_sim,2))

        return cosine_sim


    def get_class_probs(self, clf_object):
        print("\n########################################\nCalculating classification probabilities...")
    
        # use the saved classifer to classify lines of this movie
        self.clf_object = clf_object # contains model, count_vect, and tfidf_transformer  
        self.preds, self.pred_probs, self.X_test = pipeline.classify_unseen(self.lines, clf_object, clf_main.FEATURE_COLS)

        male, female = pipeline.calculate_class_probs(self.X_test)
        
        # Normalize
        male_class_avg = (male - clf_main.male_class_mean) / clf_main.male_class_sd
        print("Male prob of male lines (normed): ", round(male_class_avg, 2))
        
        female_class_avg = (female - clf_main.female_class_mean) / clf_main.female_class_sd
        print("Female prob of female lines (normed): ", round(female_class_avg, 2))

        return male_class_avg, female_class_avg
        

    def get_network_degree(self):
        print("\n########################################\nCalculating network degree...")

        f_degree = nt.get_centrality(self.counts, 'degree', 'f')
        m_degree = nt.get_centrality(self.counts, 'degree', 'm')
        
        print("Female degree: ", round(f_degree,2))
        print("Male degree: ", round(m_degree,2))

        #normalize
        f_degree_norm = (f_degree - degree_mean) / degree_sd
        m_degree_norm = (m_degree - degree_mean) / degree_sd

        network_degree = abs(m_degree_norm - f_degree_norm)

        print("Network degree diff (normed): ", round(network_degree, 2))
        return network_degree, m_degree_norm, f_degree_norm


    def get_network_betweenness(self):
        print("\n########################################\nCalculating network betweenness...")

        f_btw = nt.get_centrality(self.counts, 'betweenness', 'f')
        m_btw = nt.get_centrality(self.counts, 'betweenness', 'm')
        
        print("Female betweenness: ", round(f_btw,2))
        print("Male betweenness: ", round(m_btw,2))

        #normalize
        f_btw_norm = (f_btw - btw_mean) / btw_sd
        m_btw_norm = (m_btw - btw_mean) / btw_sd

        network_btw = abs(m_btw_norm - f_btw_norm)

        print("Network betweenness diff (normed): ", round(network_btw, 2))
        return network_btw, m_btw_norm, f_btw_norm


    def score_movie(self, classifier):
        """
        Inputs
            classifer: pickled classification model object
        """
        self.female_prop, self.male_prop, self.prop_diff = self.get_proportions()
        self.cosine_sim = self.get_cosine_sim()
        self.male_class_avg, self.female_class_avg = self.get_class_probs(classifier)
        self.network_degree, self.m_degree_norm, self.f_degree_norm = self.get_network_degree()
        self.network_btw, self.m_btw_norm, self.f_btw_norm = self.get_network_betweenness()

        # need to weight these based on observed distributions
        self.final_score = np.mean([self.prop_diff, self.cosine_sim,
                               self.male_class_avg, self.female_class_avg, 
                               self.network_degree, self.network_btw])
    
        print("\n########################################\nFinal score: ",
              round(self.final_score, 2))




if __name__ == "__main__":
    movies = pickle.load(open("../data/movies_lines_holdout.p", 'rb'))
    classifier_object = pickle.load(open("../text_classification/mnb_final.p", 'rb'))

    # test code on one movie
    m0_df = movies[movies.movie_id == 'm0']
    m0_movie = Movie(m0_df, 'm0')
    m0_score = m0_movie.score_movie(classifier_object)
