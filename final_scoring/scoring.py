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

#import our modules containing functions to be called from within the class
sys.path.insert(0, '../text_classification')
import classification_pipeline as pipeline
from classification_main import FEATURE_COLS, clf_ratios_mean, clf_ratios_sd

sys.path.insert(0, '../network')
from network_train import degree_mean, degree_sd, btw_mean, btw_sd
import network_functions as nt

##########################################################

class Movie():
    def __init__(self, lines_df, movie_id):
        """
        lines_df: dataframe containing lines of movie(s) to be tested
        """
        self.lines = lines_df
        self.id = movie_id
        self.counts = nt.create_movie_df(self.lines, self.id)
        
        # for female prop
        self.female_prop = None
        
        # for text classification ratio
        self.X_test = None
        self.preds = None
        self.pred_probs = None
        self.clf_object = None
        self.ratio_of_probs = None

        # for cosine similarity
        self.cosine_sim = None

        # from network connectedness
        self.network_degree = None
        self.network_btw = None


    def get_female_prop(self):
        print("\n########################################\nCalculating female proportion...")
        tot_lines = self.lines.shape[0]
        female_lines = len(self.lines[self.lines.gender_from == 'f'])
        self.female_prop = female_lines/tot_lines

        print('Total lines: ', tot_lines)
        print('Female lines: ', female_lines)
        print('Female proportion: ', round(self.female_prop, 2))
        return self.female_prop


    def get_cosine_sim(self):
        print("\n########################################\nCalculating cosine similarity...")

        self.cosine_sim = 0
        ### CODE TO CALCULATE COSINE SIMILARITY ###

        print("\nCosine similarity: ", round(self.cosine_sim,2))

        return self.cosine_sim


    def get_class_ratio(self, clf_object):
        print("\n########################################\nCalculating classification ratio...")
    
        # use the saved classifer to classify lines of this movie
        self.clf_object = clf_object # contains model, count_vect, and tfidf_transformer  
        self.preds, self.pred_probs, self.X_test = pipeline.classify_unseen(self.lines, clf_object, FEATURE_COLS)
        '''
        # Option 1:
        # The ratio of the avg probability female vs probability male for the entire script. 
        # This addresses the question: Does the script have a gender leaning, and to what extent?
        avg_male_prob = np.mean(self.X_test.male_prob)
        avg_female_prob = np.mean(self.X_test.female_prob)
        
        self.ratio_of_probs = avg_female_prob / avg_male_prob
        print("Ratio of avg female/male prob: ", round(self.ratio_of_probs, 5))
        
        return self.ratio_of_probs
        
        
        # Option 2:
        # The ratio of the probability male of male dialogue vs probability female of female dialogue. 
        # This addresses the question: How well does male dialogue fit the male class and female dialogue 
        # fit the female class?        
        male_lines = self.X_test[self.X_test.gender_from == 0]
        male_lines_male_prob = np.mean(male_lines.male_prob)
        
        female_lines = self.X_test[self.X_test.gender_from == 1]
        female_lines_female_prob = np.mean(female_lines.female_prob)
        
        self.ratio_of_probs = female_lines_female_prob / male_lines_male_prob
        print("Ratio of avg female/male prob: ", round(self.ratio_of_probs, 5))
        '''
        
        # NEED TO CHOOSE AN OPTION. Also, should it be like networks, the diff btw m and f?
        # Option 1: 
        ratio_of_probs = pipeline.calculate_ratio1(self.X_test)
        
        # Option 2:
        ratio_of_probs = pipeline.calculate_ratio2(self.X_test)
        
        # Normalize
        
        self.ratio_of_probs = (ratio_of_probs - clf_ratios_mean) / clf_ratios_sd
        print("Ratio of avg female/male prob: ", round(self.ratio_of_probs, 5))

        return self.ratio_of_probs
        


    def get_network_degree(self):
        print("\n########################################\nCalculating network degree...")

        f_degree = nt.get_centrality(self.counts, 'degree', 'f')
        m_degree = nt.get_centrality(self.counts, 'degree', 'm')

        #normalize
        f_degree_norm = (f_degree - degree_mean) / degree_sd
        m_degree_norm = (m_degree - degree_mean) / degree_sd

        self.network_degree = m_degree_norm - f_degree_norm

        print("\nNetwork degree: ", round(self.network_degree, 2))
        return self.network_degree


    def get_network_betweenness(self):
        print("\n########################################\nCalculating network betweenness...")

        f_btw = nt.get_centrality(self.counts, 'betweenness', 'f')
        m_btw = nt.get_centrality(self.counts, 'betweenness', 'm')

        #normalize
        f_btw_norm = (f_btw - btw_mean) / btw_sd
        m_btw_norm = (m_btw - btw_mean) / btw_sd

        self.network_btw = m_btw_norm - f_btw_norm

        print("\nNetwork betweenness: ", round(self.network_btw, 2))
        return self.network_btw


    def score_movie(self, classifier):
        """
        Inputs
            classifer: pickled classification model object
        """
        female_prop = self.get_female_prop()
        cosine_sim = self.get_cosine_sim()
        class_ratio = self.get_class_ratio(classifier)
        network_degree = self.get_network_degree()
        network_betweenness = self.get_network_betweenness()

        # need to weight these based on observed distributions
        final_score = np.mean([female_prop, cosine_sim, class_ratio,
                               network_degree, network_betweenness])

        print("\n########################################\nFinal score: ",
              round(final_score, 2))




if __name__ == "__main__":
    movies = pickle.load(open("../data/movies_lines_holdout.p", 'rb'))
    classifier_object = pickle.load(open("../text_classification/mnb_final.p", 'rb'))

    # test code on one movie
    m0_df = movies[movies.movie_id == 'm1']
    m0_movie = Movie(m0_df, 'm1')
    m0_score = m0_movie.score_movie(classifier_object)
