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
from classification_main import FEATURE_COLS

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

        # for text classification ratio
        self.X_test = None
        self.clf_object = None
        self.model = None
        self.class_ratio = None

        # for cosine similarity
        self.cosine_sim = None

        # from network connectedness
        self.network_degree = None
        self.network_betweenness = None


    def get_female_prop(self):
        print("\n########################################\nCalculating female proportion...")
        tot_lines = self.lines.shape[0]
        female_lines = len(self.lines[self.lines.gender_from == 'f'])
        female_prop = female_lines/tot_lines

        print('Total lines: ', tot_lines)
        print('Female lines: ', female_lines)
        print('Female proportion: ', round(female_prop, 2))
        return female_prop


    def get_cosine_sim(self):
        print("\n########################################\nCalculating cosine similarity...")

        cosine_sim = 0
        ### CODE TO CALCULATE COSINE SIMILARITY ###

        print("\nCosine similarity: ", round(cosine_sim,2))

        return cosine_sim


    def get_class_ratio(self, clf_object):
        print("\n########################################\nCalculating classification ratio...")

        self.clf_object = clf_object
        self.model = clf_object.clf_fitted

        self.X_test = pipeline.add_columns(self.lines)
        X_test_tfidf = pipeline.transform_test(self.X_test[FEATURE_COLS],
                                               self.clf_object.count_vect,
                                               self.clf_object.tfidf_transformer)

        predicted = self.model.predict(X_test_tfidf)

        # using this as a dummy class ratio for now
        pct_female = np.mean(predicted)
        print("% of lines predicted female: ", round(pct_female, 2))

        class_ratio = pct_female
        ### CODE TO CALCULATE CLASSIFICATION RATIO ###

        print("Class ratio: ", round(class_ratio, 2))

        return class_ratio


    def get_network_degree(self):
        print("\n########################################\nCalculating network degree...")

        f_degree = nt.get_centrality(self.counts, 'degree', 'f')
        m_degree = nt.get_centrality(self.counts, 'degree', 'm')

        #normalize
        f_degree_norm = (f_degree - degree_mean) / degree_sd
        m_degree_norm = (m_degree - degree_mean) / degree_sd

        network_degree = m_degree_norm - f_degree_norm

        print("\nNetwork degree: ", round(network_degree, 2))
        return network_degree


    def get_network_betweenness(self):
        print("\n########################################\nCalculating network betweenness...")

        f_btw = nt.get_centrality(self.counts, 'betweenness', 'f')
        m_btw = nt.get_centrality(self.counts, 'betweenness', 'm')

        #normalize
        f_btw_norm = (f_btw - btw_mean) / btw_sd
        m_btw_norm = (m_btw - btw_mean) / btw_sd

        network_btw = m_btw_norm - f_btw_norm

        print("\nNetwork betweenness: ", round(network_btw, 2))
        return network_btw


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
    m0_df = movies[movies.movie_id == 'm0']
    m0_movie = Movie(m0_df, 'm0')
    m0_score = m0_movie.score_movie(classifier_object)
