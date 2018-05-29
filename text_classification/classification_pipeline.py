# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:40:44 2018

@author: JoanWang
"""
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, metrics
from sklearn.grid_search import ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer



class TextClassifier():
    '''
    To store the sklearn model object and the trained vocabulary for later use
    '''
    def __init__(self, clf_fitted, count_vect, tfidf_transformer):
        self.clf_fitted = clf_fitted
        self.count_vect = count_vect
        self.tfidf_transformer = tfidf_transformer
        

def prepare_data(filename, genre = 'all', speaker_pairs = False, random_state = 42):
    '''
    Prepare data for modeling. Get the same train/test split unless random_state is specified.
    
    filename: path to pickled dataframe
    '''
    # load pre-processed df
    df = pickle.load(open(filename, 'rb'))
   
    # specify genre
    if genre != 'all':
        df = df[df.genre == genre]
    
    # split into train and test sets
    if speaker_pairs: # use gender pairing as outcome classes
        X_train, X_test, y_train, y_test = train_test_split(df.words, df.gender_pair.astype('int'), test_size=0.33, random_state=random_state)
    else: # use gender_from (M/F) as outcome classes
        X_train, X_test, y_train, y_test = train_test_split(df.words, df.gender_from.astype('int'), test_size=0.33, random_state=random_state)

    # transform train and test sets
    X_train_tfidf, count_vect, tfidf_transformer = transform_train(X_train)
    X_test_tfidf = transform_test(X_test, count_vect, tfidf_transformer)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, count_vect, tfidf_transformer


def transform_train(X_train):
    count_vect = CountVectorizer() # using bag of words
    tfidf_transformer = TfidfTransformer()

    X_train_counts = count_vect.fit_transform(X_train)    
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    return X_train_tfidf, count_vect, tfidf_transformer


def transform_test(X_test, count_vect, tfidf_transformer):
    X_test_counts = count_vect.transform(X_test) # JW TO DO: how does this treat unseen ngrams?
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    
    return X_test_tfidf


def fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS, GRID, count_vect, tfidf_transformer, multiclass = False):
    """
    Loop through a number of classification fit_models and parameters, saving results for each run
    Based off code from previous homework assignment: https://github.com/joan-wang/Machine-Learning/blob/master/hw3/model.py
    """
    classifier_objects = []
    results =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc', 
                                     'accuracy', 'precision', 'recall',
                                     'runtime', 'confusion_matrix', 'y_pred_probs'))
    for key, clf in CLASSIFIERS.items():

        print(key)

        for p in ParameterGrid(GRID[key]):
            try:
                start_time = time.time()
                clf.set_params(**p)
                clf_fitted = clf.fit(X_train, y_train)
                clf_object = TextClassifier(clf_fitted, count_vect, tfidf_transformer)
                
                end_time = time.time()
                tot_time = end_time - start_time
                print(p)
                
                predicted = clf_fitted.predict(X_test)
                
                if multiclass:
                    y_pred_probs = clf_fitted.predict_proba(X_test)
                    auc_roc = None
                    precision = metrics.precision_score(y_test, predicted, average = None)
                    recall = metrics.recall_score(y_test, predicted, average = None)
                else:
                    y_pred_probs = clf_fitted.predict_proba(X_test)[:,1]
                    auc_roc = metrics.roc_auc_score(y_test, y_pred_probs)
                    precision = metrics.precision_score(y_test, predicted, average = 'binary')
                    recall = metrics.recall_score(y_test, predicted, average = 'binary')
                
                accuracy = metrics.accuracy_score(y_test, predicted)
                
                conf = metrics.confusion_matrix(y_test, predicted)
                
                results.loc[len(results)] = [key, clf, p,
											auc_roc,
											accuracy, precision, recall,									
											tot_time, conf, y_pred_probs]
                classifier_objects.append(clf_object)
                # plot_precision_recall(y_test,y_pred_probs,clf)
            except IndexError:
                print('Error')
                continue
    return results, classifier_objects




def plot_precision_recall(y_true, y_score, model_name):
    """
    Plot precision and recall curves for varying percents of the population 
    """
    precision_curve, recall_curve, pr_thresholds = metrics.precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    plt.show()