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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer



CLASSIFIERS = {'RF': RandomForestClassifier(),
        'LR': LogisticRegression(),
        'GB': GradientBoostingClassifier(),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'MNB' : MultinomialNB()
            }

GRID = { 
'RF':{'n_estimators': [1, 10, 100, 1000], 'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20], 'max_features': ['sqrt', 'log2'],'min_samples_split': [2, 5, 10, 20]},
'LR': { 'penalty': ['l1', 'l2'], 'C': [.001, 0.01, .1, 1, 10]},
'GB': {'n_estimators': [1, 10, 100, 1000], 'learning_rate' : [0.001, 0.01, 0.1, 0.5],'subsample' : [0.1, 0.5, 1], 'max_depth': [1, 5, 10, 20]},
'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10, 20]},
'KNN' :{'n_neighbors': [5, 10, 50, 100], 'weights': ['uniform', 'distance'],'algorithm': ['auto']},
'MNB' :{'alpha': [1.0, 0.5, 1.5]}
       }


def prepare_data(filename):
    '''
    Prepare data for modeling
    
    filename: path to pickled dataframe
    '''
    # load pre-processed df
    df = pickle.load(open(filename, 'rb'))
    
    # remove unknown gender. 0 is male and 1 is female.
    df = df[df.gender_from != '?']
    df['gender_from'] = np.where(df.gender_from == 'f', 1, 0)
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(df.words, df.gender_from.astype('int'), test_size=0.33, random_state=42)

    # transform training data
    count_vect = CountVectorizer() # using bag of words
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    # transform testing data
    X_test_counts = count_vect.transform(X_test) # JW TO DO: how does this treat unseen ngrams?
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return X_train_tfidf, X_test_tfidf, y_train, y_test


def fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS, GRID):
    """
    Loop through a number of classification fit_models and parameters, saving results for each run
    Based off code from previous homework assignment: https://github.com/joan-wang/Machine-Learning/blob/master/hw3/model.py
    """
    #X_train, X_test = align_cols(X_train, X_test)
    
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
                
                end_time = time.time()
                tot_time = end_time - start_time
                print(p)
                
                predicted = clf_fitted.predict(X_test)
                y_pred_probs = clf_fitted.predict_proba(X_test)[:,1]
                
                auc_roc = metrics.roc_auc_score(y_test, y_pred_probs)
                accuracy = metrics.accuracy_score(y_test, predicted)
                precision = metrics.precision_score(y_test, predicted)
                recall = metrics.recall_score(y_test, predicted)
                conf = metrics.confusion_matrix(y_test, predicted)
                results.loc[len(results)] = [key, clf, p,
											auc_roc,
											accuracy, precision, recall,									
											tot_time, conf, y_pred_probs]
                plot_precision_recall(y_test,y_pred_probs,clf)
            except IndexError:
                print('Error')
                continue
        return results


def align_cols(X_train, X_test):
    """
    Fix column mismatch that may occur between train and test sets due to rare catergories that are turned into dummy columns
    """  
    train_cols = set(X_train)
    test_cols = set(X_test)

    missing_from_test = train_cols - test_cols
    missing_from_train = test_cols - train_cols

    for col in missing_from_test:
        X_test[col] = 0
    
    for col in missing_from_train:
        X_train[col] = 0
    
    X_test = X_test[X_train.columns]
    
    return X_train, X_test



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