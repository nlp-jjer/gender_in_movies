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
from sklearn import preprocessing, metrics
from sklearn.grid_search import ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from scipy import sparse

pd.options.mode.chained_assignment = None



class TextClassifier():
    '''
    To store the sklearn model object and the trained vocabulary for later use
    '''
    def __init__(self, clf_fitted, count_vect, tfidf_transformer):
        self.clf_fitted = clf_fitted
        self.count_vect = count_vect
        self.tfidf_transformer = tfidf_transformer
        

def prepare_data(filename, feature_cols, genre = 'all', speaker_pairs = False, random_state = 42):
    '''
    Prepare data for modeling. Get the same train/test split unless random_state is specified.
    
    filename: path to pickled dataframe
    feature_cols: list of columns in df to use as features
    ''' 
    # load pre-processed df
    df = pickle.load(open(filename, 'rb'))
    
    # add columns to pre-processed df
    df = add_columns(df)
    
    # specify genre
    if genre != 'all':
        df = df[df.genre == genre]
    
    # split into train and validation sets
    if speaker_pairs: # use gender pairing as outcome classes
        X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df.gender_pair.astype('int'), test_size=0.33, random_state=random_state)
    else: # use gender_from (M/F) as outcome classes
        X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df.gender_from.astype('int'), test_size=0.33, random_state=random_state)

    # transform train and test sets
    X_train_tfidf_combined, count_vect, tfidf_transformer = transform_train(X_train)
    X_test_tfidf_combined = transform_test(X_test, count_vect, tfidf_transformer)

    return X_train_tfidf_combined, X_test_tfidf_combined, y_train, y_test, count_vect, tfidf_transformer


def add_columns(df):    
    '''
    Add columns to input dataframe (which has been preprocessed) for non-text features,
    i.e. power/agency analysis and topic modeling.
    
    Also convert output columns into usable format for machine learning.
    '''
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

    # load df with power/agency verb columns and join with main
    df_verbs = pickle.load(open("../data/movies_verbs.p", 'rb'))
    df_verbs.fillna(0, inplace=True)
    df = pd.merge(df, 
                  df_verbs[['line_id', 'agency_pos_prop','power_pos_prop', 'agency_neg_prop','power_neg_prop']], 
                  how='left', 
                  on='line_id')
    
    return df


def transform_train(X_train):
    '''
    Process text features of training dataframe
    '''
    count_vect = CountVectorizer() # using bag of words
    tfidf_transformer = TfidfTransformer()
    
    # vectorize the text column
    X_train_counts = count_vect.fit_transform(X_train.words)    
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    # combine sparse matrix of text features with other features
    X_train_others = sparse.csr_matrix(X_train.loc[:, X_train.columns != 'words'])
    X_train_tfidf_combined = sparse.hstack((X_train_tfidf, X_train_others)).tocsr()
    
    return X_train_tfidf_combined, count_vect, tfidf_transformer


def transform_test(X_test, count_vect, tfidf_transformer):
    '''
    Process test features of testing dataframe, using count_vect and 
    tfidf_transformer created on training data
    '''
    X_test_counts = count_vect.transform(X_test.words)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    
    # combine sparse matrix of text features with other features
    X_test_others = sparse.csr_matrix(X_test.loc[:, X_test.columns != 'words'])
    X_test_tfidf_combined = sparse.hstack((X_test_tfidf, X_test_others)).tocsr()
    
    return X_test_tfidf_combined


def fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS, GRID, count_vect, tfidf_transformer, multiclass = False):
    """
    Loop through a number of classification fit_models and parameters, saving results for each run
    Based off code from previous homework assignment: https://github.com/joan-wang/Machine-Learning/blob/master/hw3/model.py
    If speaker_pairs = True when prepare_data function was called, then multiclass must also be True.
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


def classify_unseen(test_df, clf_object, feature_cols):
    '''
    Function for scoring.py to classify movies from the holdout set using 
    a specific saved classifier object.
    Attach predictions and predicted probabilities to the last columns of test_df.
    
    test_df: df containing movie lines to be tested
    clf_object: object of class TextClassifier containing 
    feature_cols: list of columns from df that were used as features
    '''
    model =  clf_object.clf_fitted
    
    test_df = add_columns(test_df)
    test_df_tfidf = transform_test(test_df[feature_cols],
                                           clf_object.count_vect,
                                           clf_object.tfidf_transformer)
    predicted = model.predict(test_df_tfidf)
    pred_probs = model.predict_proba(test_df_tfidf)
    test_df['male_prob'] = pred_probs[:, 0]
    test_df['female_prob'] = pred_probs[:, 1]
    
    return predicted, pred_probs, test_df


def calculate_class_probs(df):
    '''
    Compare the probability male of male dialogue vs probability female of female dialogue. 
    This addresses the question: How well does male dialogue fit the male class and female dialogue 
    fit the female class?  

    A high score for for either class means the class closely fits the Hollywood standard for that gender    
    '''
    male_lines = df[df.gender_from == 0]
    male_lines_male_avg = np.mean(male_lines.male_prob)
    
    female_lines = df[df.gender_from == 1]
    female_lines_female_avg = np.mean(female_lines.female_prob)
    
    return male_lines_male_avg, female_lines_female_avg


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