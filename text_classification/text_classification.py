# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:12:50 2018

@author: JoanWang
"""
import pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier



CLASSIFIERS = {'RF': RandomForestClassifier(),
        'LR': LogisticRegression(),
        #'GB': GradientBoostingClassifier(),
        'DT': DecisionTreeClassifier(),
        'MNB' : MultinomialNB()
            }

GRID = { 
'RF':{'n_estimators': [1, 10, 100, 1000], 'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20], 'max_features': ['sqrt', 'log2'],'min_samples_split': [2, 5, 10, 20]},
'LR': { 'penalty': ['l1', 'l2'], 'C': [.001, 0.01, .1, 1, 10]},
'GB': {'n_estimators': [10, 100], 'learning_rate' : [ 0.01, 0.1],'subsample' : [0.5, 1], 'max_depth': [5, 10]},
'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10, 20]},
'MNB' :{'alpha': [1.0, 0.5, 1.5]}
       }

CLASSIFIERS_BEST = {'MNB': MultinomialNB(), 'LR': LogisticRegression()}

GRID_BEST = {'MNB': {'alpha':[0.5]}, 'LR': {'penalty': ['l1', 'l2'], 'C': [1]}}

###################################

# Ran the code below multiple times for different classifiers groups
# Logistic regression and multinomial NB do the best at 71% accuracy
# But recall is really bad in all these models! Not capturing most 
X_train, X_test, y_train, y_test = pipeline.prepare_data("../movies.p")
results = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS, GRID) 
results.to_csv('results_mnb.csv')

###################################

# Create separate models by genre
# Romance gets accuracy of 80%
X_train, X_test, y_train, y_test = pipeline.prepare_data("../movies.p", genre = 'romance')
results = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST)
results.to_csv('results_romance.csv')

# Action gets accuracy of 70%
X_train, X_test, y_train, y_test = pipeline.prepare_data("../movies.p", genre = 'action')
results = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST)
results.to_csv('results_action.csv')

# Sci-fi gets accuracy of 76%
X_train, X_test, y_train, y_test = pipeline.prepare_data("../movies.p", genre = 'sci-fi')
results = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST)
results.to_csv('results_scifi.csv')

###################################
# Clasisfy by speaker pair
# Accuracy is 47%. 
X_train, X_test, y_train, y_test = pipeline.prepare_data("../movies.p", speaker_pairs = True)
results = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, multiclass = True)
results.to_csv('results_speaker_pairs.csv')