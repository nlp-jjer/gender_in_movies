# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:12:50 2018

@author: JoanWang
"""
import pickle
import classification_pipeline as pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# define columns to use as features
FEATURE_COLS = ['words']#, 'agency_pos_prop','power_pos_prop', 'agency_neg_prop','power_neg_prop']

# big loop
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

# narrowed down to the best performing 3
CLASSIFIERS_BEST = {'MNB': MultinomialNB(), 'LR': LogisticRegression()}
GRID_BEST = {'MNB': {'alpha':[0.5]}, 'LR': {'penalty': ['l1', 'l2'], 'C': [1]}}

# final classifer being used
CLASSIFIER_FINAL = {'MNB': MultinomialNB()}
GRID_FINAL = {'MNB': {'alpha':[0.5]}}

def train_model(input_name, output_name, feature_cols, classifier_dict, grid_dict, genre = 'all', speaker_pairs = False):
    """
    input_name: pickle file containing preprocessed dataframe, before feature generation
    output_name: csv file to write results
    """
    X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data(input_name, feature_cols, genre=genre, speaker_pairs = speaker_pairs)
    results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, count_vect, tfidf_transformer) 
    results.to_csv(output_name)
    
    return results, classifier_objects


results, classifier_objects = train_model("../data/movies_lines_train.p", "results_lr_mnb.csv", FEATURE_COLS,CLASSIFIERS_BEST, GRID_BEST)
pickle.dump(classifier_objects[0], open('mnb_final.p', 'wb'))

'''
OLD CODE THAT WAS PREVIOUSLY RUN
###################################

# Ran the code below multiple times for different classifiers groups
# Logistic regression and multinomial NB do the best at 71% accuracy
# But recall is really bad in all these models! Not capturing most 
X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data("../data/movies_lines_train_features.p", FEATURE_COLS)
results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, count_vect, tfidf_transformer) 
results.to_csv('results_lr_mnb.csv')

###################################
# Create separate models by genre
# Romance gets accuracy of 80%
X_train, X_test, y_train, y_test,count_vect, tfidf_transformer = pipeline.prepare_data("../data/movies_features.p", FEATURE_COLS, genre = 'romance')
results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, count_vect, tfidf_transformer)
results.to_csv('results_romance.csv')

# Action gets accuracy of 70%
X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data("../data/movies_features.p", FEATURE_COLS, genre = 'action')
results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, count_vect, tfidf_transformer)
results.to_csv('results_action.csv')

# Drama gets accuracy of 69%
X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data("../data/movies_features.p", FEATURE_COLS, genre = 'drama')
results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, count_vect, tfidf_transformer)
results.to_csv('results_drama.csv')

# Comedy gets accuracy of 65%
X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data("../data/movies_features.p", FEATURE_COLS, genre = 'comedy')
results,classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, count_vect, tfidf_transformer, )
results.to_csv('results_comedy.csv')

# Crime gets accuracy of 69%
X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data("../data/movies_features.p", FEATURE_COLS, genre = 'crime')
results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, count_vect, tfidf_transformer, )
results.to_csv('results_crime.csv')


###################################
# Clasisfy by speaker pair - probably not useful for this project
# Accuracy is 47%. 
X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data("../data/movies_features.p", FEATURE_COLS, speaker_pairs = True)
results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIERS_BEST, GRID_BEST, count_vect, tfidf_transformer, multiclass = True)
results.to_csv('results_speaker_pairs.csv')


###################################
# Save the final model as a pickle file
X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data("../data/movies_features.p", FEATURE_COLS)
results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, CLASSIFIER_FINAL, GRID_FINAL, count_vect, tfidf_transformer)
pickle.dump(classifier_objects[0], open('mnb_final.p', 'wb'))
'''