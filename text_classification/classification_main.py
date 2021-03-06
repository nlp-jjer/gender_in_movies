# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:12:50 2018

@author: JoanWang
"""
import pickle
import numpy as np
import classification_pipeline as pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# define columns to use as features
FEATURE_COLS = ['words', 'agency_pos_prop','power_pos_prop', 'agency_neg_prop','power_neg_prop',
                'MT1','MT2', 'MT3','WT1','WT2','WT3']

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

# get normalization metrics for classifier using training data
male_class = pickle.load(open('../text_classification/male_avgs', 'rb'))
male_class_mean = np.nanmean(male_class)
male_class_sd = np.nanstd(male_class)

female_class = pickle.load(open('../text_classification/female_avgs', 'rb'))
female_class_mean = np.nanmean(female_class)
female_class_sd = np.nanstd(female_class)


def train_model(input_name, output_name, feature_cols, classifier_dict, grid_dict, genre = 'all', speaker_pairs = False):
    """
    input_name: pickle file containing preprocessed dataframe, before feature generation
    output_name: csv file to write results
    """
    X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = pipeline.prepare_data(input_name, feature_cols, genre=genre, speaker_pairs = speaker_pairs)
    results, classifier_objects = pipeline.fit_models(X_train, X_test, y_train, y_test, classifier_dict, grid_dict, count_vect, tfidf_transformer) 
    results.to_csv(output_name)
    
    return results, classifier_objects


def get_normalization_clf(movies_filename, movies_lines_filename, clf_object):
    '''
    Get average probabilities per gender for each movie in training set 
    (movies used to test and validate the classifer) to use as normalization factor for final scoring.
    
    training_movies: list of movies in training set (whole db minus holdout)
    training_df: lines of movies in the training_movies list, classified using
        chosen best classifier
    '''    
    training_movies = list(pickle.load(open(movies_filename, 'rb')).movie_id)
    training_lines = pickle.load(open(movies_lines_filename, 'rb'))
    clf_object = pickle.load(open(clf_object, 'rb'))
    predicted, pred_probs, training_df = pipeline.classify_unseen(training_lines, clf_object, FEATURE_COLS)

    male_avgs = []
    female_avgs = []
    for movie_id in training_movies:
        df = training_df[training_df.movie_id == movie_id]
        male, female = pipeline.calculate_class_probs(df)
        male_avgs.append(male)
        female_avgs.append(female)
        
    return male_avgs, female_avgs


if __name__ == '__main__':
    # train best classifier
    results, classifier_objects = train_model("../data/movies_lines_train.p", "results_csvs/results_lr_mnb.csv", FEATURE_COLS, CLASSIFIERS_BEST, GRID_BEST)
    pickle.dump(classifier_objects[0], open('mnb_final.p', 'wb'))   
    
    # get list of class probabilities for training set, save for later use when this module is called
    male_avgs, female_avgs = get_normalization_clf('../data/movies_train.p', 
                        '../data/movies_lines_train.p', "../text_classification/mnb_final.p")
    pickle.dump(female_avgs, open('female_avgs', 'wb'))
    pickle.dump(male_avgs, open('male_avgs', 'wb'))


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