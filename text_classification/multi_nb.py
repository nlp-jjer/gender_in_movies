# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:12:50 2018

@author: JoanWang
"""
import pipeline


X_train, X_test, y_train, y_test = pipeline.prepare_data("../movies.p")
results = pipeline.fit_models(X_train, X_test, y_train, y_test, pipeline.CLASSIFIERS, pipeline.GRID) 
results.to_csv('results.csv')

