import pandas as pd
import numpy as np
import pickle
import nltk
import gensim, logging
import cython
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# DataSet
# Movies_holdout= pickle.load(open("../data/movies_lines_holdout.p", 'rb'))

# Calculate Raw Cosine Similarity Score
def calc_cosim_gender(movie_df):
    # Convert each movie texts into a long string/document, separated by gender
    # Context is not considered in this scoring method
    mcosim = movie_df.groupby(['movie_id', 'gender_from']).apply(lambda x: " ".join(x['words'])).reset_index(name='raw_text')
    mcosim = mcosim.pivot(index='movie_id', columns='gender_from', values='raw_text').reset_index().fillna('Empty')

    # Vectorize Documents (Sentences) using TF-IDF
    mcosim['fit'] = mcosim[['f','m']].apply(lambda x: TfidfVectorizer().fit_transform([x[0], x[1]]), axis=1)

    # Calculate cosine similarity score
    mcosim['gender_cosim'] = mcosim['fit'].apply(lambda x: cosine_similarity(x[0], x[1])[0,0])

    mcosim = pd.merge(mcosim, movie_df[['movie_id', 'genre']], how='inner', on='movie_id')
    return mcosim[['movie_id', 'gender_cosim', 'genre']].drop_duplicates().reset_index(drop=True)


# Normalized cosine similarity score by genre and return score dataframe
def normalize_cosim(cosim_df):
    desc_stat = cosim_df.describe().reset_index()

    cosim_mean = desc_stat.loc['mean', 'gender_cosim']
    cosim_std = desc_stat.loc['std', 'gender_cosim']

    cosim_df['norm_cosim'] = (cosim_df['gender_cosim']-cosim_mean)/cosim_std

    return cosim_df[]


# Calculate cosim score for One Movie
def one_cosim_score(movie_df):
    mcosim = calc_cosim_gender(movie_df)
    mcosim = normalize_cosim(mcosim)

    return float(mcosim['norm_cosim'])
