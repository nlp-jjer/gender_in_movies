import pandas as pd
import numpy as np
import pickle
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class cosim:
    def __init__(self):
        self.train_df = pd.DataFrame()
        self.train_mean = 0
        self.train_std = 0

    def similarity(self, movie_df, train=False):
        if ('m' not in (set(one_movie['gender_from']))) | ('f' not in (set(one_movie['gender_from']))):
            return 1.0       
        movie_cm = movie_df.groupby(['movie_id', 'gender_from']).apply(lambda x: " ".join(x['words'])).reset_index(name='raw_text')
        movie_cm = movie_cm.pivot(index='movie_id', columns='gender_from', values='raw_text').reset_index().fillna('Empty')

        movie_cm['fit'] = movie_cm[['f','m']].apply(lambda x: TfidfVectorizer().fit_transform([x[0], x[1]]), axis=1)
        movie_cm['gender_cosim'] = movie_cm['fit'].apply(lambda x: 1-cosine_similarity(x[0], x[1])[0,0])

        movie_cm = pd.merge(movie_cm, movie_df[['movie_id', 'genre']], how='inner', on='movie_id')

        return_df = movie_cm[['movie_id', 'gender_cosim', 'genre']].drop_duplicates().reset_index(drop=True)

        if train:
            return return_df

        return float(return_df['gender_cosim'])

    def cosim_train(self, train_df):
        train_proc = self.similarity(train_df, train=True)

        train_stat = train_proc.describe()
        self.train_mean = train_stat.loc['mean', 'gender_cosim']
        self.train_std = train_stat.loc['std', 'gender_cosim']


    def norm_similarity(self, movie_df):
        cosim_df = self.similarity(movie_df, train=True)
        cosim_df['norm_cosim'] = (cosim_df['gender_cosim']-self.train_mean)/self.train_std

        return float(cosim_df['norm_cosim'])



movies_train = pickle.load(open("../data/movies_lines_train.p", 'rb'))    
    
movie_cosine = cosim()
movie_cosine.cosim_train(movies_train)



