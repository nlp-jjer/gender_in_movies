import pickle
import pandas as pd
import numpy as np

from scoring import Movie

movies = pickle.load(open('../data/movies_lines_holdout.p', 'rb'))
ids = pickle.load(open('../data/movies_holdout.p', 'rb'))

ids = pd.concat([ids, pd.DataFrame(columns = ['female_prop', 'cosine_sim',
                                             'class_ratio', 'nt_degree',
                                             'nt_betweenness', 'final_score'])])

classifier_object = pickle.load(open('../text_classification/mnb_final.p', 'rb'))

def score_movie(x, classifier_object):
    id = x['movie_id']
    movie_df = movies[movies['movie_id'] == id]

    movie = Movie(movie_df, id)

    try:
        movie_score = movie.score_movie(classifier_object)

        x[['female_prop', 'cosine_sim',
            'class_ratio', 'nt_degree',
            'nt_betweenness', 'final_score']] = [movie.female_prop, movie.cosine_sim,
                                                 movie.ratio_of_probs, movie.network_degree,
                                                 movie.network_btw, movie.final_score]

    except (ValueError, TypeError):
        x[['female_prop', 'cosine_sim',
           'class_ratio', 'nt_degree',
           'nt_betweenness', 'final_score']] = [np.NaN, np.NaN, np.NaN,
                                                np.NaN, np.NaN, np.NaN]

ids.apply(score_movie, args = (classifier_object, ), axis = 1, reduce = False)
