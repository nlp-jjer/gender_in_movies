import pickle
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functools import reduce



def create_movie_df(movies_df, movie_id):
    '''
    Take the entire movies dataframe and return the subset for a given movie,
    with counts of interactions by character pair.

        movies_df: movies dataframe
        movie: (string)
    '''

    df = movies_df[movies_df['movie_id'] == movie_id]
    counts = df.groupby(['char_id_from', 'char_id_to']).size().reset_index(name = 'count')
    df = df.merge(counts, on = ['char_id_from', 'char_id_to'])
    return df



def create_gender_dict(movie_df):
    '''
    Helper function for build_network():

    Take a movie dataframe from create_movie_df() and return
    a {character: gender} dictionary
    '''

    gender_from = pd.Series(movie_df['gender_from'].values,
                            index = movie_df['char_id_from']).to_dict()

    gender_to = pd.Series(movie_df['gender_to'].values,
                          index = movie_df['char_id_to']).to_dict()

    gender_dict = {**gender_from, **gender_to} #combines the two dictionaries

    return gender_dict



def build_network(movie_df):
    '''
    Take a movie-specific dataframe from create_movie_df() and return
    a networkx graph object with edges and nodes
    '''

    g = nx.from_pandas_edgelist(movie_df,
                                source = 'char_id_from',
                                target = 'char_id_to',
                                edge_attr = ['count'])

    gender_dict = create_gender_dict(movie_df)

    nx.set_node_attributes(g, name = 'gender', values = gender_dict)

    return g



def measure_centrality(movie_df, g):
    '''
    Take a networkx object from build_network() and return
    a dataframe with various measures of centrality
    by character
    '''

    gender_dict = create_gender_dict(movie_df)

    gender = pd.DataFrame(list(gender_dict.items()),
                          columns = ['char_id', 'gender'])

    degree = pd.DataFrame(list(nx.degree_centrality(g).items()),
                          columns=['char_id','degree'])

    closeness = pd.DataFrame(list(nx.closeness_centrality(g).items()),
                             columns=['char_id','closeness'])

    betweenness = pd.DataFrame(list(nx.betweenness_centrality(g).items()),
                               columns=['char_id','betweenness'])

    eigenvector = pd.DataFrame(list(nx.eigenvector_centrality(g).items()),
                               columns=['char_id','eigenvector'])

    dfs = [gender, degree, closeness, betweenness, eigenvector]

    centralities = reduce(lambda left, right: pd.merge(left, right, on='char_id'), dfs)

    return centralities



def centrality_by_gender(movie_df, g):
    '''
    Take a df of centralities from measure_centrality() and return
    a grouped df with averages by gender
    '''

    centralities = measure_centrality(movie_df, g)

    operations = {'degree': np.mean, 'closeness': np.mean,
                  'betweenness': np.mean, 'eigenvector': np.mean}

    grouped = centralities.groupby('gender').agg(operations).reset_index()

    grouped = grouped[grouped['gender'] != '?']

        #left the unknown gender characters in up until this point so that
        #connections with those characters could be included in the
        #centrality calculations

    grouped['overall_avg'] = (grouped['degree'] + grouped['closeness'] +
                             grouped['betweenness'] + grouped['eigenvector']) / 4

    return grouped



def run_all(movies_df, movie_id):
    '''
    Run all functions above to get the final dataframe of centrality
    by gender. Could be run in a loop over all movies...
    '''

    movie_df = create_movie_df(movies_df, movie_id)
    g = build_network(movie_df)
    centrality = centrality_by_gender(movie_df, g)

    return centrality
