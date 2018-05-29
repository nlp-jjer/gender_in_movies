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



def measure_centrality(movie_df):
    '''
    Take a networkx object from build_network() and return
    a dataframe with various measures of centrality
    by character
    '''

    g = build_network(movie_df)

    gender_dict = create_gender_dict(movie_df)

    gender = pd.DataFrame(list(gender_dict.items()),
                          columns = ['char_id', 'gender'])

    degree = pd.DataFrame(list(nx.degree_centrality(g).items()),
                          columns=['char_id','degree'])

    #closeness = pd.DataFrame(list(nx.closeness_centrality(g).items()),
    #                         columns=['char_id','closeness'])

    betweenness = pd.DataFrame(list(nx.betweenness_centrality(g).items()),
                               columns=['char_id','betweenness'])

    #eigenvector = pd.DataFrame(list(nx.eigenvector_centrality(g, max_iter = 1000).items()),
    #                           columns=['char_id','eigenvector'])

    dfs = [gender, degree, betweenness]

    centralities = reduce(lambda left, right: pd.merge(left, right, on='char_id'), dfs)

    return centralities



def centrality_by_gender(movie_df):
    '''
    Take a df of centralities from measure_centrality() and return
    a grouped df with averages by gender
    '''

    centralities = measure_centrality(movie_df)

    centralities = centralities[centralities['gender'] != '?']

        #left the unknown gender characters in up until this point so that
        #connections with those characters could be included in the
        #centrality calculations

    if centralities.shape[0] == 0: #for movies where all genders are unknown
        return None

    #operations = {'degree': np.mean, 'closeness': np.mean,
    #              'betweenness': np.mean, 'eigenvector': np.mean}

    operations = {'degree': np.mean, 'betweenness': np.mean}

    grouped = centralities.groupby('gender').agg(operations).reset_index()

    #grouped['overall_avg'] = (grouped['degree'] + grouped['closeness'] +
    #                         grouped['betweenness'] + grouped['eigenvector']) / 4

    grouped['movie_id'] = movie_df['movie_id']
    grouped['year'] = movie_df['movie_year']
    grouped['genre'] = movie_df['genre']

    return grouped



def get_centrality(movie_df, centrality_type, gender):

    grouped = centrality_by_gender(movie_df)
    centrality = float(grouped[grouped['gender'] == gender][centrality_type])

    return centrality



def draw_graph(g, movie_df):

    gender_dict = create_gender_dict(movie_df)

    #setup
    plt.figure(figsize = (8, 8))
    layout = nx.spring_layout(g, iterations = 50)

    #nodes
    female = [char for char, gender in gender_dict.items() if gender == 'f']
    male = [char for char, gender in gender_dict.items() if gender == 'm']
    unknown = [char for char, gender in gender_dict.items() if gender == '?']

    #node sizes
    female_sizes = [g.degree(char) * 50 for char in female]
    male_sizes = [g.degree(char) * 50 for char in male]
    unknown_sizes = [g.degree(char) * 50 for char in unknown]

    #edge widths based on number of interactions between two nodes
    lst = list(g.edges(data = True))
    counts = np.array([d['count'] for c1, c2, d in lst])

    #normalize the counts
    minimum = np.min(counts)
    maximum = np.max(counts)

    counts = (counts - minimum) / (maximum - minimum) + 0.05 #add a little bit so none are zero
    counts = counts * 5

    movie_id = movie_df['movie_id'][1]

    characters = pickle.load(open('../data/characters.p', 'rb'))
    characters = characters[characters['movie_id'] == movie_id]

    movie_df = movie_df.merge(characters, on = 'movie_id')
    title = characters['movie_title'].unique()[0]

    #top three characters for labels
    f = dict(zip(female, female_sizes))
    m = dict(zip(male, male_sizes))
    fm = pd.DataFrame(list({**f, **m}.items()), columns=['character_id','size'])
    top_3 = list(fm.sort_values('size', ascending = False).head(3)['character_id'])
    top_3_chars = characters[characters['character_id'].isin(top_3)]
    fm = fm.merge(top_3_chars, on = 'character_id')
    fm = pd.Series(fm['name'].values, index = fm['character_id']).to_dict()

    nx.draw_networkx_nodes(g, layout, nodelist = female, node_color = '#ADA342', node_size = female_sizes)
    nx.draw_networkx_nodes(g, layout, nodelist = male, node_color = '#40616c', node_size = male_sizes)
    nx.draw_networkx_nodes(g, layout, nodelist = unknown, node_color = '#f6ca0e', node_size = unknown_sizes)

    nx.draw_networkx_edges(g, layout, width=counts, edge_color="#AEAEAE")

    plt.axis('off')

    node_labels = fm
    nx.draw_networkx_labels(g, layout, labels=node_labels)

    plt.title(title)

    legend_elements = [Line2D([0], [0], color = '#ADA342', marker = 'o', label = 'Female', markersize = 8, linestyle = 'None'),
                       Line2D([0], [0], color = '#40616c', marker = 'o', label = 'Male', markersize = 8, linestyle = 'None'),
                       Line2D([0], [0], color = '#f6ca0e', marker = 'o', label = 'Unknown', markersize = 8, linestyle = 'None')]

    plt.legend(handles = legend_elements, loc = 'best', frameon = False, handletextpad = 0)

    plt.show()



def run_all(movies_df, movie_id, draw_the_graph = False):
    '''
    Run all functions above to get the final dataframe of centrality
    by gender. Could be run in a loop over all movies...
    '''

    movie_df = create_movie_df(movies_df, movie_id)
    g = build_network(movie_df)
    centrality = centrality_by_gender(movie_df)

    if draw_the_graph:
        draw_graph(g, movie_df)

    return centrality



def all_movies(movies_df, filename):

    filepath = '../data/' + filename + '.p'

    ids = list(movies_df['movie_id'].unique())
    dfs = []

    for id in ids:
        print(id)
        df = run_all(movies_df, id)

        if df is None: #movies with all unknown genders will have a None value
            continue

        dfs.append(df)

    final = pd.concat(dfs)
    pickle.dump(final, open(filepath, 'wb'))
    print("it's pickled!")



#if __name__ == '__main__':
#    holdout = pickle.load(open('../data/movies_lines_holdout.p', 'rb'))
#    train = pickle.load(open('../data/movies_lines_train.p', 'rb'))

#    all_movies(holdout, 'holdout_centrality')
#    all_movies(train, 'train_centrality')
