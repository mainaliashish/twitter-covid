"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : April 15, 2020
"""
import os

from tweeter_covid19.utils import read_pickle_data, mkdir
from tweeter_covid19.graph_generator import GraphGenerator

N_SETS = 10

if __name__ == '__main__':
    corpus_path = os.path.join('data', 'distance_based', 'processing', 'corpus')
    nodes_path = os.path.join('data', 'distance_based', 'processing', 'graph_nodes',
                              'nodes.pkl')
    log_path = os.path.join('data', 'distance_based', 'processing', 'graph_nodes',
                            'graph_resume.pkl')
    graph_path = os.path.join('data', 'distance_based', 'processing', 'graph_nodes',
                              'nodes')
    for fold in range(N_SETS):
        mkdir(os.path.join(corpus_path, 'set_' + str(fold + 1)))
        mkdir(os.path.join(nodes_path, 'set_' + str(fold + 1), 'nodes'))
        corpus = read_pickle_data(os.path.join(corpus_path, 'set_' + str(fold + 1), 'corpus_stemmer.pkl'))
        model = GraphGenerator(corpus=corpus)
        model.fit(graph_path=os.path.join(nodes_path, 'set_' + str(fold + 1), 'nodes'), write_path=
        os.path.join(nodes_path, 'set_' + str(fold + 1), 'nodes.pkl'))
        model.connect_edges(os.path.join(nodes_path, 'set_' + str(fold + 1), 'graph_resume.pkl'))
