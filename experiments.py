from __future__ import print_function

import argparse
import collections
import re
from pprint import pprint
import numpy as np
import networkx as nx
from tqdm import tqdm
import node2vec
import os
import pickle
import config
from train_node2vec import Node2Vec


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?',
                        default=config.input_edgelist,
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default=config.output_file,
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=config.dimensions,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=config.walk_length,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=config.num_walks,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=config.window_size,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=config.epochs, type=int,
                        help='Number of epochs')

    parser.add_argument('--p', type=float, default=config.p,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=config.q,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(file, get_connected_graph=True, remove_selfloops=True, get_directed=False):
    if args.weighted:
        G = nx.read_edgelist(file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    print('Graph created!!!')

    if remove_selfloops:
        # remove the edges with selfloops
        for node in G.nodes_with_selfloops():
            G.remove_edge(node, node)

    if not get_directed:
        G = G.to_undirected()
        if get_connected_graph and not nx.is_connected(G):
            connected_sub_graph = max(nx.connected_component_subgraphs(G), key=len)
            print('Initial graph not connected...returning the largest connected subgraph..')
            return connected_sub_graph
        else:
            print('Returning undirected graph!')
            return G
    else:
        print('Returning directed graph!')
        return G


def learn_embeddings(walks, train_pos=None, train_neg=None, test_pos=None, test_neg=None, eval_bool=False, embeddings_file=None, checkpoint_file=None):
    if not eval_bool:
        print('Creating walk corpus..')
        exit()
        model = Node2Vec(walks=walks, output_file=args.output, walk_length=args.walk_length,
                         embedding_dim=args.dimensions,
                         epochs=args.iter, batch_size=config.batch_size, window_size=args.window_size, neg_sample_num=config.neg_samples)
        print('Optimization started...')
        model.train()
        embeddings = model.wv
        return embeddings
    else:
        model = Node2Vec(walks=None, output_file=args.output, walk_length=args.walk_length,
                         embedding_dim=args.dimensions,
                         epochs=args.iter, batch_size=config.batch_size, window_size=args.window_size, neg_sample_num=config.neg_samples)
        print('Evaluation started...')
        #TODO add the file from which the embeddings will be loaded
        model.eval(train_pos, train_neg, test_pos, test_neg, embeddings_file, checkpoint_file)


bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                            t.replace('"', '').replace('/', ' ').replace('\\', '').replace("'",
                                                                                           '').strip().lower()).split()


def tokenize(x):
    return bioclean(x)


def get_index(w, vocab):
    try:
        return vocab[w]
    except KeyError:
        return vocab['UNKN']


def phr2idx(phr, word_vocab):
    p = [get_index(t, word_vocab) for t in phr]
    return p


def clean_dictionary(phrase_dic):
    for nodeid, phrase in phrase_dic.items():
        phrase_dic[nodeid] = tokenize(phrase)
    return phrase_dic


def main(args):
    nx_G = read_graph(file=args.input, get_connected_graph=True, remove_selfloops=True, get_directed=False)

    print('Original Graph: nodes: {}, edges: {}'.format(nx_G.number_of_nodes(), nx_G.number_of_edges()))
    print()
    train_pos = pickle.load(open(config.train_pos, 'rb'))
    test_pos = pickle.load(open(config.test_pos, 'rb'))
    train_neg = pickle.load(open(config.train_neg, 'rb'))
    test_neg = pickle.load(open(config.test_neg, 'rb'))

    print('Number of positive training samples: ', len(train_pos))
    print('Number of negative training samples: ', len(train_neg))
    print('Number of positive testing samples: ', len(test_pos))
    print('Number of negative testing samples: ', len(test_neg))
    train_graph = read_graph(
        file=config.train_graph,
        get_connected_graph=False,
        remove_selfloops=False,
        get_directed=False)

    print(
        'Train graph created: {} nodes, {} edges'.format(train_graph.number_of_nodes(), train_graph.number_of_edges()))
    print('Number of connected components: ', nx.number_connected_components(train_graph))
    if config.train:
        if config.resume_training:
            _ = learn_embeddings(walks=None)
        else:
            G = node2vec.Graph(train_graph, args.directed, args.p, args.q)
            G.preprocess_transition_probs()
            walks = G.simulate_walks(args.num_walks, args.walk_length)
            # learn the embeddings
            _ = learn_embeddings(walks)

    embeddings_file = None
    checkpoint_file = None
    if config.evaluate:

        if config.model is not 'rnn':
            embeddings_file = config.embeddings_dir + config.output_file
        else:
            checkpoint_file = config.checkpoint_dir + config.checkpoint_name
            print(checkpoint_file)
        # evaluate embeddings in link prediction
        _ = learn_embeddings(walks=None,
                             train_pos=train_pos,
                             train_neg=train_neg,
                             test_pos=test_pos,
                             test_neg=test_neg,
                             eval_bool=True,
                             embeddings_file=embeddings_file,
                             checkpoint_file=checkpoint_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
