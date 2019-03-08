from __future__ import print_function

import argparse
import networkx as nx
import pickle
import numpy as np
import os
import torch
from tqdm import tqdm
from pprint import pprint
import codecs
import config
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?',
                        default='C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/isa2_relations.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='isa_average_words_link_predict.emb',
                        help='Embeddings path')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(file, get_connected_graph=True, remove_selfloops=True):
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

    if not args.directed:
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


bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                   '').strip().lower()).split()).strip()


def get_index(w, vocab):
    try:
        return vocab[w]
    except KeyError:
        return vocab['UNKN']


def phr2idx(phr, word_vocab):
    p = [get_index(t, word_vocab) for t in phr.split()]
    return p


def clean_dictionary(phrase_dic):
    for nodeid, phrase in phrase_dic.items():
        phrase_dic[nodeid] = tokenize(phrase)
    return phrase_dic


def tokenize(x):
    return bioclean(x)


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


# in the paper they get edge embeddings..they use a lot of methods but for link prediction they state that
# the Hadamard product is highly stable and gives the best performance.
def get_edge_embeddings(edge_list, node_embeddings, phrase_dic):
    # create a list containing edge embeddings
    edge_embeddings = []
    for idx, edge in enumerate(edge_list):
        phrase_node1 = get_average_embedding(phrase_dic[edge[0]], node_embeddings)
        phrase_node2 = get_average_embedding(phrase_dic[edge[1]], node_embeddings)
        hadamard = np.multiply(phrase_node1, phrase_node2)
        edge_embeddings.append(hadamard)
    edge_embeddings = np.array(edge_embeddings)
    return edge_embeddings


def load_embeddings(file):
    node_embeddings = {}
    with codecs.open("{}".format(file), 'r', 'utf-8') as embeddings:
        embeddings.readline()
        for i, line in enumerate(embeddings):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            assert len(embedding) == 30
            node_embeddings[word] = embedding
    return node_embeddings


def get_average_embedding(phrase, node_embeddings):
    length = len(phrase)
    phrase = phrase.split()
    for idx, word in enumerate(phrase):
        if idx == 0:
            sum = node_embeddings[word]
        else:
            sum = np.add(sum, node_embeddings[word])

    average_embedding = np.divide(sum, float(length))
    return average_embedding


def main(args):
    train_pos = pickle.load(open(config.train_pos, 'rb'))
    test_pos = pickle.load(open(config.test_pos, 'rb'))
    train_neg = pickle.load(open(config.train_neg, 'rb'))
    test_neg = pickle.load(open(config.test_neg, 'rb'))

    # train_pos, train_neg, test_pos, test_neg = create_train_test_splits(0.5, 0.5, nx_G)
    # train_neg, test_neg = create_train_test_splits(0.5, 0.5, nx_G)
    print('Number of positive training samples: ', len(train_pos))
    print('Number of negative training samples: ', len(train_neg))
    print('Number of positive testing samples: ', len(test_pos))
    print('Number of negative testing samples: ', len(test_neg))
    train_graph = read_graph(
        file=config.train_graph,
        get_connected_graph=False,
        remove_selfloops=False)
    print(
        'Train graph created: {} nodes, {} edges'.format(train_graph.number_of_nodes(), train_graph.number_of_edges()))
    print('Number of connected components: ', nx.number_connected_components(train_graph))
    print(
        'Train graph created: {} nodes, {} edges'.format(train_graph.number_of_nodes(), train_graph.number_of_edges()))
    print('Number of connected components: ', nx.number_connected_components(train_graph))
    node_embeddings = load_embeddings('/home/sotiris/Downloads/part_of_rnn_test_words_link_predict.emb')
    phrase_dic = clean_dictionary(pickle.load(
        open(config.phrase_dic, 'rb')))

    # for training
    # for training
    train_pos_edge_embs = get_edge_embeddings(train_pos, node_embeddings, phrase_dic)
    train_neg_edge_embs = get_edge_embeddings(train_neg, node_embeddings, phrase_dic)
    train_set = np.concatenate([train_pos_edge_embs, train_neg_edge_embs])

    # labels: 1-> link exists, 0-> false edge
    train_labels = np.zeros(len(train_set), dtype=int)
    train_labels[:len(train_pos_edge_embs)] = 1
    # for testing
    test_pos_edge_embs = get_edge_embeddings(test_pos, node_embeddings, phrase_dic)
    test_neg_edge_embs = get_edge_embeddings(test_neg, node_embeddings, phrase_dic)
    test_set = np.concatenate([test_pos_edge_embs, test_neg_edge_embs])

    # labels: 1-> link exists, 0-> false edge
    test_labels = np.zeros(len(test_set), dtype=int)
    test_labels[:len(test_pos_edge_embs)] = 1

    # train the classifier and evaluate in the test set
    classifier = LogisticRegression(random_state=1997)
    classifier.fit(train_set, train_labels)
    print('done')

    # evaluate
    test_preds = classifier.predict_proba(test_set)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, test_preds)
    test_auc = auc(false_positive_rate, true_positive_rate)
    test_roc = roc_auc_score(test_labels, test_preds)
    print('node2vec Test ROC score: ', str(test_roc))
    print('node2vec Test AUC score: ', str(test_auc))


if __name__ == "__main__":
    args = parse_args()
    main(args)
