from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import node2vec
import os
import pickle
from tqdm import tqdm
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from node2vec_pytorch import Node2Vec


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?',
                        default='C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/isa2_relations.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='isa_average_words_link_predict.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=30,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=5,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
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


def learn_embeddings(walks):
    # walks = [map(str, walk) for walk in walks] # this will work on python2 but not in python3
    print('Creating walk corpus..')
    # walks = [list(map(str, walk)) for walk in walks]  # this is for python3
    # odir = '/home/paperspace/sotiris/thesis/'
    # with open('{}.p'.format(os.path.join(odir, 'walks')), 'wb') as dump_file:
    #     pickle.dump(walks, dump_file)
    model = Node2Vec(walks=walks, output_file=args.output, embedding_dim=args.dimensions,
                     epochs=args.iter, batch_size=32, window_size=args.window_size, neg_sample_num=5)
    print('Optimization started...')
    model.train()
    embeddings = model.wv
    return embeddings


def create_train_test_splits(percent_pos, percent_neg, graph):
    # This method creates the train,test splits...from the dataset
    # we remove some percentage of the existing edges(ensuring the graph remains connected)
    # and use them as positive samples
    # then we sample the same amount of non-existing edges, using them as negative samples
    # then do the same for testing set.
    np.random.seed(0)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_pos_train_edges = int(num_edges * percent_pos)
    num_neg_train_edges = int(num_edges * percent_neg)
    num_pos_test_edges = num_edges - num_pos_train_edges
    num_neg_test_edges = num_edges - num_neg_train_edges
    all_edges = [edge for edge in graph.edges]
    all_nodes = [node for node in graph.nodes]
    all_edges_set = set(all_edges)  # for quick access
    train_edges = set(all_edges)
    test_edges = set()
    counter1 = 0
    counter2 = 0
    if not args.directed:
        original_connected_comps = nx.number_connected_components(graph)
        print('Connected components: ', original_connected_comps)

    print('Creating positive test samples..')
    # shuffle the edges and iterate over them creating the test set
    np.random.shuffle(all_edges)
    for idx, edge in enumerate(all_edges):
        if idx % 100 == 0:
            print('Edge: {}/{}'.format(idx, num_edges))
            print('Added: ', counter2)
            print('Not Added: ', counter1)
        node1 = edge[0]
        node2 = edge[1]
        # make sure that the graph remains connected
        # --from https://github.com/lucashu1/link-prediction/blob/master/gae/preprocessing.py
        graph.remove_edge(node1, node2)
        # if not args.directed:
        #     if nx.number_connected_components(graph) > original_connected_comps:
        #         graph.add_edge(node1, node2)
        #         continue
        reachable_from_v1 = nx.connected._plain_bfs(graph, edge[0])
        if edge[1] not in reachable_from_v1:
            graph.add_edge(node1, node2)
            counter1 += 1
            continue
        # remove edges from the train_edges set and add them to the test_edges set --positive samples
        if len(test_edges) < num_pos_test_edges:
            test_edges.add(edge)
            train_edges.remove(edge)
            counter2 += 1
        elif len(test_edges) == num_pos_test_edges:
            if not args.directed:
                graph.add_edge(node1, node2)
            break

    # now create false edges for test and train sets..making sure the edge is not a real edge
    # and not already sampled
    # first for test_set
    print('Creating negative test samples..')
    test_false_edges = set()
    while len(test_false_edges) < num_neg_test_edges:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        # we dont want to sample the same node
        if idx_i == idx_j:
            continue

        sampled_edge = (all_nodes[idx_i], all_nodes[idx_j])
        # check if we sampled a real edge or an already sampled edge
        if sampled_edge in all_edges_set:
            continue
        if sampled_edge in test_false_edges:
            continue
        # everything is ok so we add the fake edge to the test_set
        test_false_edges.add(sampled_edge)

    # do the same for the train_set
    print('Creating negative training samples...')
    train_false_edges = set()
    while len(train_false_edges) < num_neg_train_edges:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        # we don't want to sample the same node
        if idx_i == idx_j:
            continue

        sampled_edge = (all_nodes[idx_i], all_nodes[idx_j])
        # check if we sampled a real edge or an already sampled edge
        if sampled_edge in all_edges_set:
            continue
        if sampled_edge in test_false_edges:
            continue
        if sampled_edge in train_false_edges:
            continue
        # everything is ok so we add the fake edge to the train_set
        train_false_edges.add(sampled_edge)

    # asserts
    assert test_false_edges.isdisjoint(all_edges_set)
    assert train_false_edges.isdisjoint(all_edges_set)
    assert test_false_edges.isdisjoint(train_false_edges)
    assert test_edges.isdisjoint(train_edges)

    # convert them back to lists and return them
    train_pos = list(train_edges)
    train_neg = list(train_false_edges)
    test_pos = list(test_edges)
    test_neg = list(test_false_edges)

    odir = 'C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/isa-undirected-dataset-train-test-splits'
    if not os.path.exists(odir):
        os.makedirs(odir)

    # save the splits
    with open("{}.p".format(os.path.join(odir, 'isa_train_pos')), 'wb') as dump_file:
        pickle.dump(train_pos, dump_file)
    with open("{}.p".format(os.path.join(odir, 'isa_train_neg')), 'wb') as dump_file:
        pickle.dump(train_neg, dump_file)
    with open("{}.p".format(os.path.join(odir, 'isa_test_pos')), 'wb') as dump_file:
        pickle.dump(test_pos, dump_file)
    with open("{}.p".format(os.path.join(odir, 'isa_test_neg')), 'wb') as dump_file:
        pickle.dump(test_neg, dump_file)

    nx.write_edgelist(graph, os.path.join(odir, 'isa_train_graph_undirected.edgelist'))
    return train_pos, train_neg, test_pos, test_neg


# in the paper they get edge embeddings..they use a lot of methods but for link prediction they state that
# the Hadamard product is highly stable and gives the best performance.
def get_edge_embeddings(edge_list, node_embeddings):
    # create a list containing edge embeddings
    edge_embeddings = []
    for edge in edge_list:
        emb_node1 = node_embeddings[str(edge[0])]
        emb_node2 = node_embeddings[str(edge[1])]
        hadamard = np.multiply(emb_node1, emb_node2)
        edge_embeddings.append(hadamard)
    edge_embeddings = np.array(edge_embeddings)
    return edge_embeddings


def load_embeddings(file):
    node_embeddings = {}
    odir = 'C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/embeddings/'
    with open("{}".format(os.path.join(odir, file)), 'r') as embeddings:
        embeddings.readline()
        for i, line in enumerate(embeddings):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            assert len(embedding) == 128
            node_embeddings[word] = embedding
    return node_embeddings


def main(args):
    # nx_G = read_graph(file=args.input, get_connected_graph=False, remove_selfloops=True)
    # print(nx_G.number_of_nodes(), nx_G.number_of_edges())
    train_pos = pickle.load(open('/home/paperspace/sotiris/thesis/isa-undirected-dataset'
                                 '-train-test-splits/isa_train_pos.p', 'rb'))
    test_pos = pickle.load(open('/home/paperspace/sotiris/thesis/isa-undirected-dataset'
                                '-train-test-splits/isa_test_pos.p', 'rb'))
    train_neg = pickle.load(
        open('/home/paperspace/sotiris/thesis/isa-undirected-dataset'
             '-train-test-splits/isa_train_neg.p', 'rb'))
    test_neg = pickle.load(
        open('/home/paperspace/sotiris/thesis/isa-undirected-dataset'
             '-train-test-splits/isa_test_neg.p', 'rb'))
    # train_pos, train_neg, test_pos, test_neg = create_train_test_splits(0.5, 0.5, nx_G)
    # train_neg, test_neg = create_train_test_splits(0.5, 0.5, nx_G)
    print('Number of positive training samples: ', len(train_pos))
    print('Number of negative training samples: ', len(train_neg))
    print('Number of positive testing samples: ', len(test_pos))
    print('Number of negative testing samples: ', len(test_neg))
    train_graph = read_graph(
        file='/home/paperspace/sotiris/thesis/isa-undirected-dataset-train-test-splits/isa_train_graph_undirected.edgelist',
        get_connected_graph=False, remove_selfloops=False)
    print(
        'Train graph created: {} nodes, {} edges'.format(train_graph.number_of_nodes(), train_graph.number_of_edges()))
    print('Number of connected components: ', nx.number_connected_components(train_graph))
    # G = node2vec.Graph(train_graph, args.directed, args.p, args.q)
    # G.preprocess_transition_probs()
    # walks = G.simulate_walks(args.num_walks, args.walk_length)
    walks = pickle.load(open('/home/paperspace/sotiris/thesis/walks.p'))
    # walks = [['1', '23345', '3356', '4446', '5354', '6123', '74657', '8445', '97890', '1022', '1133'],
    #          ['6914', '1022', '97890', '8445', '74657', '6123', '5354', '4446', '3356', '23345', '1'],
    #          ['6914', '1022', '97890', '8445', '74657', '6123', '5354', '4446', '3356', '23345', '1']]
    node_embeddings = learn_embeddings(walks)

    # for training
    train_pos_edge_embs = get_edge_embeddings(train_pos, node_embeddings)
    train_neg_edge_embs = get_edge_embeddings(train_neg, node_embeddings)
    train_set = np.concatenate([train_pos_edge_embs, train_neg_edge_embs])

    # labels: 1-> link exists, 0-> false edge
    train_labels = np.zeros(len(train_set))
    train_labels[:len(train_pos_edge_embs)] = 1

    # for testing
    test_pos_edge_embs = get_edge_embeddings(test_pos, node_embeddings)
    test_neg_edge_embs = get_edge_embeddings(test_neg, node_embeddings)
    test_set = np.concatenate([test_pos_edge_embs, test_neg_edge_embs])

    # labels: 1-> link exists, 0-> false edge
    test_labels = np.zeros(len(test_set))
    test_labels[:len(test_pos_edge_embs)] = 1

    # train the classifier and evaluate in the test set
    classifier = LogisticRegression(random_state=0)
    classifier.fit(train_set, train_labels)

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
