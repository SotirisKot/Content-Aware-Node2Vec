from __future__ import print_function

import argparse
from collections import OrderedDict
from pprint import pprint

import numpy as np
import networkx as nx
from tqdm import tqdm
import json
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
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

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


def create_train_test_splits_1st_way(percent_pos, percent_neg, graph, percent_dev=None):
    #TODO add the code that creates a developemnt set----MUST

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
    # if not args.directed:
    #     original_connected_comps = nx.number_connected_components(graph)
    #     print('Connected components: ', original_connected_comps)

    print("Converting graph back to undirected..")
    graph = graph.to_undirected()
    print('Creating positive test samples..')
    # shuffle the edges and iterate over them creating the test set
    np.random.shuffle(all_edges)
    for edge in tqdm(all_edges):
        # if idx % 100 == 0:
        #     print('Edge: {}/{}'.format(idx, num_edges))
        #     print('Added: ', counter2)
        #     print('Not Added: ', counter1)
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

    print("Added: {} number of edges to positive test".format(counter2))

    # now create false edges for test and train sets..making sure the edge is not a real edge
    # and not already sampled
    # first for test_set
    print('Creating negative test samples..')
    test_false_edges = set()
    while len(test_false_edges) < 61849:
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
    while len(train_false_edges) < 294692:
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

    odir = 'datasets/isa_easy_splits'
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


def return_parents(graph, node, hops=None):
    if hops is not None:
        hops += 1
    return list(graph.predecessors(node)), hops


def create_train_test_splits_2nd_way(percent_pos, percent_neg, graph, percent_dev=None):

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
    # if not args.directed:
    #     original_connected_comps = nx.number_connected_components(graph)
    #     print('Connected components: ', original_connected_comps)

    np.random.shuffle(all_edges)
    # shuffle the edges and iterate over them creating the test set
    # create false edges for test and train sets..making sure the edge is not a real edge
    # and not already sampled
    # first for test_set
    # to generate this type of negative examples we must convert the graph to directed
    print('Creating negative test samples..')
    test_false_edges = set()
    while len(test_false_edges) < 61849:
        for node in tqdm(all_nodes):
            hop = 1
            parents = []

            # for parents
            # we can skip the og_parent and not append it in the list.
            # this is the first hop
            og_parent, hop = return_parents(graph, node, hop)
            while hop != 6:

                # if a node has more than one parents generate a random number and select a random one
                if len(og_parent) != 0:
                    og_parent, hop = return_parents(graph, og_parent[np.random.randint(0, len(og_parent))], hop)
                    if len(og_parent) != 0:
                        parents.append(og_parent[np.random.randint(0, len(og_parent))])
                # no more parents
                else:
                    break

            # the first parent in the list is 2 hops away from our focus node.
            if len(parents) > 2:
                # select a random parent +1 hop away and get his children.
                rand_par = parents[int(np.random.uniform(0, len(parents)))]

                # get his children and select a random to node to create a negative example with
                children = list(graph.successors(rand_par))

                # we must exclude ancestors of the focus node from the sampling..so we remove the common elements from these lists
                cleaned_children = [child for child in children if child not in parents]
                if len(cleaned_children) != 0:
                    path_exists = False
                    rev_path_exists = False
                    rand_child = cleaned_children[int(np.random.uniform(0, len(cleaned_children)))]
                    reachable_from_v1 = nx.connected._plain_bfs(graph, node)
                    if rand_child in reachable_from_v1:
                        path_exists = True

                    reachable_from_v1 = nx.connected._plain_bfs(graph, rand_child)
                    if node in reachable_from_v1:
                        rev_path_exists = True

                    # the focus node might belong in the children so we must check it
                    if node != rand_child and not path_exists and not rev_path_exists:
                        sampled_edge = (node, rand_child)
                        sampled_edge_rev = (rand_child, node)
                    else:
                        continue
                else:
                    continue
                # check if we sampled a real edge or an already sampled one
                if sampled_edge in all_edges_set or sampled_edge_rev in all_edges_set:
                    continue
                if sampled_edge in test_false_edges or sampled_edge_rev in test_false_edges:
                    continue
                # everything is ok so we add the fake edge to the test_set
                if len(test_false_edges) == 61849:
                    # we generated all the false edges we wanted
                    break
                else:
                    test_false_edges.add(sampled_edge)
            else:
                continue

    # do the same for the train_set
    print('Creating negative training samples...')
    train_false_edges = set()
    while len(train_false_edges) < 294692:
        for node in tqdm(all_nodes):
            hop = 1
            parents = []

            # for parents
            # we can skip the og_parent and not append it in the list.
            # this is the first hop
            og_parent, hop = return_parents(graph, node, hop)
            while hop != 6:

                # if a node has more than one parents generate a random number and select a random one
                if len(og_parent) != 0:
                    og_parent, hop = return_parents(graph, og_parent[np.random.randint(0, len(og_parent))], hop)
                    if len(og_parent) != 0:
                        parents.append(og_parent[np.random.randint(0, len(og_parent))])
                # no more parents
                else:
                    break

            # the first parent in the list is 2 hops away from our focus node.
            if len(parents) > 2:
                # select a random parent and get his children.
                rand_par = parents[int(np.random.uniform(0, len(parents)))]

                # get his children and select a random to node to create a negative example with
                children = list(graph.successors(rand_par))

                # we must exclude ancestors of the focus node from the sampling..so we remove the common elements from these lists
                cleaned_children = [child for child in children if child not in parents]
                if len(cleaned_children) != 0:
                    path_exists = False
                    rev_path_exists = False
                    rand_child = cleaned_children[int(np.random.uniform(0, len(cleaned_children)))]
                    reachable_from_v1 = nx.connected._plain_bfs(graph, node)
                    if rand_child in reachable_from_v1:
                        path_exists = True

                    reachable_from_v1 = nx.connected._plain_bfs(graph, rand_child)
                    if node in reachable_from_v1:
                        rev_path_exists = True

                    # the focus node might belong in the children so we must check it
                    if node != rand_child and not path_exists and not rev_path_exists:
                        sampled_edge = (node, rand_child)
                        sampled_edge_rev = (rand_child, node)
                    else:
                        continue
                else:
                    continue
                # check if we sampled a real edge or an already sampled one
                if sampled_edge in all_edges_set or sampled_edge_rev in all_edges_set:
                    continue
                if sampled_edge in test_false_edges or sampled_edge_rev in test_false_edges:
                    continue
                if sampled_edge in train_false_edges or sampled_edge_rev in train_false_edges:
                    continue
                # everything is ok so we add the fake edge to the test_set
                if len(train_false_edges) == 294692:
                    break
                else:
                    train_false_edges.add(sampled_edge)
            else:
                continue

    # phrase_dic = pickle.load(open('data_utilities/part_of/part_of_phrase_dic.p', 'rb'))
    print("Total number of negative test samples: {}".format(len(test_false_edges)))
    print("Total number of negative train samples: {}".format(len(train_false_edges)))

    print("Converting graph to undirected..")
    graph = graph.to_undirected()
    print('Creating positive test samples..')
    # shuffle the edges and iterate over them creating the test set
    np.random.shuffle(all_edges)
    patience = 0
    for edge in tqdm(all_edges):
        # if idx % 100 == 0:
        #     print('Edge: {}/{}'.format(idx, num_edges))
        #     print('Added: ', counter2)
        #     print('Not Added: ', counter1)
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
        elif len(test_edges) == 61849:
            patience += 1

        if patience == 3:
            break

    print("Added: {} number of edges to positive test".format(counter2))
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

    odir = 'datasets/isa_hard_splits/'
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


def main(args):
    nx_G = read_graph(file=args.input, get_connected_graph=False, remove_selfloops=True, get_directed=False)
    # children = list(nx_G.successors(18060))
    # print(len(children))
    # print(children)
    # phrase_dic = pickle.load(open(config.phrase_dic, 'rb'))
    # max_len = 0
    # phr_str = None
    # total_len = 0
    # for id, phrase in phrase_dic.items():
    #     phr = phrase.split()
    #     total_len += len(phr)
    #     if len(phr) > max_len:
    #         phr_str = phr
    #         max_len = len(phr)
    # avg =float(total_len) / float(len(phrase_dic))
    # print('The average is: ', avg)
    #
    # print('The maximun length of a sentence is: ', max_len)
    # print('The phrase', phr_str)
    # exit(0)
    # for child in children:
    #     print(phrase_dic[child])
    # exit(0)
    # print(nx_G.number_of_nodes(), nx_G.number_of_edges())
    # print()
    train_pos = pickle.load(open(config.train_pos, 'rb'))
    test_pos = pickle.load(open(config.test_pos, 'rb'))
    train_neg = pickle.load(open(config.train_neg, 'rb'))
    test_neg = pickle.load(open(config.test_neg, 'rb'))

    # train_pos, train_neg, test_pos, test_neg = create_train_test_splits_1st_way(0.5, 0.5, nx_G)
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

    # print("Created new Dataset..")
    # exit(0)
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
