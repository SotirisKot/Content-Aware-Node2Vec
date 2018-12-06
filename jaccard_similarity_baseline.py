from __future__ import print_function
import pickle
import numpy as np
import os
from tqdm import tqdm
from pprint import pprint
import codecs
import networkx as nx
import re
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score

bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                   '').strip().lower()).split()).strip()


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


def get_phrase_list(edge_list, phrase_dic):
    phrase_list = []
    for edge in edge_list:
        phrase0 = phrase_dic[edge[0]]
        phrase1 = phrase_dic[edge[1]]
        phrase_list.append((phrase0, phrase1))
    return phrase_list


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / len(a.union(b))


def main():
    train_pos = pickle.load(open(
        'isa-undirected-dataset-train-test-splits/isa_train_pos.p',
        'rb'))
    test_pos = pickle.load(
        open('isa-undirected-dataset-train-test-splits/isa_test_pos.p', 'rb'))
    train_neg = pickle.load(
        open('isa-undirected-dataset-train-test-splits/isa_train_neg.p', 'rb'))
    test_neg = pickle.load(
        open(
            'isa-undirected-dataset-train-test-splits/isa_test_neg.p',
            'rb'))
    print('Number of positive training samples: ', len(train_pos))
    print('Number of negative training samples: ', len(train_neg))
    print('Number of positive testing samples: ', len(test_pos))
    print('Number of negative testing samples: ', len(test_neg))
    # development_pos = test_pos[:20000]
    # development_neg = test_neg[:78000]
    # print('Number of positive development samples: ', len(development_pos))
    # print('Number of negative development samples: ', len(development_neg))
    phrase_dic = clean_dictionary(pickle.load(open('relation_utilities/isa/isa_reversed_dic.p', 'rb')))
    # for training
    thresholds = [0.7]
    train_pos_phrases = get_phrase_list(train_pos, phrase_dic)
    train_neg_phrases = get_phrase_list(train_neg, phrase_dic)
    train_set = train_pos_phrases + train_neg_phrases
    print('The length of the train set is: ', len(train_set))
    # labels: 1-> link exists, 0-> false edge
    train_labels = np.zeros(len(train_set), dtype=int)
    train_labels[:len(train_pos_phrases)] = 1
    counter1 = 0
    counter2 = 0
    auc_scores = []
    for threshold in thresholds:
        threshold_predictions = []
        for phrase_tuple in tqdm(train_set):
            phrase_0 = phrase_tuple[0]
            phrase_1 = phrase_tuple[1]
            jaccard_sim = get_jaccard_sim(phrase_0, phrase_1)
            if jaccard_sim >= threshold:
                threshold_predictions.append(1)

            else:
                threshold_predictions.append(0)
        for idx, label in enumerate(threshold_predictions):
            if label == 1:
                if train_labels[idx] == 0:
                    # print('True label: ', test_labels[idx])
                    # phrase_0 = test_set[idx][0]
                    # phrase_1 = test_set[idx][1]
                    # print('Phrase: ', phrase_0, '-----', phrase_1)
                    # print(test_set[idx])
                    counter1 += 1
                else:
                    counter2 += 1
        auc = roc_auc_score(train_labels, threshold_predictions)
        print('For theshold: {}, the AUC is: {}'.format(threshold, auc))
        auc_scores.append((threshold, auc))
    sorted_aucs = sorted(auc_scores, key=lambda tup: tup[1], reverse=True)
    print('The best threshold learned in the training set is: {} with AUC: {}'.format(sorted_aucs[0][0], sorted_aucs[0][1]))
    best_threshold = sorted_aucs[0][0]
    # for testing
    test_pos_phrases = get_phrase_list(test_pos, phrase_dic)
    test_neg_phrases = get_phrase_list(test_neg, phrase_dic)
    # pprint(test_neg_phrases)
    # exit()
    test_set = test_pos_phrases + test_neg_phrases
    print('The length of the test set is: ', len(test_set))
    # labels: 1-> link exists, 0-> false edge
    test_labels = np.zeros(len(test_set), dtype=int)
    test_labels[:len(test_pos_phrases)] = 1
    # evaluate
    test_predictions = []
    for phrase_tuple in tqdm(test_set):
        phrase_0 = phrase_tuple[0]
        phrase_1 = phrase_tuple[1]
        jaccard_sim = get_jaccard_sim(phrase_0, phrase_1)
        if jaccard_sim >= best_threshold:
            test_predictions.append(1)
        else:
            test_predictions.append(0)
    for idx, label in enumerate(test_predictions):
        if label == 1:
            if test_labels[idx] == 0:
                print('True label: ', test_labels[idx])
                phrase_0 = test_set[idx][0]
                phrase_1 = test_set[idx][1]
                print('Phrase: ', phrase_0, '-----', phrase_1)
                print(test_set[idx])
                counter1 += 1
            else:
                counter2 += 1
    auc = roc_auc_score(test_labels, test_predictions)
    print('For theshold: {}, the AUC is: {}'.format(best_threshold, auc))
    print('Number: {}/{} negative phrases are above threshold: {}'.format(counter1,len(train_neg)+len(test_neg) ,best_threshold))
    print('Number: {}/{} positive phrases are above threshold: {}'.format(counter2,len(train_pos)+len(test_pos) ,best_threshold))



if __name__ == "__main__":
    main()
