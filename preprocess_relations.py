import os
import pickle
import re
from pprint import pprint

bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                   '').strip().lower()).split()).strip()


def tokenize(x):
    return bioclean(x)


def load_instances(file, odir):
    path_to_file = os.path.join(odir, file)
    instances = list(pickle.load(open(path_to_file, 'rb')))
    # reversed_dic = pickle.load(open('relation_utilities/part_of/part_of_reversed_dic.p', 'rb'))
    pprint(instances[:10])
    # for instance in instances:
    #     # parent_index = str(phrase_dic[instance[1]])
    #     # child_index = str(phrase_dic[instance[0]])
    #     fromidx = reversed_dic[instance[0]]
    #     toidx = reversed_dic[instance[1]]
    #     if fromidx == toidx:
    #         print(40*'fuck')
    #     # print(fromidx, '-----', toidx)
    #     # print('\n')
    return instances


def build_phrase_dic(instances):
    odir = 'relation_utilities/tributary_of'
    if not os.path.exists(odir):
        os.makedirs(odir)
    phrase_vocab = sorted(list(set([t[0] for t in instances] + [t[1] for t in instances])))
    phrase_dic = {}
    for phrase in phrase_vocab:
        phrase_dic[phrase] = len(phrase_dic)
    reversed_dictionary = dict(zip(phrase_dic.values(), phrase_dic.keys()))
    with open('{}.p'.format(os.path.join(odir, 'tributary_of_phrase_vocab')), 'wb') as dump_file:
        pickle.dump(phrase_vocab, dump_file)
    with open('{}.p'.format(os.path.join(odir, 'tributary_of_phrase_dic')), 'wb') as dump_file:
        pickle.dump(phrase_dic, dump_file)
    with open('{}.p'.format(os.path.join(odir, 'tributary_of_reversed_dic')), 'wb') as dump_file:
        pickle.dump(reversed_dictionary, dump_file)

    return phrase_vocab, phrase_dic, reversed_dictionary


def build_relation_edgelist(file_name, instances, phrase_dic):
    odir = 'relation_instances_edgelists'
    if not os.path.exists(odir):
        os.makedirs(odir)
    with open(os.path.join(odir, '{}.edgelist'.format(file_name)), 'w') as data_file:
        for instance in instances:
            fromidx = str(phrase_dic[instance[0]])
            toidx = str(phrase_dic[instance[1]])
            data_file.write(fromidx + ' ' + toidx + '\n')


if __name__ == '__main__':
    odir = 'C:/Users/sotir/Documents/UMLS-node2vec/umls-node2vec/relation_instances'
    instances = load_instances('tributary_of.p', odir)
    phrase_vocab, phrase_dic, reversed_dictionary = build_phrase_dic(instances)
    print(len(instances))
    print(len(phrase_vocab))
    print(len(reversed_dictionary))
    build_relation_edgelist('tributary_of_relations', instances, phrase_dic)
