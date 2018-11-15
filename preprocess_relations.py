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
    pprint(instances[:10])
    return instances


def build_phrase_dic(instances):
    odir = 'C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/relation_utilities/isa'
    if not os.path.exists(odir):
        os.makedirs(odir)
    phrase_vocab = sorted(list(set([t[0] for t in instances] + [t[1] for t in instances])))
    phrase_dic = {}
    for phrase in phrase_vocab:
        phrase = tokenize(phrase)
        phrase_dic[phrase] = len(phrase_dic)
    reversed_dictionary = dict(zip(phrase_dic.values(), phrase_dic.keys()))
    with open('{}.p'.format(os.path.join(odir, 'isa_phrase_vocab')), 'wb') as dump_file:
        pickle.dump(phrase_vocab, dump_file)
    with open('{}.p'.format(os.path.join(odir, 'isa_phrase_dic')), 'wb') as dump_file:
        pickle.dump(phrase_dic, dump_file)
    with open('{}.p'.format(os.path.join(odir, 'isa_reversed_dic')), 'wb') as dump_file:
        pickle.dump(reversed_dictionary, dump_file)

    return phrase_vocab, phrase_dic, reversed_dictionary


def build_relation_edgelist(file_name, instances, phrase_dic):
    odir = 'C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/relation_instances_edgelists'
    if not os.path.exists(odir):
        os.makedirs(odir)
    with open(os.path.join(odir, '{}.edgelist'.format(file_name)), 'w') as data_file:
        for instance in instances:
            # parent_index = str(phrase_dic[instance[1]])
            # child_index = str(phrase_dic[instance[0]])
            fromidx = str(phrase_dic[tokenize(instance[0])])
            toidx = str(phrase_dic[tokenize(instance[1])])
            data_file.write(fromidx + ' ' + toidx + '\n')


if __name__ == '__main__':
    odir = 'C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/relation_instances'
    instances = load_instances('isa.p', odir)
    phrase_vocab, phrase_dic, reversed_dictionary = build_phrase_dic(instances)
    print(len(instances))
    print(len(phrase_vocab))
    print(len(reversed_dictionary))
    build_relation_edgelist('isa2_relations', instances, phrase_dic)
