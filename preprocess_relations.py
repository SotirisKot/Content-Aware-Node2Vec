import os
import pickle
import re
import json
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
    # pprint(instances[:10])
    # for instance in instances:
    #     # parent_index = str(phrase_dic[instance[1]])
    #     # child_index = str(phrase_dic[instance[0]])
    #     fromidx = reversed_dic[instance[0]]
    #     toidx = reversed_dic[instance[1]]
    #     # print(fromidx, '-----', toidx)
    #     # print('\n')
    return instances


def build_phrase_dic(instances):
    # odir = 'relation_utilities/tributary_of'

    # if not os.path.exists(odir):
    #     os.makedirs(odir)
    phrase_vocab = sorted(list(set([t[0] for t in instances] + [t[1] for t in instances])))
    phrase_dic = {}
    for phrase in phrase_vocab:
        phrase_dic[phrase] = len(phrase_dic)
    reversed_dictionary = dict(zip(phrase_dic.values(), phrase_dic.keys()))

    # with open('{}.p'.format(os.path.join(odir, 'tributary_of_phrase_vocab')), 'wb') as dump_file:
    #     pickle.dump(phrase_vocab, dump_file)
    # with open('{}.p'.format(os.path.join(odir, 'tributary_of_phrase_dic')), 'wb') as dump_file:
    #     pickle.dump(phrase_dic, dump_file)
    # with open('{}.p'.format(os.path.join(odir, 'tributary_of_reversed_dic')), 'wb') as dump_file:
    #     pickle.dump(reversed_dictionary, dump_file)

    return phrase_vocab, phrase_dic, reversed_dictionary


def build_relation_edgelist(file_name, instances, phrase_dic):
    odir = 'datasets/relation_instances_edgelists'
    if not os.path.exists(odir):
        os.makedirs(odir)
    with open(os.path.join(odir, '{}.edgelist'.format(file_name)), 'w') as data_file:
        for instance in instances:
            fromidx = str(phrase_dic[instance[0].lower()])
            toidx = str(phrase_dic[instance[1].lower()])
            # data_file.write(fromidx + ' ' + toidx + '\n')
            data_file.write(toidx + ' ' + fromidx + '\n')


if __name__ == '__main__':
    odir = '/home/sotiris/Desktop/'
    instances = load_instances('isa.p', odir)
    # with open('part_of.json', 'w') as fp:
    #     json.dump(instances, fp)
    #pprint(instances)
    # phrases = sorted(list(set([t[0] for t in instances] + [t[1] for t in instances])))
    # with open(os.path.join(odir, '{}.edgelist'.format('dummy_test_1')), 'w') as data_file:
    #     for phr in phrases:
    #         data_file.write(phr + '\n')
    # print(len(phrases))
    # exit()
    #phrase_vocab, phrase_dic, reversed_dictionary = build_phrase_dic(instances)
    phrase_dic = pickle.load(open('data_utilities/isa/isa_phrase_dic.p', 'rb'))
    # pprint(phrase_dic)
    # exit()
    # print(phrase_dic['Pleural cupula'])
    # print(phrase_dic['DNA molecule'])
    # exit()
    # print(reversed_dictionary[4451])
    # with open(os.path.join(odir, '{}.edgelist'.format('dummy_test_3')), 'w') as data_file:
    #     for instance in instances:
    #         # fromidx = str(reversed_dictionary[instance[0]])
    #         # toidx = str(reversed_dictionary[instance[1]])
    #         data_file.write(instance[0] +"----------"+ instance[1] + '\n')
    # print(len(instances))
    # print(len(phrase_vocab))
    # print(len(reversed_dictionary))
    build_relation_edgelist('isa_directed', instances, phrase_dic)
