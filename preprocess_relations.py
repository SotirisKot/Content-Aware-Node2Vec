import os
import pickle
import re
import json
from pprint import pprint
import config

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', ' ').replace('\\', '').replace("'", '').strip().lower()).split()


def clean_dictionary(phrase_dic):
    for nodeid, phrase in phrase_dic.items():
        phrase_dic[nodeid] = tokenize(phrase)
    return phrase_dic


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
    odir = 'data_utilities/isa'

    if not os.path.exists(odir):
        os.makedirs(odir)
    phrase_vocab = sorted(list(set([t[0] for t in instances] + [t[1] for t in instances])))
    print("Total number of phrases: {}".format(len(phrase_vocab)))
    phrase_dic = {}
    for phrase in phrase_vocab:
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
    phrase_vocab, phrase_dic, reversed_dictionary = build_phrase_dic(instances)
    build_relation_edgelist('isa_directed', instances, phrase_dic)
    # with open('part_of.json', 'w') as fp:
    #     json.dump(instances, fp)
    #pprint(instances)
    # phrases = sorted(list(set([t[0] for t in instances] + [t[1] for t in instances])))
    # with open(os.path.join(odir, '{}.edgelist'.format('dummy_test_1')), 'w') as data_file:
    #     for phr in phrases:
    #         data_file.write(phr + '\n')
    # print(len(phrases))
    # exit()
    #
    #
    # with open(os.path.join(odir, '{}.txt').format('graph'), 'r') as data_file:
    #     edges = data_file.readlines()
    #
    # with open(os.path.join(odir, '{}.txt').format('graph_2'), 'a') as data_file:
    #     for line in edges:
    #         spl = line.split()
    #         data_file.write(spl[0] + " " + spl[1] + '\n')

    # test_pos = pickle.load(open(config.test_pos, 'rb'))
    # pprint(test_pos)
    # with open(os.path.join('/home/sotiris/Desktop/', '{}.txt').format('test_graph_part_of'), 'a') as data_file:
    #     for edge in test_pos:
    #         data_file.write(str(edge[0]) + " " + str(edge[1]) + '\n')
    # exit()
    # pprint(phrase_dic)
    # exit()
    # phrase_dic = pickle.load(open(config.phrase_dic, 'rb'))
    # for item in phrase_dic.items():
    #     if item[1] == 'anterior tongue carcinoma':
    #         print(item)
    #         exit(0)
    #
    # pprint(phrase_dic)
    # print(phrase_dic['root of uwda hierarchy'])
    # exit()
    # print(reversed_dictionary[4451])
    # with open(os.path.join(odir, '{}.edgelist'.format('isa_cleaned_dummy_test_3')), 'w') as data_file:
    #     for instance in instances:
    #         print(instance)
    #         exit()
    #         fromidx = str(phrase_dic[instance[0]])
    #         toidx = str(phrase_dic[instance[1]])
    #         data_file.write(toidx +"----------"+ fromidx + '\n')
    # print(len(instances))
    # exit()

