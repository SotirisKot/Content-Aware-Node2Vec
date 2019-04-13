import json
import os
import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
import codecs
import logging
import random
import torch
import models
import config
# my_seed = 1997
# random.seed(my_seed)

handler = None
output_dir = '/home/sotiris/Documents/logger/'


def init_logger(handler):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    od = output_dir.split('/')[-1]
    logger = logging.getLogger(od)
    if handler is not None:
        logger.removeHandler(handler)
    handler = logging.FileHandler(os.path.join(output_dir, 'model__bigru.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger, handler


def tsne_plot(dictionary):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    vocab = list(dictionary.keys())
    random.shuffle(vocab)
    counter = 0
    for word in tqdm(vocab[:100]):
        # tokens.append(dictionary[word])
        # labels.append(word)
        #if 'carcinoma' in word or 'syndrome' in word:
        tokens.append(dictionary[word])
        labels.append(word)
        #     counter += 1
        # if counter == 100:
        #     break
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(tokens)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


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


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def get_dictionary(reversed_dic, all_embeddings_dic):
    final_dic = {}
    the_keys = list(all_embeddings_dic.keys())
    for phr_id in the_keys:
        phr_name = reversed_dic[int(phr_id)]
        final_dic[phr_name] = all_embeddings_dic[phr_id]

    return final_dic


# all_embeddings_dic = load_embeddings('/home/sotiris/Downloads/rnn_bigru/part_of_rnn_bigru_words_link_predict.emb')
# all_embeddings_dic_def = load_embeddings('/home/sotiris/Downloads/rnn_1_epoch/part_of_rnn_max_words_link_predict.emb')

# f = open('C:/Users/sotir/Desktop/part_of/part_of_average_2_words_link_predict.emb')
# f.readline()
#all_embeddings_dic = {}
#all_embeddings_dic = pickle.load(open('/home/sotiris/PycharmProjects/Content-Aware-N2V/rnn_node2vec/node_embeddings_phrases.p', 'rb'))
# all_embeddings_dic = pickle.load(open('/home/sotiris/Downloads/rnn_bigru/node_embeddings_phrases_concat.p', 'rb'))
#all_embeddings_dic = pickle.load(open('/home/sotiris/Downloads/rnn_1_epoch/node_embeddings_phrases.p', 'rb'))
# for i, line in enumerate(f):
#     line = line.strip().split(' ')
#     word = int(line[0])
#     embedding = [float(x) for x in line[1:]]
#     assert len(embedding) == 30
#     all_embeddings_dic[part_of_reverse_dic[word]] = embedding

# all_embeddings_dic = load_embeddings('part_of_baseline_3_link_predict_BEST.emb')
# all_embeddings_dic = get_dictionary(reversed_dic, all_embeddings_dic)
#  isa_embeddings_dic = load_embeddings('isa_average_sum_words_link_predict.emb')
# tsne_plot(all_embeddings_dic)
# target_embed = node_all_embeddings_dic['mouth']
# phrase_target = 'skin'
# target_embed = all_embeddings_dic[phrase_target]
# # the_keys_default = list(all_embeddings_dic_default.keys())
# the_keys = list(all_embeddings_dic.keys())
# the_keys = the_keys[2:]
# random.shuffle(the_keys)
# all_similarities = []
# for phrase in tqdm(the_keys):
#     # phrase = int(phrase)
#     # if phrase == 'septum of penis':
#     #     continue
#     # all_similarities = []
#     #all_similarities_def = []
#     # for phrase_tar in the_keys:
#     if phrase == phrase_target:
#         continue
#     # target_embed = all_embeddings_dic[phrase_target]
#     #target_embed_def = all_embeddings_dic_default[phrase]
#
#     phrase_emb = all_embeddings_dic[phrase]
#     #phrase_emb_def = all_embeddings_dic_default[phrase_tar]
#
#     cosine_sim = cos_sim(target_embed, phrase_emb)
#     #cosine_sim_def = cos_sim(target_embed_def, phrase_emb_def)
#     all_similarities.append((phrase, cosine_sim))
#     #all_similarities_def.append((phrase_tar, cosine_sim_def))
# #
# sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)
# sorted_sims = sorted(all_similarities_def, key=lambda tup: tup[1], reverse=True)
######

# pprint(phrase_target)
#
# phrase_words = phrase_target.split()
# phrase_words = set(phrase_words)
# ll = len(phrase_words)
# for ss in sorted_sims:
#     ss_parts = ss[0].split()
#     if len(phrase_words.intersection(ss_parts)) <= 1 and ss[1] >= 0.9:
#         pprint(ss)


def load_dictionaries(words=False):
    if words:
        all_embeddings_words = load_embeddings('/home/sotiris/Downloads/isa_gru_normal_lr/isa_gru_words_link_predict.emb')
    else:
        all_embeddings_words = pickle.load(open('/home/sotiris/Downloads/isa_gru_normal_lr/hard/node_embeddings_phrases.p', 'rb'))

    return all_embeddings_words


def plot(all_embeddings):
    tsne_plot(all_embeddings)


def stress_test(all_embeds):
    the_keys = list(all_embeds.keys())
    random.shuffle(the_keys)
    for phrase in tqdm(the_keys):
        all_similarities = []
        for phrase_tar in the_keys:
            if phrase == phrase_tar:
                continue
            target_embed = all_embeds[phrase]
            phrase_emb = all_embeds[phrase_tar]
            cosine_sim = cos_sim(target_embed, phrase_emb)
            all_similarities.append((phrase_tar, cosine_sim))

        sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)
        #####
        print(phrase)
        phrase_words = phrase.split()
        phrase_words = set(phrase_words)
        ll = len(phrase_words)
        for ss in sorted_sims:
            ss_parts = ss[0].split()
            if len(phrase_words.intersection(ss_parts)) <= 0 and ss[1] >= 0.9:
                pprint(ss)


def find_most_least_sim(target, all_embeddings, word=False):
    phrase_target = target
    target_embed = all_embeddings[phrase_target]
    the_keys = list(all_embeddings.keys())

    if word:
        the_keys = the_keys[2:]

    random.shuffle(the_keys)
    all_similarities = []
    for phrase in tqdm(the_keys):
        if phrase == phrase_target:
            continue
        phrase_emb = all_embeddings[phrase]
        cosine_sim = cos_sim(target_embed, phrase_emb)
        all_similarities.append((phrase, cosine_sim))
    sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)
    pprint(sorted_sims[:10])
    print()
    pprint(sorted_sims[-10:])


def create_json_words(all_embeddings_words):
    the_keys = list(all_embeddings_words.keys())
    the_keys = the_keys[2:]
    random.shuffle(the_keys)
    json_list = []
    for phrase_target in tqdm(the_keys[:100]):
        all_similarities = []
        target_embed = all_embeddings_words[phrase_target]
        for phrase in the_keys:
            if phrase == phrase_target:
                continue
            phrase_emb = all_embeddings_words[phrase]

            cosine_sim = cos_sim(target_embed, phrase_emb)
            all_similarities.append((phrase, cosine_sim))

        sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)

        url_target = 'https://en.wikipedia.org/wiki/{}'.format(phrase_target)
        most_similar = []
        least_similar = []
        for sim in sorted_sims[:10]:
            url = 'https://en.wikipedia.org/wiki/{}'.format(sim[0])
            most_similar.append((sim[0]+", "+str(sim[1]) + ', ' + url))

        for sim in sorted_sims[-10:]:

            url = 'https://en.wikipedia.org/wiki/{}'.format(sim[0])
            least_similar.append((sim[0]+", "+str(sim[1]) + ', ' + url))

        json_list.append(OrderedDict([('word', "target/" + phrase_target), ("url", url_target), ('most_similars', most_similar), ('least_similars', least_similar)]))
    with open("json_words.json", 'w') as fp:
        json.dump(json_list, fp)


def create_json_phrases(all_embeddings_phr, stress=None):
    the_keys = list(all_embeddings_phr.keys())
    random.shuffle(the_keys)
    json_list = []
    for phrase_target in tqdm(the_keys[:100]):
        all_similarities = []
        target_embed = all_embeddings_phr[phrase_target]
        for phrase in the_keys:
            if phrase == phrase_target:
                continue
            phrase_emb = all_embeddings_phr[phrase]

            cosine_sim = cos_sim(target_embed, phrase_emb)
            all_similarities.append((phrase, cosine_sim))

        sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)
        if stress is None:
            most_similar = []
            least_similar = []
            for sim in sorted_sims[:10]:
                most_similar.append((sim[0]+", "+str(sim[1])))

            for sim in sorted_sims[-10:]:
                least_similar.append((sim[0]+", "+str(sim[1])))

            json_list.append(OrderedDict([('phrase', "target/" + phrase_target), ('most_similars', most_similar), ('least_similars', least_similar)]))
        elif stress == 1:
            phrase_words = phrase_target.split()
            phrase_words = set(phrase_words)
            most_similar = []
            for ss in sorted_sims:
                ss_parts = ss[0].split()
                if len(phrase_words.intersection(ss_parts)) <= 1 and ss[1] >= 0.9:
                    most_similar.append(ss[0] + ", " + str(ss[1]))
            json_list.append(OrderedDict([('phrase', "target/" + phrase_target), ('most_similars', most_similar[:5])]))
        else:
            phrase_words = phrase_target.split()
            phrase_words = set(phrase_words)
            most_similar = []
            for ss in sorted_sims:
                ss_parts = ss[0].split()
                if len(phrase_words.intersection(ss_parts)) <= 0 and ss[1] >= 0.9:
                    most_similar.append(ss[0]+", "+str(ss[1]))
            json_list.append(OrderedDict([('phrase', "target/" + phrase_target), ('most_similars', most_similar[:5])]))
    with open("json_phrases_zero_common_word.json", 'w') as fp:
        json.dump(json_list, fp)


def get_index(w, vocab):
    try:
        return vocab[w]
    except KeyError:
        return vocab['UNKN']


def phr2idx(phr, word_vocab):
    p = [get_index(t, word_vocab) for t in phr]
    return p


def encode_one_unknown_sentence(all_embs, sent):
    checkpoint_file = '/home/sotiris/Downloads/rnn_residual_new/part_of_rnn_residual_2_checkpoint_epoch_1.pth.tar'
    modelcheckpoint = torch.load(checkpoint_file, map_location='cpu')
    vocabulary_size = len(modelcheckpoint['word2idx'])
    word2idx = modelcheckpoint['word2idx']
    model = models.GRUEncoder(vocabulary_size,
                              config.dimensions,
                              config.hidden_size,
                              config.neg_samples,
                              config.batch_size,
                              config.window_size)

    model.eval()
    model.load_state_dict(modelcheckpoint['state_dict'])
    with torch.no_grad():
        phr = [phr2idx(sent, word2idx)]
        phr_emb, _, _ = model(phr)

    the_keys = list(all_embs.keys())
    all_similarities = []
    for phrase in tqdm(the_keys):
        if phrase == sent:
            continue
        phrase_emb = all_embeddings[phrase]
        cosine_sim = cos_sim(phr_emb, phrase_emb)
        all_similarities.append((phrase, cosine_sim))
    sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)
    pprint(sorted_sims[:10])
    print()
    pprint(sorted_sims[-10:])


all_embeddings = load_dictionaries(words=False)
# phrase_dic = (pickle.load(open('/home/sotiris/PycharmProjects/Content-Aware-N2V/data_utilities/part_of/part_of_reversed_dic.p', 'rb')))
# for id, phr in phrase_dic.items():
#     if phr == 'type v collagen':
#         print(id, phr)
#         exit(0)

# encode_one_unknown_sentence(all_embeddings, 'white hair')
find_most_least_sim("tp53 gene", all_embeddings=all_embeddings, word=False)
# print(cos_sim(all_embeddings['oxyphil cell of parathyroid gland'], all_embeddings['plasma membrane']))
# stress_test(all_embeddings)
# create_json_words(all_embeddings)
# create_json_phrases(all_embeddings, stress=1)
# nums = [i for i in range(0, 1915)]
# num = np.random.choice(nums, size=20).tolist()
# print(num)
# plot(all_embeddings)
