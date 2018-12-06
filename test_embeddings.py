import os
import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
import codecs

my_seed = 1997
import random

random.seed(my_seed)


def tsne_plot(dictionary):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    vocab = list(dictionary.keys())
    #random.shuffle(vocab)
    counter = 0
    for word in vocab[:2000]:
            tokens.append(dictionary[word])
            labels.append(word)

    #tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tokens)
    # new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    z = []
    for value in pca_result:
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
    odir = 'C:/Users/sotir/Desktop/isa_embeddings/'
    with codecs.open("{}".format(os.path.join(odir, file)), 'r', 'utf-8') as embeddings:
        embeddings.readline()
        for i, line in enumerate(embeddings):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            assert len(embedding) == 30
            node_embeddings[word] = np.array(embedding)
    return node_embeddings


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


isa_embeddings_dic = load_embeddings('isa_average_sum_words_link_predict.emb')
#tsne_plot(isa_embeddings_dic)
# target_embed = isa_embeddings_dic['prednival']
# all_similarities = []
# the_keys = list(isa_embeddings_dic.keys())
# for phrase in tqdm(the_keys):
#     if phrase == 'prednival':
#         continue
#     phrase_emb = isa_embeddings_dic[phrase]
#
#     cosine_sim = cos_sim(target_embed, phrase_emb)
#     all_similarities.append((phrase, cosine_sim))
#
# sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)
#
# pprint(sorted_sims[:10])
# print()
# pprint(sorted_sims[-10:])
