import os
import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
import codecs

#my_seed = 1997
import random

#random.seed(my_seed)


def tsne_plot(dictionary):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    vocab = list(dictionary.keys())
    random.shuffle(vocab)
    counter = 0
    for word in vocab[:100]:
        # tokens.append(dictionary[word])
        # labels.append(word)
        # if 'toe' in word:
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
    odir = 'C:/Users/sotir/Desktop/part_of/rnn/'
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

def get_dictionary(reversed_dic, all_embeddings_dic):
    final_dic = {}
    the_keys = list(all_embeddings_dic.keys())
    for phr_id in the_keys:
        phr_name = reversed_dic[int(phr_id)]
        final_dic[phr_name] = all_embeddings_dic[phr_id]

    return final_dic


# f = open('C:/Users/sotir/Desktop/part_of/part_of_average_2_words_link_predict.emb')
# f.readline()
# all_embeddings_dic = {}
all_embeddings_dic = pickle.load(open('C:/Users/sotir/Desktop/part_of/rnn/node_embeddings_phrases.p', 'rb'))

# for i, line in enumerate(f):
#     line = line.strip().split(' ')
#     word = int(line[0])
#     embedding = [float(x) for x in line[1:]]
#     assert len(embedding) == 30
#     all_embeddings_dic[part_of_reverse_dic[word]] = embedding

# all_embeddings_dic = load_embeddings('part_of_baseline_3_link_predict_BEST.emb')
# all_embeddings_dic = get_dictionary(reversed_dic, all_embeddings_dic)
#  isa_embeddings_dic = load_embeddings('isa_average_sum_words_link_predict.emb')
tsne_plot(all_embeddings_dic)
# target_embed = node_all_embeddings_dic['mouth']
# target_embed = all_embeddings_dic['left ninth intercostal nerve']
# print(target_embed)
#
# all_similarities = []
# the_keys = list(all_embeddings_dic.keys())
# for phrase in tqdm(the_keys):
#     # phrase = int(phrase)
#     if phrase == 'left ninth intercostal nerve':
#         continue
#     phrase_emb = all_embeddings_dic[phrase]
#
#     cosine_sim = cos_sim(target_embed, phrase_emb)
#     all_similarities.append((phrase, cosine_sim))
#
# sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)
#
# pprint(sorted_sims[:10])
# print()
# pprint(sorted_sims[-10:])
