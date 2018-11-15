import os
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

my_seed = 1997
import random

random.seed(my_seed)


def tsne_plot(dictionary):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    vocab = list(dictionary.keys())
    random.shuffle(vocab)
    counter = 0
    for word in vocab:
        if 'lung cancer' in word:
            tokens.append(dictionary[word])
            labels.append(word)
            counter += 1
        if counter == 75:
            break

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
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


f = open('C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/isa_link_predict.emb')
f.readline()
all_embeddings_dic = {}
# all_words=[]
isa_reverse_dic = pickle.load(open(
    'C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/relation_utilities/isa/isa_reversed_dic.p',
    'rb'))
for i, line in enumerate(f):
    line = line.strip().split(' ')
    word = int(line[0])
    embedding = [float(x) for x in line[1:]]
    assert len(embedding) == 128
    all_embeddings_dic[isa_reverse_dic[word]] = embedding
    # all_words.append(word)

odir = 'C:/Users/sotir/PycharmProjects/node2vec_average_embeddings/'
if not os.path.exists(odir):
    os.makedirs(odir)
with open("{}.p".format(os.path.join(odir, 'isa_embeddings_dic')), 'wb') as dump_file:
    pickle.dump(all_embeddings_dic, dump_file)
isa_embeddings_dic = pickle.load(open(os.path.join(odir, 'isa_embeddings_dic.p'), 'rb'))


# pprint(list(isa_embeddings_dic.keys()))
# tsne_plot(isa_embeddings_dic)

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


target_embed = isa_embeddings_dic['angelman syndrome']
all_similarities = []
the_keys = list(isa_embeddings_dic.keys())
for phrase in tqdm(the_keys):
    if phrase == 'angelman syndrome':
        continue
    phrase_emb = isa_embeddings_dic[phrase]

    cosine_sim = cos_sim(target_embed, phrase_emb)
    all_similarities.append((phrase, cosine_sim))

sorted_sims = sorted(all_similarities, key=lambda tup: tup[1], reverse=True)

print('Ten most similar: ', sorted_sims[:10])
print('Ten least similar: ', sorted_sims[-10:])
