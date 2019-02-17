lr = 0.001
batch_size = 128
neg_samples = 2
dimensions = 30
walk_length = 40
num_walks = 10
window_size = 5
p = 1
q = 1

# Server

# input_edgelist = '/home/sotkot/node2vec_word_embeds/relation_instances_edgelists/part_of_relations.edgelist'
# output_file = 'part_of_rnn_test_words_link_predict.emb'
# train_pos = '/home/sotkot/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_train_pos.p'
# train_neg = '/home/sotkot/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_train_neg.p'
# test_pos = '/home/sotkot/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_test_pos.p'
# test_neg = '/home/sotkot/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_test_neg.p'
# train_graph = '/home/sotkot/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_train_graph_undirected.edgelist'
# phrase_dic = '/home/sotkot/node2vec_word_embeds/relation_utilities/part_of/part_of_reversed_dic.p'
# checkpoint_dir = '/home/sotkot/checkpoints/'
# embeddings_dir = '/home/sotkot/embeddings/'
# checkpoint_name = 'part_of_rnn_test_checkpoint_epoch_{}.pth.tar'


# Laptop

input_edgelist = '/home/sotiris/PycharmProjects/node2vec_word_embeds/relation_instances_edgelists/part_of_relations.edgelist'
output_file = 'part_of_rnn_test_words_link_predict.emb'
train_pos = '/home/sotiris/PycharmProjects/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_train_pos.p'
train_neg = '/home/sotiris/PycharmProjects/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_train_neg.p'
test_pos = '/home/sotiris/PycharmProjects/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_test_pos.p'
test_neg = '/home/sotiris/PycharmProjects/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_test_neg.p'
train_graph = '/home/sotiris/PycharmProjects/node2vec_word_embeds/part_of-undirected-dataset-train-test-splits/part_of_train_graph_undirected.edgelist'
phrase_dic = '/home/sotiris/PycharmProjects/node2vec_word_embeds/relation_utilities/part_of/part_of_reversed_dic.p'
checkpoint_dir = '/home/sotiris/PycharmProjects/checkpoints/'
embeddings_dir = '/home/sotiris/PycharmProjects/embeddings/'
checkpoint_name = 'part_of_rnn_test_checkpoint_epoch_{}.pth.tar'
