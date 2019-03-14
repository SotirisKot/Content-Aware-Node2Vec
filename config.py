lr = 0.0001
batch_size = 128  # don't set batch_size = 1
neg_samples = 2
dimensions = 30
walk_length = 40
num_walks = 10
window_size = 5
p = 1
q = 1
n_layers = 1
bidirectional = True
residual = True
dropout = 0.0
epochs = 1
pool_type = 'max'
max_pad = False  # you can remove the zero-padding during max pooling by setting it -> -1e9
hidden_size = 30
gru_encoder = 2  # first encoder: last timesteps, second encoder: residual + max pooling, third encoder: residual + self attention
model = 'rnn'  # models: {'baseline' , 'average', 'rnn'}
train = False
evaluate = True

# Server

# input_edgelist = '/home/sotkot/node2vec_word_embeds/relation_instances_edgelists/part_of_relations.edgelist'
# output_file = 'part_of_rnn_test_pack_words_link_predict.emb'
# train_pos = '/home/sotkot/node2vec_word_embeds/part_of_easy_splits/part_of_train_pos.p'
# train_neg = '/home/sotkot/node2vec_word_embeds/part_of_easy_splits/part_of_train_neg.p'
# test_pos = '/home/sotkot/node2vec_word_embeds/part_of_easy_splits/part_of_test_pos.p'
# test_neg = '/home/sotkot/node2vec_word_embeds/part_of_easy_splits/part_of_test_neg.p'
# train_graph = '/home/sotkot/node2vec_word_embeds/part_of_easy_splits/part_of_train_graph_undirected.edgelist'
# phrase_dic = '/home/sotkot/node2vec_word_embeds/relation_utilities/part_of/part_of_reversed_dic.p'
# checkpoint_dir = '/home/sotkot/checkpoints/'
# embeddings_dir = '/home/sotkot/embeddings/'
# checkpoint_name = 'part_of_rnn_test_pack_checkpoint_epoch_{}.pth.tar'
# output_dir = '/home/sotkot/checkpoints/'

input_edgelist = '/home/sotkot/PycharmProjects/node2vec_word_embeds/datasets/relation_instances_edgelists/isa_directed.edgelist'
output_file = 'isa_gru_words_link_predict.emb'
train_pos = '/home/sotkot/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_train_pos.p'
train_neg = '/home/sotkot/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_train_neg.p'
test_pos = '/home/sotkot/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_test_pos.p'
test_neg = '/home/sotkot/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_test_neg.p'
train_graph = '/home/sotkot/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_train_graph_undirected.edgelist'
phrase_dic = '/home/sotkot/PycharmProjects/node2vec_word_embeds/data_utilities/isa/isa_reversed_dic.p'
# checkpoint_dir = '/home/sotiris/PycharmProjects/checkpoints/'
checkpoint_dir = '/home/sotkot/Downloads/isa_gru/'
embeddings_dir = '/home/sotkot/Downloads/'
checkpoint_name = 'isa_gru_checkpoint_epoch_1.pth.tar'
output_dir = '/home/sotkot/checkpoints/'

# Laptop

# input_edgelist = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/relation_instances_edgelists/part_of_directed.edgelist'
# output_file = 'part_of_rnn_residual_words_link_predict.emb'
# train_pos = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_hard_splits/part_of_train_pos.p'
# train_neg = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_hard_splits/part_of_train_neg.p'
# test_pos = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_hard_splits/part_of_test_pos.p'
# test_neg = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_hard_splits/part_of_test_neg.p'
# train_graph = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_hard_splits/part_of_train_graph_undirected.edgelist'
# phrase_dic = '/home/sotiris/PycharmProjects/node2vec_word_embeds/data_utilities/part_of/part_of_reversed_dic.p'
# # checkpoint_dir = '/home/sotiris/PycharmProjects/checkpoints/'
# checkpoint_dir = '/home/sotiris/Downloads/rnn_best_model/'
# embeddings_dir = '/home/sotiris/Downloads/'
# checkpoint_name = 'part_of_rnn_residual_checkpoint_epoch_1.pth.tar'
# output_dir = '/home/sotiris/checkpoints/'

# input_edgelist = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/relation_instances_edgelists/part_of_directed.edgelist'
# output_file = 'part_of_rnn_residual_words_link_predict.emb'
# train_pos = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_easy_splits/part_of_train_pos.p'
# train_neg = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_easy_splits/part_of_train_neg.p'
# test_pos = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_easy_splits/part_of_test_pos.p'
# test_neg = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_easy_splits/part_of_test_neg.p'
# train_graph = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/part_of_easy_splits/part_of_train_graph_undirected.edgelist'
# phrase_dic = '/home/sotiris/PycharmProjects/node2vec_word_embeds/data_utilities/part_of/part_of_reversed_dic.p'
# # checkpoint_dir = '/home/sotiris/PycharmProjects/checkpoints/'
# checkpoint_dir = '/home/sotiris/Downloads/rnn_best_model/'
# embeddings_dir = '/home/sotiris/Downloads/'
# checkpoint_name = 'part_of_rnn_residual_checkpoint_epoch_1.pth.tar'
# output_dir = '/home/sotiris/checkpoints/'

# input_edgelist = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/relation_instances_edgelists/isa_directed.edgelist'
# output_file = 'isa_rnn_residual_words_link_predict.emb'
# train_pos = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_train_pos.p'
# train_neg = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_train_neg.p'
# test_pos = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_test_pos.p'
# test_neg = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_test_neg.p'
# train_graph = '/home/sotiris/PycharmProjects/node2vec_word_embeds/datasets/isa_easy_splits/isa_train_graph_undirected.edgelist'
# phrase_dic = '/home/sotiris/PycharmProjects/node2vec_word_embeds/data_utilities/isa/isa_reversed_dic.p'
# # checkpoint_dir = '/home/sotiris/PycharmProjects/checkpoints/'
# checkpoint_dir = '/home/sotiris/Downloads/isa_gru/'
# embeddings_dir = '/home/sotiris/Downloads/'
# checkpoint_name = 'isa_gru_checkpoint_epoch_1.pth.tar'
# output_dir = '/home/sotiris/checkpoints/'

# Colab

# input_edgelist = 'drive/My Drive/pytorch-node2vec-umls-relations/relation_instances_edgelists/part_of_directed.edgelist'
# output_file = 'part_of_rnn_attention_words_link_predict.emb'
# train_pos = 'drive/My Drive/node2vec_average_embeddings/part_of_hard_splits/part_of_train_pos.p'
# train_neg = 'drive/My Drive/node2vec_average_embeddings/part_of_hard_splits/part_of_train_neg.p'
# test_pos = 'drive/My Drive/node2vec_average_embeddings/part_of_hard_splits/part_of_test_pos.p'
# test_neg = 'drive/My Drive/node2vec_average_embeddings/part_of_hard_splits/part_of_test_neg.p'
# train_graph = 'drive/My Drive/node2vec_average_embeddings/part_of_hard_splits/part_of_train_graph_undirected.edgelist'
# phrase_dic = 'drive/My Drive/node2vec_average_embeddings/relation_utilities/part_of/part_of_reversed_dic.p'
# checkpoint_dir = 'drive/My Drive/node2vec_average_embeddings/checkpoints/'
# embeddings_dir = 'drive/My Drive/node2vec_average_embeddings/embeddings/'
# checkpoint_name = 'part_of_rnn_attention_checkpoint_epoch_{}.pth.tar'
# output_dir = 'drive/My Drive/node2vec_average_embeddings/checkpoints/'


# Colab is-a

# input_edgelist = 'drive/My Drive/pytorch-node2vec-umls-relations/isa_relations_undirected/isa-undirected-dataset-train-test-splits/isa_relations.edgelist'
# output_file = 'isa_rnn_residual_words_link_predict.emb'
# train_pos = 'drive/My Drive/pytorch-node2vec-umls-relations/isa_relations_undirected/isa-undirected-dataset-train-test-splits/isa_train_pos.p'
# train_neg = 'drive/My Drive/pytorch-node2vec-umls-relations/isa_relations_undirected/isa-undirected-dataset-train-test-splits/isa_train_neg.p'
# test_pos = 'drive/My Drive/pytorch-node2vec-umls-relations/isa_relations_undirected/isa-undirected-dataset-train-test-splits/isa_test_pos.p'
# test_neg = 'drive/My Drive/pytorch-node2vec-umls-relations/isa_relations_undirected/isa-undirected-dataset-train-test-splits/isa_test_neg.p'
# train_graph = 'drive/My Drive/pytorch-node2vec-umls-relations/isa_relations_undirected/isa-undirected-dataset-train-test-splits/isa_train_graph_undirected.edgelist'
# phrase_dic = 'drive/My Drive/pytorch-node2vec-umls-relations/relation_utilities/isa/isa_reversed_dic.p'
# checkpoint_dir = 'drive/My Drive/node2vec_average_embeddings/checkpoints/'
# embeddings_dir = 'drive/My Drive/node2vec_average_embeddings/embeddings/'
# checkpoint_name = 'isa_rnn_residual_checkpoint_epoch_{}.pth.tar'
# output_dir = 'drive/My Drive/node2vec_average_embeddings/checkpoints/'