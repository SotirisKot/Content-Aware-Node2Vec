# Content-Aware Node2vec
Source code and datasets of BioNLP 2019 paper: "Embedding Biomedical Ontologies by Jointly Encoding Network Structure and Textual Node Descriptors"

## Datasets
The folder "datasets" contains the edgelists of the two datasets, denoted Part-of and Is-a, used
in Content-Aware Node2vec. For each dataset, exist some dictionaries in the folder data_utilities.
For example for the Is-a dataset:
* isa_phrase_dic.p (mapping between nodes and textual descriptors--the keys are the textual descriptors -- you must use the reversed_dic)
* isa_phrase_vocab.p (the textual descriptors associated with each node)
* isa_reversed_dic.p (the reversed dictionary of isa_phrase_dic.p)

## Run
First run the following script to generate the train/test graphs

    python3 create_dataset.py [--input path-to-edgelist] [--dataset [part_of,isa]]

Then you can run the experiments file to train
    
    python3 experiments.py
   
All of the parameters can be modified from the config file, but also passed as arguments too.

## Dependencies

* pytorch == 1.0.1
* networkx == 2.2
* scikit_learn == 0.20.2

## Cite
If you use the code, please cite this paper:

S. Kotitsas, D. Pappas, I. Androutsopoulos, R. McDonald and M. Apidianaki, 
"Embedding Biomedical Ontologies by Jointly Encoding Network Structure and Textual Node Descriptors". 
Proceedings of the 18th Workshop on Biomedical Natural Language Processing (BioNLP 2019) of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, 2019.