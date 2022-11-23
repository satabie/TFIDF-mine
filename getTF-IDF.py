"""
author: takeru sakata
data: 2022/11/22
"""
import pickle
import collections
import numpy as np


def main():
    f = open("corpus.bin", "rb")
    corpus = pickle.load(f)
    f.close()

    all_featurenames = getFeaturenames(corpus)
    idf_list = getIDFlist(corpus, all_featurenames)
    tf_list = getTFlist(corpus, all_featurenames)
    tfidf_list = getTFIDF(tf_list, idf_list)
    f = open('tf_model.bin', "wb")
    pickle.dump(tf_list, f)
    f.close()
    f = open('tfidf_model.bin', "wb")
    pickle.dump(tfidf_list, f)
    f.close()


def getFeaturenames(corpus):
    all_featurenames = []
    for document in corpus:
        word_set = set([word for word in document])
        featurenames = [*word_set]
        all_featurenames.append(featurenames)
    return all_featurenames


def getIDF(word, docs):
    N = len(docs)
    df = np.sum(np.array([int(word in doc)
                for doc in docs], dtype="float32"))  # wordを含む文書の数
    N += 1.0
    df += 1.0
    return np.log(N / df)


def getTF(doc, featurenames):
    N = len(doc)  # 単語の総数
    tf = collections.Counter(doc)

    tf_dict = {}
    for word in featurenames:
        tf_dict[word] = tf[word] / N
    return tf_dict


def getTFlist(corpus, all_featurenames):
    tf_list = []
    for i in range(len(all_featurenames)):
        tf = getTF(corpus[i], all_featurenames[i])
        tf_list.append(tf)
    return tf_list


def getIDFlist(corpus, all_featurenames):
    word_idf_list = []
    for featurenames in all_featurenames:
        word_idf_dict = {}
        for word in featurenames:
            word_idf_dict[word] = getIDF(word, corpus)
        word_idf_list.append(word_idf_dict)

    return word_idf_list


def getTFIDF(tf_list, idf_list):
    tf_idf_list = []
    for tf, idf in zip(tf_list, idf_list):
        tf_idf_dict = {}
        for word in tf:
            tf_idf_dict[word] = tf[word] * idf[word]
        tf_idf_list.append(tf_idf_dict)

    return tf_idf_list


if __name__ == "__main__":
    main()
