from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import scipy.stats

# TODO: Make Clustering a defined function with configurable parameters

INPUT_TEXT_PATH = '/Users/david/PycharmProjects/LSTM-Text-Generator/MainModules/MainPackage/sentsHP1.txt'


def cluster_texts(documents, sentence, word_count, true_k,):
    # initialise counters
    words_in_clus, hit_list = ([] for i in range(2))

    # convert sentence to lowercase & split into list of words
    sentence.lower()
    word_list = sentence.split(" ")

    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(documents)

    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(x)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()


    for i in range(true_k):
        print("Cluster %d:" % i),
        # Print x amount of words from each cluster
        for ind in order_centroids[i, :20]:
            hit_list.insert(ind, terms[ind])
            print(' %s' % terms[ind])

        # check if a specific word is in a cluster
        if terms[ind] in word_list:
            # TODO: Spot duplicate words and add to probability e.g. harry 3/10
            words_in_clus.append(str(terms[ind]) + " " + "[" + "%d" % i + "] ")

        if terms[ind] in out_str.split(" "):
            out_str.replace(terms[ind], "hi")

    print("\n" + "Test sentence: " + "\n" + sentence + "\n")

    print("\n" + "Cluster to word mapping: ")

    print(out_str)

    print("\n" + "Words in clusters: ")
    print(words_in_clus)

    sorted_hits = [item for items, c in Counter(hit_list).most_common() for item in [items] * c]

    total_hits = dict(Counter(sorted_hits))

    print("\n" + "Total Cluster Hits: ")
    print(total_hits.__len__())

    print("\n" + "Entropy: ")
    for x in word_list:
        print(entropy([words_in_clus / word_count], base=2))

    # print("\n" + "Sorted Hits: ")
    # print(total_hits)
