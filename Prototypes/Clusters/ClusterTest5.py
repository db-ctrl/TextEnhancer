from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import scipy.stats
# TODO: Make Clustering a defined function with configurable parameters

##########################################################################

INPUT_TEXT_PATH = '/Users/david/PycharmProjects/LSTM-Text-Generator/MainModules/MainPackage/sentsHP1.txt'
# Test sentence
sentence1 = "Harry said that dumbledore had let out a foul smell in the house".lower()

# Get raw text as string.
with open(INPUT_TEXT_PATH) as f:
    rawtext = f.read()

# Convert sentence string into iterable list

# TODO: Ensure below is pulling through full sentences

sentList = rawtext.split(",")
documents = sentList

###################################################################

def cluster_texts (documents, true_k):

    # Counter for words in clusters
    words_in_clus = 0
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    
    true_k = 250
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    
    word_list = sentence1.split(" ")
    out_str = ''
    hit_list = []
    for i in range(true_k):
        print("Cluster %d:" % i),
    # Print x amount of words from each cluster
        for ind in order_centroids[i, :20]:
            hit_list.insert(ind, terms[ind])
            print(' %s' % terms[ind])
    
    # check if a specific word is in a cluster
        if terms[ind] in word_list:
            # TODO: Spot duplicate words and add to probability e.g. harry 3/10
            out_str += str(terms[ind])
            out_str += str(" ")
            out_str += ("[" + "%d" % i + "] ")
            words_in_clus += 1
    
        if terms[ind] in out_str.split(" "):
            out_str.replace(terms[ind], "hi")
    
    
    print("\n" + "Test sentence: " + "\n" + sentence1 + "\n")
    
    print("\n" + "Cluster to word mapping: ")
    
    print(out_str)
    
    print("\n" + "Words in clusters: ")
    print(words_in_clus)
    
    sorted_hits = [item for items, c in Counter(hit_list).most_common() for item in [items] * c]
    
    total_hits = dict(Counter(sorted_hits))
    
    print("\n" + "Total Cluster Hits: ")
    print(total_hits.__len__())
    
    word_count = 26
    print("\n" + "Entropy: ")
    for x in word_list:
        print(entropy([1/word_count], base=2))
    
    # print("\n" + "Sorted Hits: ")
    #print(total_hits)







