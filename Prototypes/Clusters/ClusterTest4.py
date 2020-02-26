from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

INPUT_TEXT_PATH = '/Users/david/PycharmProjects/LSTM-Text-Generator/MainModules/MainPackage/Out_Docs.txt'

sentence1 = "Harry and the Weasleys spent a happy afternoon having a furious snowball fight on the grounds.".lower()

# Get raw text as string.
with open(INPUT_TEXT_PATH) as f:
    rawtext = f.read()

# Convert sentence string into iterable list

# TODO: Ensure below is pulling through full sentences

sentList = rawtext.split(",")
documents = sentList

# Counter for words in clusters
wordsInClus = 0

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)



true_k = 25
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

wordList = sentence1.split(" ")
out_str = ''
hitList = []
# TODO Calculate how many times terms occour in each cluster, fill hitlist and then count instances
for i in range(true_k):
    print("Cluster %d:" % i),
# Print x amount of words from each cluster
    for ind in order_centroids[i, :10]:
        hitList.insert(ind, terms[ind])
        print(' %s' % terms[ind])

# check if a specific word is in a cluster
    if terms[ind] in wordList:
        out_str += str(terms[ind])
        out_str += str(" ")
        out_str += ("[" + "%d" % i + "] ")
        wordsInClus += 1

print("\n" + "Test sentence: " + "\n" + sentence1 + "\n")

print("Cluster to word mapping: ")

print(out_str)
print(wordsInClus)


sortedHits = [item for items, c in
             Counter(hitList).most_common() for item in
             [items] * c]

totalHits2 = dict(Counter(sortedHits))

print(totalHits2)




