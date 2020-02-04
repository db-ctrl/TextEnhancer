from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from sklearn.cluster import KMeans


from future.moves import tkinter
from sklearn.decomposition import PCA

from sklearn.metrics import adjusted_rand_score
from Prototype1Test import SpacyFuncs


# Get raw text as string.
#with open("t.txt") as f:
 #   text = f.read()

from Prototype1Test import SpacyFuncs
# Get raw text as string.
with open("blank.txt") as f:
    rawtext = f.read()

# Convert sentence string into iterable list
sentList = rawtext.split(",")

documents = sentList

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 99
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

sentence1 = "Mr dursley is a nasty man"
sentence2 = "harry potter stole my wife."

Y = vectorizer.transform([sentence1])
prediction = model.predict(Y)
print("The Sentence:", sentence1, prediction)

Y = vectorizer.transform([sentence2])
prediction = model.predict(Y)
print("The Sentence:", sentence2, prediction)
