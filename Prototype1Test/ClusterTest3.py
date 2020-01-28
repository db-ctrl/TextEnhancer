from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from Prototype1Test import SpacyFuncs
# Get raw text as string.
#with open("t.txt") as f:
 #   text = f.read()

from Prototype1Test import SpacyFuncs
# Get raw text as string.
with open("t3.txt") as f:
    rawtext = f.read()

# Convert sentence string into iterable list
#TODO : Investigate whether oneline text is causing issue with split
sentList = rawtext.split(",")

documents = sentList

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 2
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

sentence1 = "chrome browser to open."
sentence2 = "My cat is hungry for google chrome."

Y = vectorizer.transform([sentence1])
prediction = model.predict(Y)
print("The Sentence:", sentence1, prediction)

Y = vectorizer.transform([sentence2])
prediction = model.predict(Y)
print("The Sentence:", sentence2, prediction)