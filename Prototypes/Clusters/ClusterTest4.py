from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

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
print("Cluster to word mapping: ")

# TODO: Make catch for blank spaces at end of sentences

sentence1 = "Mr. Dursley hummed as he picked out his most boring tie for work."

wordList = sentence1.split(" ")
out_str = ''
# TODO : Make nested for multiple sentences

# TODO: investigate csrMatrix issue
for i in wordList:

    Y = vectorizer.transform([i])
    prediction = model.predict(Y)
    out_str += str(i)
    out_str += str(prediction)
    out_str += " "

print(out_str)





