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

true_k = 300
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Cluster to word mapping: ")

# TODO: Make catch for blank spaces at end of sentences

sentence1 = "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much."

wordList = sentence1.split(" ")
out_str = ''
# TODO : Make nested for multiple sentences

# TODO: investigate where 'Terms' are used in relation to clusters (see what attributes are reduced , e.g. and)

for i in wordList:
    Y = vectorizer.transform([i])
    if i not in terms:
        out_str += str(i)
        out_str += "[N/A]"
        out_str += " "
    else:
        prediction = model.predict(Y)
        out_str += str(i)
        out_str += str(prediction)
        out_str += " "

print(out_str)





