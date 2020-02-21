from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_TEXT_PATH = '/Users/david/PycharmProjects/LSTM-Text-Generator/MainModules/MainPackage/Out_Docs.txt'

# Get raw text as string.
with open(INPUT_TEXT_PATH) as f:
    rawtext = f.read()

# Convert sentence string into iterable list

# TODO: Ensure below is pulling through full sentences

#Splits via comma
# sentList = rawtext.split(",")

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

sentence1 = "lot ti sores and tp suederily as the wall and the cat hn a sery off houn the bat had been hare a siiht"

wordList = sentence1.split(" ")
out_str = ''
# TODO : Make nested for multiple sentences

# TODO: investigate where 'Terms' are used in relation to clusters (see what attributes are reduced , e.g. and)
clustEnt = 0

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
        clustEnt + 1

print(out_str, clustEnt)





