from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_TEXT_PATH = '/Users/david/PycharmProjects/LSTM-Text-Generator/MainModules/MainPackage/Out_Docs.txt'

sentence1 = "Harry and the Weasleys spent a happy afternoon having a furious snowball fight on the grounds.".lower()

# Get raw text as string.
with open(INPUT_TEXT_PATH) as f:
    rawtext = f.read()

# Convert sentence string into iterable list

# TODO: Ensure below is pulling through full sentences

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

wordList = sentence1.split(" ")
out_str = ''

for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind])
    if terms[ind] in wordList:
        out_str += str(terms[ind])
        out_str += str(" ")
        out_str += ("Cluster %d:" % i)

#TODO: Make for loop only print ONCE per cluster hit

print("\n")
print("Cluster to word mapping: ")
print(out_str)


#else:
#out_str += str(w)
#out_str += "[N/A]"
#out_str += " "






