import spacy
from textstat.textstat import textstatistics, easy_word_set, legacy_round

from Prototype1Test import ClusterTest
# Get raw text as string.
with open("TestSentences2.txt") as f:
    text = f.read()

print(ClusterTest(text))
