import spacy
from textstat.textstat import textstatistics, easy_word_set, legacy_round

from Prototype1Test import ClusterTest
# Get raw text as string.
with open("t.txt") as f:
    text = f.read()

print(ClusterTest.cluster_texts(["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."], 3))
