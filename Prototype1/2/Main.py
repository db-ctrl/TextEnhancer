import spacy
from textstat.textstat import textstatistics, easy_word_set, legacy_round

from Prototype1Test import SpacyFuncs
# Get raw text as string.
with open("TestSentences2.txt") as f:
    text = f.read()

print(SpacyFuncs.avg_sentence_length(text))
