import spacy
from textstat.textstat import textstatistics, easy_word_set, legacy_round
import en_core_web_sm

from Tools import SpacyFuncs
# Get raw text as string.
with open("HP1.txt") as f:
    text = f.read()
    f.close()

file = open('sentsHP1.txt', 'w')

str1 = ''.join(SpacyFuncs.break_sentences(text))

# Convert sentence string into iterable list

# TODO: Ensure below is pulling through full sentences

sentList = str1.split(",")
documents = sentList

file.write(documents)
file.close()

print(str1)
