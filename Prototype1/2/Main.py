import spacy
from textstat.textstat import textstatistics, easy_word_set, legacy_round
import en_core_web_sm

from Prototype1Test import SpacyFuncs
# Get raw text as string.
with open("t.txt") as f:
    text = f.read()

print(SpacyFuncs.dale_chall_readability_score(text))


