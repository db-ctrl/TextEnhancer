import spacy
from textstat.textstat import textstatistics, easy_word_set, legacy_round
import en_core_web_sm

def text_2_list(corpus):
    from Tools import SpacyFuncs
    # Get raw text as string.
    with open(corpus) as f:
        text = f.read()
        f.close()

    sent_list = (SpacyFuncs.break_sentences(text)).split(",")
    documents = sent_list

    return documents



