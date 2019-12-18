import nltk

fp = open("TestSentences.txt")
text = fp.read()

sentence = nltk.sent_tokenize(text)
for sent in sentence:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))
