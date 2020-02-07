from Tools import SpacyFuncs

with open("p.txt") as f:
    text = f.read()
print(SpacyFuncs.word_count(text))
