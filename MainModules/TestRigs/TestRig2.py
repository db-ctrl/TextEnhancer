from Tools import SpacyFuncsTest

with open("p.txt") as f:
    text = f.read()
print(SpacyFuncsTest.flesch_reading_ease(text))
