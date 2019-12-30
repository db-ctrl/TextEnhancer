import markovify
from Prototype1Test import Predict
# Get raw text as string.
with open("TestSentences.txt") as f:
    text = f.read()

# Build the model.
text_model = markovify.Text(text)
# generate 100 and choose the 5th. 1st line boring, hgh line it starts not making sense, so 5 seemed good.

model = markovify.Text(text) # get your model as normal
Predict.getSent(model, 100, 4)#[5]


# Print five randomly-generated sentences
#for i in range(5):
#    print(text_model.make_sentence())
#
# Print three randomly-generated sentences of no more than 280 characters
#for i in range(3):
#    print(text_model.make_short_sentence(280))