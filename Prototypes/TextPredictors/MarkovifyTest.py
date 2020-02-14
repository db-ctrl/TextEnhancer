import markovify
# Get raw text as string.

with open("HP1FULLBOOK.txt") as f:
    text = f.read()

# Build the model.
text_model = markovify.Text(text)
model = markovify.Text(text) # get your model as normal

# Print five randomly-generated sentences
print("\n" + "Random Sentences:" + "\n")
for i in range(5):
    print(text_model.make_sentence())

# Print three randomly-generated sentences of no more than 280 characters
print("\n" + "Short Sentences:" + "\n")
for i in range(3):
    print(text_model.make_short_sentence(280))

#print(Predict.getSent(model, 100, 4))


# Print five randomly-generated sentences
#for i in range(5):
#    print(text_model.make_sentence())
#
# Print three randomly-generated sentences of no more than 280 characters
#for i in range(3):
#    print(text_model.make_short_sentence(280))