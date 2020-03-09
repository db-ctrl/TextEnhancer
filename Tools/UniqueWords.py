

with open ("/Users/david/PycharmProjects/LSTM-Text-Generator/MainModules/MainPackage/HP5.txt", "r") as f:
    # Get a list of lines in the file and covert it into a set
    words = f.readlines()
    count = len(words)

print(count)
