

file = open("blank.txt", 'w')
word_list = open("/Users/david/PycharmProjects/LSTM-Text-Generator/MainModules/MainPackage/t.txt").readlines()
unique_words = set(word_list)
print(unique_words.__len__())
