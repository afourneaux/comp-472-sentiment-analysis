import spacy
import os
from afinn import Afinn
import csv

nlp = spacy.load("en_core_web_sm")

textFilePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textfile.txt")
file = open(textFilePath, "r")
text = file.read()
file.close()

doc = nlp(text)
afinn = Afinn()

# for sentence in doc.sents:
#     print(sentence)
#     print(afinn.score(sentence))

with open('output', "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["test", "row"])