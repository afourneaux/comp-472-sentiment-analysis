import spacy
import os
from afinn import Afinn
import csv

nlp = spacy.load("en_core_web_sm")

textFilePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textfile.txt")
file = open(textFilePath, "r")
text = file.read()
file.close()

# doc = nlp(text)
doc = nlp('Donald Trump sleeps.')
afinn = Afinn()

header1 = ['Text', 'Named Entity?', 'NE Type', 'Governor', 'Sentiment - Token', 'Sentiment - Sentence']
header2 = ['Named Entity', 'NE Type', 'Governor', 'Sentiment - Token', 'Sentiment - Sentence']

mergedEntity = ''
isInEntity = False
rowNumber = 0
entityStartRow = 0
entityRowCount = 0
table1 = []
table2 = []

for token in doc:
    if (token.ent_iob == 1):
        mergedEntity += token.text_with_ws
        entityRowCount += 1
    if ((token.ent_iob == 2 or token.ent_iob == 3) and isInEntity):
        isInEntity == False
        for index in range(entityStartRow, entityStartRow + entityRowCount):
            table1[index][1] = mergedEntity
    if (token.ent_iob == 3):
        entityStartRow = rowNumber
        entityRowCount = 1
        isInEntity = True
        mergedEntity = token.text_with_ws
    row = [token.text, '', token.ent_type_, token.head.text, token.sentiment, token.sent.sentiment]
    table1.append(row)
    rowNumber += 1


for ent in doc.ents:
    row = [ent.text, ent.label_, ent.root.head.text, ent.sentiment, ent.sent.sentiment]
    table2.append(row)

with open('output.csv', "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header1)
    writer.writerows(table1)

    writer.writerow([])

    writer.writerow(header2)
    writer.writerows(table2)