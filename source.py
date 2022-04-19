from sklearn.preprocessing import StandardScaler
import spacy
import os
from afinn import Afinn
import csv
import numpy
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from spacy import displacy

# Initialise dependency libraries
nlp = spacy.load("en_core_web_sm")
afinn = Afinn()


# Text Parsing

# Read and parse the text from neighbouring files
filename = "textfile.txt"
# filename = "submissiontext.txt"
textFilePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
file = open(textFilePath, "r")
doc = nlp(file.read())
# doc = nlp('Donald Trump sleeps.')
file.close()

# Headers for each table
header1 = ['Text', 'Named Entity', 'Is in NE?', 'NE Type', 'Governor', 'Sentiment - Token', 'Sentiment - Sentence']
header2 = ['Text', 'NE Type', 'Governor', 'Sentiment - Token', 'Sentiment - Sentence']
header3 = ['Sentences']

# Tracker variables for determining Named Entity
mergedEntity = ''
isInEntity = False
rowNumber = 0
entityStartRow = 0
entityRowCount = 0

# Tables stored in memory
table1 = []
floatTable1 = []
table2 = []
floatTable2 = []
table3 = []

# Get token data T1
for token in doc:
    # Determine which named entity the token belongs to
    # NOTE - Prior to the assignment clarification, I had assumed the field "NE?" to mean "What NE is this token in?"
    #        The latter, as demonstrated below, requires a great deal more effort. I have included both fields primarily 
    #        to provide additional information, and partly because I did not want to throw away my algorithm

    # IOB 1: This token belongs to a named entity. Add it to the under-construction entity
    # IOB 2: This token does not belong to a NE. If currently building an entity, complete it and add it to the relevant rows
    # IOB 3: This token is the first in an NE. Begin construction of an entity, and complete any under-construction one.
    if (token.ent_iob == 1):
        mergedEntity += token.text_with_ws      # Merge token text to build the NE
        entityRowCount += 1                     # Track to how many tokens this NE belongs
    if ((token.ent_iob == 2 or token.ent_iob == 3) and isInEntity):
        isInEntity == False                                                     # End the current entity construction
        for index in range(entityStartRow, entityStartRow + entityRowCount):    # 
            table1[index][1] = mergedEntity
    if (token.ent_iob == 3):
        entityStartRow = rowNumber
        entityRowCount = 1
        isInEntity = True
        mergedEntity = token.text_with_ws
    isInEntity = 0
    if (token.ent_iob == 1 or token.ent_iob == 3):
        isInEntity = 1
    
    # String to float algorithm: Sum of word vector
    textToFloat = 0
    for value in token.head.vector:
        textToFloat += value

    row = [token.text, '', isInEntity, token.ent_type_, token.head.text, afinn.score(token.text), afinn.score(token.sent.text)]
    floatRow = [isInEntity, token.ent_type, textToFloat, afinn.score(token.text), afinn.score(token.sent.text)]
    table1.append(row)
    floatTable1.append(floatRow)
    rowNumber += 1

# Get entity data T2
for ent in doc.ents:
    # String to float algorithm: Sum of word vector
    textToFloat = 0
    for value in ent.root.head.vector:
        textToFloat += value

    row = [ent.text, ent.label_, ent.root.head.text, afinn.score(ent.text), afinn.score(ent.sent.text)]
    floatRow = [ent.label, textToFloat, afinn.score(ent.text), afinn.score(ent.sent.text)]
    floatTable2.append(floatRow)
    table2.append(row)

# Get sentence data
for sent in doc.sents:
    table3.append(sent.text_with_ws)

with open('parsing.csv', "w", newline='') as outfile:
    writer = csv.writer(outfile)

    writer.writerow(header1)
    writer.writerows(table1)

    writer.writerow([])

    writer.writerow(header2)
    writer.writerows(table2)

    writer.writerow([])

    writer.writerow(header3)
    for row in table3:
        writer.writerow([row])


# K-Means Clustering

k = 2       # K value for k-means clustering

# Generate the k-means clustering engine using random starting points
kmeans1 = KMeans(init="random", n_clusters=k)
kmeans2 = KMeans(init="random", n_clusters=k)
# Scale data to a uniform range and run the k-means clustering algorithm
make_pipeline(StandardScaler(), kmeans1).fit(floatTable1)
make_pipeline(StandardScaler(), kmeans2).fit(floatTable2)

# Begin extracting token data into an array keyed by cluster index
groupedByCluster1 = [[]]
for cluster in range(0, k):
    entries = numpy.where(kmeans1.labels_ == cluster)[0]
    group = []
    for entry in entries:
        # Re-add the token text for clarity
        group.append([table1[entry][0]] + floatTable1[entry])
    groupedByCluster1.append([cluster, group])

# Begin extracting entity data into an array keyed by cluster index
groupedByCluster2 = [[]]
for cluster in range(0, k):
    entries = numpy.where(kmeans2.labels_ == cluster)[0]
    group = []
    for entry in entries:
        # Re-add the entity text for clarity
        group.append([table2[entry][0]] + floatTable2[entry])
    groupedByCluster2.append([cluster, group])


# Write the cluster contents to csv
with open(str(k) + 'means.csv', "w", newline='') as outfile:
    writer = csv.writer(outfile)

    # Print table 1
    writer.writerow(["Centroids"])
    writer.writerow(['Is in NE?', 'NE Type', 'Governor', 'Sentiment - Token', 'Sentiment - Sentence'])
    for centroid in kmeans1.cluster_centers_:
        writer.writerow(centroid)
    
    writer.writerow([])
    writer.writerow(['Text', 'Is in NE?', 'NE Type', 'Governor', 'Sentiment - Token', 'Sentiment - Sentence'])

    for cluster in groupedByCluster1:
        if (len(cluster) == 0):
            continue
        writer.writerow(["CLUSTER " + str(cluster[0])])
        writer.writerows(cluster[1])
    
    writer.writerow([])
    writer.writerow([])
    writer.writerow([])

    # Print table 2
    writer.writerow(["Centroids"])
    writer.writerow(['NE Type', 'Governor', 'Sentiment - Token', 'Sentiment - Sentence'])
    for centroid in kmeans2.cluster_centers_:
        writer.writerow(centroid)

    writer.writerow([])
        
    writer.writerow(['Text', 'NE Type', 'Governor', 'Sentiment - Token', 'Sentiment - Sentence'])

    for cluster in groupedByCluster2:
        if (len(cluster) == 0):
            continue
        writer.writerow(["CLUSTER " + str(cluster[0])])
        writer.writerows(cluster[1])
    

# Display dependency graph
displacy.serve(doc, style="dep", options= {"compact": True})