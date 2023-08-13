# Databricks notebook source
# DBTITLE 1,Assignment 1 Part 1
# Ryan Bell (rtb210000@utdallas.edu)
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/6597278576552559/3392345356514563/5495876241316747/latest.html

# COMMAND ----------

pip install nltk


# COMMAND ----------

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.tree import Tree
from functools import reduce

# COMMAND ----------

# Import book
book = sc.textFile("/FileStore/tables/war_and_peace-2.txt")
book.collect()

# COMMAND ----------

# Split up book by spaces, remove empty strings, and create map reduce task to re-assemble book.
words = book.flatMap(lambda x: x.split(" ")).filter(lambda x: x != '').map(lambda x: (1, x))
words.collect()

# COMMAND ----------

# Re-assemble book for sentince chunker
book2 = words.reduceByKey(lambda x, y: x + " " + y)
book2.collect()

# COMMAND ----------

# Chunk book by sentince
def sent_tokins(x):
    cp = nltk.sent_tokenize(str(x))
    return cp

sents = book2.flatMap(sent_tokins)
sents.collect()

# COMMAND ----------

# Chunk book by words in sentences
def word_tokins(x):
    cp = nltk.word_tokenize(str(x))
    return cp

words = sents.map(word_tokins)
words.collect()

# COMMAND ----------

# Create tags for words in each sentence (input whole sentences)
def pos_tag(x):
    splitted = nltk.pos_tag(x)
    return splitted

tags = words.map(pos_tag)
tags.collect()

# COMMAND ----------

# Create trees of word structure to get named entities (input whole sentences)
def parser(x):
    cp = nltk.ne_chunk(x)
    return cp

trees = tags.map(parser)
trees.collect()

# COMMAND ----------

# Extract named entities from trees
def extract_ne(x): 
    whole_chunk = []
    current_chunk = []
    for i in x:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            if current_chunk:
                named_entity = " ".join(current_chunk)
                whole_chunk.append(named_entity)
                current_chunk = []
    return whole_chunk
    
named_e = trees.flatMap(extract_ne)
named_e.collect()

# COMMAND ----------

# Create map reduce task to count names and sort 
output = named_e.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y).sortBy(lambda x: -x[1])
output.collect()

# COMMAND ----------


