# Databricks notebook source
# DBTITLE 1,Assignment 1 Part 2
# Ryan Bell (rtb210000@utdallas.edu)

# COMMAND ----------

pip install nltk

# COMMAND ----------

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# COMMAND ----------

# Load file containing summaries
summaries = sc.textFile("/FileStore/tables/plot_summaries.txt")
summaries.collect()

# COMMAND ----------

# Split summaries from their movie label on the tab
summaries_split = summaries.map(lambda x: x.split("\t"))

# Count the number of documents for tf-idf
num_docs = summaries_split.count()
summaries_split.collect()

# COMMAND ----------

nltk.download('stopwords')
from nltk.corpus import stopwords # Avoid class error

stop_words = set(stopwords.words('english'))

# Function to parse the movie summaries and filter stop words from them
def filter_stops(x):
    
    reassemb = []
    reassemb.append(x[0])
    keep_words = []
    for word in nltk.word_tokenize(str(x[1])):
        if word.isalpha() and word.lower() not in stop_words:
            keep_words.append(word.lower())
    reassemb.append(keep_words)
    return reassemb

summaries_processed = summaries_split.map(filter_stops)
summaries_processed.collect()

# COMMAND ----------

# Function to re-map the document number to each individual word along with the number of times that word appears in the document
def tf_mapping(x):
    output = []
    words = {}
    for word in x[1]:
        if word not in words:
            words[word] = 1
        else:
            words[word] = words[word] + 1
    for val in words:
        output.append((val, [x[0], words[val]]))
    return output

term_freq_map = summaries_processed.flatMap(tf_mapping)
term_freq_map.collect()

# COMMAND ----------

# Get each distinct term per document
terms_per_document = term_freq_map.map(lambda x: (x[0], 1))
terms_per_document.collect()

# COMMAND ----------

# Reduce to find the # of documents which contain each term. Then get idf per term.
import math
idf_per_term = terms_per_document.reduceByKey(lambda x,y: x + y).map(lambda x: (x[0], math.log(num_docs/x[1])))
idf_per_term.collect()

# COMMAND ----------

# Join the tf with the idf per term and multiply them to get a weight for each term
tf_idf_words = term_freq_map.join(idf_per_term).map(lambda x: (x[0], [x[1][0][0], x[1][0][1]*x[1][1]]))
tf_idf_words.cache()
tf_idf_words.collect()

# COMMAND ----------

# Group the tf-idf for each document then normalize for querying
import math

def normalize(x):
    normalize_factor = 0
    for word in x:
        normalize_factor = normalize_factor + (word[1] * word[1])
    normalize_factor = math.sqrt(normalize_factor)
    return_array = []
    for word in x:
        return_array.append([word[0], word[1]/normalize_factor])
    return return_array

tf_idf_docs = tf_idf_words.map(lambda x: (x[1][0], [x[0], x[1][1]])).groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0], normalize(x[1])))
tf_idf_docs.cache()
tf_idf_docs.collect()

# COMMAND ----------

# Load the movie metadata and create a lookup table
movie_meta = sc.textFile("/FileStore/tables/movie_metadata-1.tsv")
movie_lookup = movie_meta.map(lambda x: x.split("\t")).map(lambda x: (x[0], x[2]))
movie_lookup.cache()
movie_lookup.collect()

# COMMAND ----------

# Load a file containing queries
queries = sc.textFile("/FileStore/tables/mixed_queries-1.txt").collect()

# Create a unique ID for each query
i = 0
remapped_queries = []
for query in queries:
    new_query = []
    new_query.append(i)
    new_query.append(query)
    i = i + 1
    remapped_queries.append(new_query)

queries = sc.parallelize(remapped_queries).cache()

# COMMAND ----------

# Run queries through all the processing done on the normal documents to get the tf-idf value for each term
tf_idf_query = queries.map(filter_stops).flatMap(tf_mapping).join(idf_per_term).map(lambda x: (x[0], [x[1][0][0], x[1][0][1]*x[1][1]])).map(lambda x: (x[1][0], [x[0], x[1][1]])).groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0], normalize(x[1])))

# Separate the short queries from the long queries
short_tf_idf_query = tf_idf_query.filter(lambda x: len(x[1]) < 2).map(lambda x: (x[1][0][0], x[0]))
long_tf_idf_query = tf_idf_query.filter(lambda x: len(x[1]) > 1).collect()

# COMMAND ----------

# Function to compute cosine similarity of every document on every long query
def score_queries(x):
    query_scores = []
    for query in long_tf_idf_query:
        score = []
        value = 0
        for query_term in query[1]:
            for doc_term in x[1]:
                if query_term[0] == doc_term[0]:
                    value = value + (query_term[1] * doc_term[1])
                    break
        score.append(query[0])
        score.append(value)
        query_scores.append(score)
    return (x[0], query_scores)

# Short query scores are found by looking at the un-normalized tf-idf values per word across documents
collected_scores_short = tf_idf_words.join(short_tf_idf_query).map(lambda x: (x[1][1], [x[1][0][1], x[1][0][0]])).groupByKey().map(lambda x: (x[0], sorted(x[1], reverse=True)[:10])).collect()

# Long query scores are found using cosine similarity
collected_scores_long = tf_idf_docs.map(score_queries)
collected_scores_long.cache()

# COMMAND ----------

# Print the results for the short queries
print("---------------------------------------------")
found_queries = {}
for result in collected_scores_short:
    query = queries.filter(lambda x: x[0] == result[0]).first()
    found_queries[query[0]] = query[1]
    print("Query: " + query[1])
    for i in range(len(result[1])):
        print("     " + str(i + 1) + ": " + movie_lookup.filter(lambda x: x[0] == result[1][i][1]).map(lambda x: x[1]).first())
    print("---------------------------------------------")

# Print the results for the long queries
for i in range(len(long_tf_idf_query)):
    query = queries.filter(lambda x: x[0] == long_tf_idf_query[i][0]).first()
    found_queries[query[0]] = query[1]
    print("Query: " + query[1])
    movie_ids = collected_scores_long.sortBy(lambda x: -x[1][i][1]).map(lambda x: x[0]).take(10)
    k = 1
    for j in movie_ids:
        print("     " + str(k) + ": " + movie_lookup.filter(lambda x: x[0] == j).map(lambda x: x[1]).first())
        k = k + 1
    print("---------------------------------------------")

query_list = queries.collect()
j = 0
for i in range(len(query_list)):
    if not(i in found_queries):
        print("No matches could be found for query: " + str(query_list[i][1]))

# COMMAND ----------


