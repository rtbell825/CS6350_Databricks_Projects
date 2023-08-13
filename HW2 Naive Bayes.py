# Databricks notebook source
# DBTITLE 1,HW 2 Naive Bayes (with Laplace Smoothing)
# Ryan Bell (rtb210000@utdallas.edu)
# ham/spam texts Dataset from https://www.kaggle.com/datasets/team-ai/spam-text-message-classification

# COMMAND ----------

# Read the dataset into a dataframe
df = spark.read.option("header", "true").option("inferSchema", "true").csv("dbfs:/FileStore/tables/SPAM_text_message_20170820___Data.csv")
display(df)

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, StringIndexer, HashingTF
from pyspark.ml import Pipeline

# Preprocesses the dataset by indexing the ham/spam, tokenizing the messages and removing stop words

stages = []

tags = StringIndexer(inputCol="Category", outputCol="tags")
stages += [tags]

regexTokenizer = RegexTokenizer(inputCol="Message", outputCol="tokens", pattern="[\\W+|\\d+]")
stages += [regexTokenizer]

stopWords = StopWordsRemover(inputCol="tokens", outputCol="stopWords")
stages += [stopWords]

pipeline = Pipeline(stages=stages)
data = pipeline.fit(df).transform(df)

data = data.select(["tags", "stopWords"])

# Filter entries that don't have a ham/spam label or had no content in the messages after preprocessing
data = data.rdd.filter(lambda x: len(x[1]) > 0).filter(lambda x: x[0] < 2).toDF(["tags", "stopWords"])

display(data)

# COMMAND ----------

# Divide dataset into training and testing data
train, test = data.randomSplit([0.7, 0.3], seed = 1)
display(train)

# COMMAND ----------

# Get number of entires in train, number of ham, and number of spam
# Compute priors

X_count = train.count()
X_ham_count = train.select("tags").where(train.tags < 1).count()
X_spam_count = train.select("tags").where(train.tags > 0).count()
X_prob_ham = X_ham_count / X_count
X_prob_spam = X_spam_count / X_count
print(X_count, X_ham_count, X_spam_count, X_prob_ham, X_prob_spam)

# COMMAND ----------

# This function takes in a message and breaks it down into individual words
def deconstruct(x):
    retList = []
    for word in x[1]:
        retList.append((x[0], word))
    return retList

# Get a list of all the words with their ham/spam label
train_words = train.rdd.flatMap(deconstruct)

# Get a vocabulary of all the words in the training set and count them
vocabulary = train_words.map(lambda x: (x[1], 1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], 1))
vocabulary_count = vocabulary.count()
print(vocabulary_count)

# COMMAND ----------

# This function gives every word in the vocabulary a value of 1 or 1 + # of occurances for laplace smoothing
def laplaceSmoothing(x):
    if x[1][1] == None:
        return (x[0], 1)
    else:
        return (x[0], x[1][1] + 1)

# Get all the words marked as ham and count them up. Then join with the vocabulary and, do laplace smoothing and compute P(X|y) = (#X+1)/(#y+#vocab)
X_ham_words = train_words.filter(lambda x: x[0] < 1.0).map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x+y)
X_ham_words = vocabulary.leftOuterJoin(X_ham_words).map(laplaceSmoothing).map(lambda x: (x[0], x[1]/(X_ham_count + vocabulary_count)))

# Get all the words marked as spam and count them up. Then join with the vocabulary and, do laplace smoothing and compute P(X|y) = (#X+1)/(#y+#vocab)
X_spam_words = train_words.filter(lambda x: x[0] > 0.0).map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x+y)
X_spam_words = vocabulary.leftOuterJoin(X_spam_words).map(laplaceSmoothing).map(lambda x: (x[0], x[1]/(X_spam_count + vocabulary_count)))

X_spam_words.collect()

# COMMAND ----------

# Give a unique ID to each test entry
test_ids = []
i = 0
for entry in test.rdd.collect():
    test_ids.append((i, entry))
    i = i + 1

test_ids = sc.parallelize(test_ids)
test_ids.collect()

# COMMAND ----------

# Create a lookup table for each test entry with the ID and ham/spam label
test_lookup = test_ids.map(lambda x: (x[0], x[1][0]))
test_lookup.collect()

# COMMAND ----------

# This function takes in a test message and breaks it down into individual words
def deconstruct_test(x):
    retList = []
    for word in x[1][1]:
        retList.append((word, x[0]))
    return retList

# Deconstruct test messages and group each word together with the ID of the messages they appear in
test_words = test_ids.flatMap(deconstruct_test).groupByKey().map(lambda x: (x[0], list(x[1])))
test_words.collect()

# COMMAND ----------

# This function takes in a word, the list of message IDs that contain that word, and the probability of that word
# and pairs each ID with the probability
def pair_probs(x):
    retList = []
    for test in x[1][0]:
        retList.append((test, x[1][1]))
    return retList

# Join the test words with the ham/spam probabilities, then compute the posterier for each message
Y_ham_test = test_words.join(X_ham_words).flatMap(pair_probs).reduceByKey(lambda x,y: x*y).map(lambda x: (x[0], x[1]*X_prob_ham))
Y_spam_test = test_words.join(X_spam_words).flatMap(pair_probs).reduceByKey(lambda x,y: x*y).map(lambda x: (x[0], x[1]*X_prob_spam))
Y_ham_test.collect()

# COMMAND ----------

# Join the posteriers for ham and spam
Y_ham_spam = Y_ham_test.join(Y_spam_test)

# COMMAND ----------

# This function takes in a message ID with it's label and if a posterier couldn't be computed it just assigns that message the priors
def fillEmpties(x):
    if x[1][1] == None:
        return (x[0], (x[1][0], (X_prob_ham, X_prob_spam)))
    else:
        return x

# Join the test lookup table with the ham/spam label and assign the posterier to the priors if a posterier couldn't be computed
test_values = test_lookup.leftOuterJoin(Y_ham_spam).map(fillEmpties)
test_values.collect()

# COMMAND ----------

# This function determines if the ham or spam label is more likely and returns the prediction
def getPredictions(x):
    if x[1][1][0] > x[1][1][1]:
        return (x[0], x[1][0], 0.0)
    else:
        return (x[0], x[1][0], 1.0)

# This function compares the prediction to the actual label and returns if the prediciton was correct or not
def getAccuracy(x):
    if x[1] == x[2]:
        return (1.0)
    else:
        return (0.0)

# Get the predictions and accuracy, then add up the ones that were correct
accuracy = test_values.map(getPredictions).map(getAccuracy).reduce(lambda x,y: x+y)

# Divide the number of correct predicitons by the number of test messages to get the accuracy
print(accuracy / test_lookup.count())
