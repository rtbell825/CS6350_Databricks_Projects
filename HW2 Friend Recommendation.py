# Databricks notebook source
# DBTITLE 1,HW2 Friend Recommendation
# Ryan Bell (rtb210000@utdallas.edu)

# COMMAND ----------

# Read the dataset and split into (user, friendlist)
data = sc.textFile("/FileStore/tables/friend_list.txt")
data = data.map(lambda x: x.split("\t"))
data.collect()

# COMMAND ----------

# Create a dataframe for the user and friend list columns
df = data.toDF(["userID", "friendIDs"])
df.tail(10)

# COMMAND ----------

display(df)

# COMMAND ----------

# Get a sample of 10 random users from the dataset and make them into an RDD
sample = df.sample(0.1).take(10)
sample = sc.parallelize(sample, 10)
sample.collect()

# COMMAND ----------

# Function that takes in a user and their friends and outputs that user with each friend in a pair
def friendPairs(x):
    ret_list = []
    if "," in x[1]:
        friends = x[1].split(",")
    else:
        return [(x[0], x[1])]
    for friend in friends:
        ret_list.append((x[0], friend))
    return ret_list

# Get all the sample users and friends in a (user, friend) flatmap
lookup_pairs = sample.flatMap(friendPairs)
lookup_pairs.collect()

# COMMAND ----------

# Flip the user and friends, then group by the friend IDs
friend_pairs = lookup_pairs.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: list(x))
friend_pairs.collect()

# COMMAND ----------

# function takes in friends, then groups the lookup friend with all their mutual friends and returns a flat map
def pairMutuals(x):
    ret_list = []
    mutuals = []

    if x[1][1] == None:
        mutuals.append('')
    elif "," in x[1][1]:
        mutuals = x[1][1].split(",")
    else:
        mutuals.append(x[1][1])

    for lookup in x[1][0]:
        for mutual in mutuals:
            if lookup != mutual:
                ret_list.append((lookup, mutual))
    return ret_list

# Join the friend IDs with the original dataframe. Then map the original friends with their friends friends
mutual_pairs = friend_pairs.leftOuterJoin(df.rdd).flatMap(pairMutuals)
mutual_pairs.collect()

# COMMAND ----------

# This function removes all friend of friends if that friend is already a friend of the user
def filterFriends(x):
    retList = []
    if "," in x[1][1]:
        friends = x[1][1].split(",")
    else:
        friends = x[1][1]
    for mutual in x[1][0]:
        if mutual[0] not in friends:
            retList.append(mutual)
    return (x[0], retList)

# Count all the lookup user's mutual friends, then group and sort them, and last select the top 10 of them
mutuals = mutual_pairs.map(lambda x : (x, 1)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().map(lambda x: (x[0], sorted(list(x[1]), key=lambda x: x[1], reverse=True))).join(df.rdd).map(filterFriends).map(lambda x: (x[0], x[1][:10]))
mutuals = mutuals.collect()
print(mutuals)

# COMMAND ----------

# Print the final recommendations in a nice format
for lookup in mutuals:
    print(lookup[0] + ": ")
    for i in range(len(lookup[1])):
        print(lookup[1][i][0])
    print()

# COMMAND ----------


