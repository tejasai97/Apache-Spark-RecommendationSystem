
import findspark
findspark.init()

from pyspark.mllib.recommendation import *
import random
from operator import *
from collections import defaultdict


# In[430]:


# Initialize Spark Context
# YOUR CODE GOES HERE
from pyspark import SparkContext
sc = SparkContext.getOrCreate()


# ## Loading data


artistData = sc.textFile('./data_raw/artist_data_small.txt').map(lambda s:(int(s.split("\t")[0]),s.split("\t")[1]))
artistAlias = sc.textFile('./data_raw/artist_alias_small.txt')
userArtistData = sc.textFile('./data_raw/user_artist_data_small.txt')


# ## Data Exploration

userArtistData = userArtistData.map(lambda s:(int(s.split(" ")[0]),int(s.split(" ")[1]),int(s.split(" ")[2])))


aliasDict = {}
entities = artistAlias.map(lambda s:(int(s.split("\t")[0]),int(s.split("\t")[1])))
for item in entities.collect():
    aliasDict[item[0]] = item[1]

userArtistData = userArtistData.map(lambda x: (x[0], aliasDict[x[1]] if x[1] in aliasDict else x[1], x[2]))

user = userArtistData.map(lambda x:(x[0],x[2]))
counts = user.map(lambda x: (x[0],x[1])).reduceByKey(lambda x,y : x+y)
countz = user.map(lambda x: (x[0],1)).reduceByKey(lambda x,y:x+y)
final = counts.leftOuterJoin(countz)

final = final.map(lambda x: (x[0],x[1][0],int(x[1][0]/x[1][1])))

l = final.top(3,key=lambda x: x[1])
for i in l:
    print('User '+str(i[0])+' has a total play count of '+str(i[1])+' and a mean play count of '+str(i[2])+'.')


# ####  Splitting Data for Testing
trainData, validationData, testData = userArtistData.randomSplit((0.4,0.4,0.2), seed=13)
trainData.cache()
validationData.cache()
testData.cache()

print(trainData.take(3))
print(validationData.take(3))
print(testData.take(3))
print(trainData.count())
print(validationData.count())
print(testData.count())


# ## The Recommender Model

# ### Model Evaluation

def modelEval(model, dataset):
    Artists = sc.parallelize(set(userArtistData.map(lambda x:x[1]).collect()))
    Users = sc.parallelize(set(dataset.map(lambda x:x[0]).collect()))
    
    TestDict ={}
    for user in Users.collect():
        filtered = dataset.filter(lambda x:x[0] == user).collect()
        for item in filtered:
            if user in TestDict:
                TestDict[user].append(item[1])
            else:
                TestDict[user] = [item[1]]
                

    TrainDict = {}
    for user in Users.collect():
        filtered = trainData.filter(lambda x:x[0] == user).collect()
        for item in filtered:
            if user in TrainDict:
                TrainDict[user].append(item[1])
            else:
                TrainDict[user] = [item[1]]

    score =0.00
    for user in Users.collect():
        predictionData =  Artists.map(lambda x:(user,x))
        predictions = model.predictAll(predictionData)
        filtered = predictions.filter(lambda x :not x[1] in TrainDict[x[0]])
        topPredictions = filtered.top(len(TestDict[user]),key=lambda x:x[2])
        l=[]
        for pre in topPredictions:
            l.append(pre[1])
        score+=len(set(l).intersection(TestDict[user]))/len(TestDict[user])    

    print("The model score for rank "+str(model.rank)+" is ~"+str(score/len(TestDict)))


# ### Model Construction

rankList = [2,10,20]
for rank in rankList:
    model = ALS.trainImplicit(trainData, rank , seed=345)
    modelEval(model,validationData)


bestModel = ALS.trainImplicit(trainData, rank=10, seed=345)
modelEval(bestModel, testData)


# Find the top 5 artists for a particular user and list their names
top5 = bestModel.recommendProducts(1059637,5)
for item in range(0,5):
    print("Artist "+str(item)+": "+artistData.filter(lambda x:x[0] == top5[item][1]).collect()[0][1])

