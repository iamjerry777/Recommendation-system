import re
import sys
import os
from pyspark import SparkConf, SparkContext
import json
import itertools
import time
import math
import random
from itertools import chain




start=time.time()
conf=SparkConf().setAppName("Zhaofeng Tong hw3 task2").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc=SparkContext(conf=conf)

train_file=sys.argv[1]
model_file=sys.argv[2]
stopwords=sys.argv[3]

lines=sc.textFile(train_file)
rdd=lines.map(json.loads)
#read in data file and find document for every business
text=rdd.map(lambda x: (x['business_id'],x['text'])).groupByKey()
    

stopword=sc.textFile(stopwords)
rddword=stopword.collect()
#remove punc,numbers and stopwords from each document
def clean_str(x):
    string=[]
    for word in x[1]:
        word=word.lower()
        word1=re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~0-9]|\n', ' ', word)
        word1=word1.split(' ')
        for w in word1:
            if len(w) > 1 and w not in rddword:
                string.append(w)
    return string

t1=text.map(lambda x:(x[0],clean_str(x))) #cleaned document

#count total number of words and find rare words
count=t1.flatMap(lambda x:x[1]).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y)
totalwords=count.map(lambda x:x[1]).sum()
rarewords=totalwords*0.000001
rare=count.filter(lambda x:x[1]<rarewords)
rare1=rare.map(lambda x:x[0]).collect()
    


    

    
# def removerareword(x):
#     w=[]
#     for word in x[1]:
#         if word not in rare1:
#             w.append(word)
#     return w
  

 
# cleaned=t1.map(lambda x:(x[0],removerareword(x)))

 #function for term frequency  
def tf(x,rare1):
    dic={}
    fre=[]
    for word in x[1]:
        if word not in rare1:
            if word in dic:
                dic[word]+=1
            else:
                dic[word]=1
    
    h11=max(dic.values())
    for w,freq in dic.items():
        fre.append([w,[(freq/h11,x[0])]])
    return fre


def tfidfcal(x):
    tfidf=[]
    for freq,bus_id in x[1]:
        idf=math.log(10253/len(x[1]),2)
        score=freq*idf
        tfidf.append([bus_id,(score,x[0])])
    return tfidf

    
# cleaned1=cleaned.flatMap(tf)
# c2=cleaned1.reduceByKey(lambda x,y: x+y)
# c3=c2.flatMap(tfidfcal)
# c4=c3.groupByKey()


#calculate term frequency and tf idf score to construct business profile
b1=t1.flatMap(lambda x:tf(x,rare1)) 
rare1.clear()
b2=b1.reduceByKey(lambda x,y:x+y)
b3=b2.flatMap(tfidfcal)
b4=b3.groupByKey()
    
#get top 200 words with highest tfidf
def top200(x):
    l1=[]
    s=sorted(x[1])
    s1=s[:200]
    for a in s1:
        l1.append(a[1])
    return l1
b5=b4.map(lambda x:(x[0],top200(x)))

#business profile
bus=b5.collect()

#user profile by aggregating business profile	
user=rdd.map(lambda x:(x['business_id'],x['user_id']))
u2=user.join(b5)
u3=u2.map(lambda x:(x[1][0],x[1][1]))
u4=u3.reduceByKey(lambda x,y:x+y)
u5=u4.mapValues(lambda x:set(x))
user_profile=u5.collect()   
	            
#write output to file
with open(model_file, 'w') as f:
    for business_id, profile in bus:
        out={'business_id': business_id,'business_profile': profile}
        f.write(json.dumps(out) + "\n")
    bus.clear()
    
    for user_id, profile in user_profile:
        out={'user_id': user_id,'user_profile': list(profile)}
        f.write(json.dumps(out) + "\n")



# pt=[]
# for item in yy:
#     pt.append(json.dumps(item))
# pt1=[]
# for item1 in yy1:
#     pt1.append(json.dumps(item1))

# with open(output_file,'w') as f:
#     f.write(pt + "\n")
#     f.write(pt1 + "\n")
# with open(output_file,'w') as f:
#     for item in yy:
#         f.write(json.dumps(item) + "\n")
#     for it in yy1:
#         f.write(json.dumps(it) + "\n")

end= time.time()
print("Duration: "+str(end-start)+ "s")

