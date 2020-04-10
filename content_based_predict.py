import sys
from pyspark import SparkConf, SparkContext
import json
import itertools
import time
import re
import math


start=time.time()

conf=SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc=SparkContext(conf=conf)


test_file=sys.argv[1]
model_file=sys.argv[2]
output_file=sys.argv[3]

testrdd=sc.textFile(test_file)
rdd1=testrdd.map(json.loads) 
rdd2=rdd1.map(lambda x:(x['user_id'],x['business_id']))

lines=sc.textFile(model_file) 
rdd=lines.map(json.loads)

bus=rdd.filter(lambda x: 'business_id' in x)
bus1=bus.map(lambda x: (x['business_id'],x['business_profile']))

user=rdd.filter(lambda x: 'user_id' in x)
user1=user.map(lambda x:(x['user_id'], x['user_profile']))

#join test file with user profile 
new=rdd2.join(user1) #(user,(business,profile))

new1=new.map(lambda x:(x[1][0],(x[0], x[1][1]))) #(business,(user,profile))

new2=new1.join(bus1) #(business,((user,user profile),(bus profile))))

new3=new2.map(lambda x:((x[1][0][0], x[0]), (x[1][0][1], x[1][1]))) #(user_id,bus_id)(user_profile,business_profile)

def cosine_sim(x):
    s1=x[1][0]
    s2=x[1][1]
    set1=set(s1)
    set2=set(s2)
    intsec=len(set1.intersection(set2))
    lenvec=math.sqrt(len(set1))*math.sqrt(len(set2))
    cos_sim=intsec/lenvec
    return(x[0][0],x[0][1],cos_sim)

#calculating cosine similarity
result=new3.map(cosine_sim).filter(lambda x: x[2] >= 0.01)

result1=result.collect()

def writetofile(x):
    resu=[]
    for i in x:
        resu.append({"user_id": i[0],"business_id": i[1],"sim": i[2]})
    return resu

yy=writetofile(result1)

with open(output_file,'w') as f:
    for item in yy:
        f.write(json.dumps(item) + "\n")

end= time.time()
print("Duration: "+str(end-start)+ "s")
