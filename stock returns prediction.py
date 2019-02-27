
# coding: utf-8

# In[1]:


import sys
import urllib
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import html2text
from newspaper import Article
import newspaper
import re
import os
import numpy as np
import pandas as pd
from textblob import TextBlob
from statistics import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def QDA(X_train,y_train,X_test,y_test):
    dc_class= QuadraticDiscriminantAnalysis()
    dc_class.fit(X_train,y_train)
    predict=dc_class.predict(X_test)
    print(" Discriminant accuracy",dc_class.score(X_test,y_test))
    return dc_class.score(X_test,y_test)

def svcClass(X_train,y_train,X_test,y_test):
    suv= SVC()
    suv.fit(X_train,y_train)
    predict=suv.predict(X_test)
    print(" suv accuracy",suv.score(X_test,y_test))
    return suv.score(X_test,y_test)

def knnClass(X_train,y_train,X_test,y_test):
    knn= neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
#     predict=knn.predict(X_test)
    print(" knn accuracy",knn.score(X_test,y_test))
    return knn.score(X_test,y_test)

def treeClass(X_train,y_train,X_test,y_test):
    detclass= DecisionTreeClassifier()
    detclass.fit(X_train,y_train)
    predict=detclass.predict(X_test)
    print(" Tree accuracy",detclass.score(X_test,y_test))
    return detclass.score(X_test,y_test)

def reglog(X_train,y_train,X_test,y_test):
    reglog= LogisticRegression()
    reglog.fit(X_train,y_train)
    predict=reglog.predict(X_test)
    print("Log-regression accuracy",reglog.score(X_test,y_test))
    return reglog.score(X_test,y_test)

def randomForest(X_train,y_train,X_test,y_test):
    ran= RandomForestClassifier(n_estimators=500,n_jobs=-1)
    ran.fit(X_train,y_train)
    predict=ran.predict(X_test)
    print("Random Forest accuracy",ran.score(X_test,y_test))
    return ran.score(X_test,y_test)

def tensplitfolds(X_train):
    print("Fold split function starts")
    total_length=len(X_train)
    fold=np.zeros(10,dtype=int)
    length_Cutter=(round(total_length*0.1))
    print(length_Cutter)
    i=0
    for i in range(10):
        if(i==0):
            fold[i]=length_Cutter
        if(i>0):
            fold[i]=length_Cutter+fold[i-1]
            if(fold[i]>train):
                fold[i]=fold[i]-(fold[i]-train)
        print(i,fold[i])
    print("Fold split function ends")
    return fold

def highestMeanAccuracy(QDA_accuracy,svcClass_accuracy,knnClass_accuracy,treeClass_accuracy,reglog_accuracy,randomForest_accuracy):
    highaccuracy=['technique',0]
    mean_store = np.array([QDA_accuracy,svcClass_accuracy,knnClass_accuracy,treeClass_accuracy,reglog_accuracy,randomForest_accuracy]) 
    mean_store.max()
    for i in range(6):
        if(mean_store[i]==mean_store.max()):
            if(i==0):
                highaccuracy=['QDA_accuracy',mean_store[i]]
            elif(i==1):
                highaccuracy=['svcClass_accuracy',mean_store[i]]
            elif(i==2):
                highaccuracy=['knnClass_accuracy',mean_store[i]]
            elif(i==3):
                highaccuracy=['treeClass_accuracy',mean_store[i]]
            elif(i==4):
                highaccuracy=['reglog_accuracy',mean_store[i]]
            elif(i==5):
                highaccuracy=['randomForest_accuracy',mean_store[i]]
    return highaccuracy  

def finalModel(X_train,y_train,X_test,y_test,recommendedList):
    accuracy=0
    if(recommendedList[0]=='reglog_accuracy'):
        accuracy=reglog(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='QDA_accuracy'):
        accuracy=QDA(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='svcClass_accuracy'):
        accuracy=svcClass(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='knnClass_accuracy'):
        accuracy=knnClass(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='treeClass_accuracy'):
        accuracy=treeClass(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='randomForest_accuracy'):
        accuracy=randomForest(X_train,y_train,X_test,y_test)
    return accuracy


namedf=pd.read_excel("/home/ec2-user/SageMaker/Jayavignesh_stock_prediction/closepricedatastocklisttworow.xlsx")
datedf=pd.read_excel("/home/ec2-user/SageMaker/Jayavignesh_stock_prediction/closepricedata15s.xlsx")





querylist=namedf["Name"].tolist()
stocklist=namedf["Symbol"].tolist()
datelist=datedf["Date"].dt.date.tolist()

qdatelist=[]
for i in range(len(datelist)):
    dform=str(datelist[i].month)+"/"+str(datelist[i].day)+"/"+str(datelist[i].year)
    qdatelist.append(dform)




sendf=pd.DataFrame(index=datelist,columns=stocklist)

for i in range(len(qdatelist)):
    for j in range(len(querylist)):
        baseurl="https://www.google.com/search?q="+querylist[j]+"&tbs=cdr:1,cd_min:"+qdatelist[i]+",cd_max:"+qdatelist[i]+"&tbm=nws"
        news_links=[]
        pagenum=2
        for k in range(0,pagenum):
                url = baseurl+"&start="+str(k*10)
                code=requests.get(url)
                soup=BeautifulSoup(code.text,'html.parser')
                for result_table in soup.findAll("div", {"class": "g"}):
                    a_click = result_table.find("a")
                    a_date = result_table.find("span")
                    link=str(a_click.get("href")).split("q=")[1]
                    link=link.split("%3Fpfrom")[0]
                    news_links.append(link.split("&sa=")[0])
        texts=[]
        for z in range(len(news_links)):
            try:
                print("This is article "+str(z) )
                article = Article(news_links[z],language='en')
                article.download()
                article.parse()
                texts.append(article.text)
            except:
                print("This is forbidden "+str(z) )
                pass
        score=[]
        for y in range(len(texts)):
            s=TextBlob(texts[y])
            score.append(s.sentiment.polarity)
        avgsentiment=sum(score)/len(score)
        sendf.loc[datelist[i],stocklist[j]]=avgsentiment

sendf

sendf.to_excel("/home/ec2-user/SageMaker/Jayavignesh_stock_prediction/sentiment.xlsx")

closedata = pd.read_excel("/home/ec2-user/SageMaker/Jayavignesh_stock_prediction/closepricedata15s.xlsx")
i=0
j=0
closedataColumn=closedata.columns.values.tolist()
columnListNames=[]
for i in range(len(closedataColumn)):
    for j in range(5):
        if(i!=0):
            name=closedataColumn[i]+" "+str((j+1))+" return"
            columnListNames.append(name)

closedf=pd.DataFrame(data=closedata,columns=['Date'])
c=0
for i in range (0,len(columnListNames),5):
    for j in range(5):
        closedf[columnListNames[i+j]]=((closedata.iloc[:,c+1]-closedata.iloc[:,c+1].shift(j+1))/closedata.iloc[:,c+1].shift(j+1))*100
    c=c+1

resultsdf=pd.DataFrame(index=stocklist)
resultsdf["Accuracy"]=0

for z in range(0,len(stocklist)):
        targetname=stocklist[z]+" 1 return"

        y=pd.Series(data=closedf[targetname]).copy()

        for i in range(1,len(y)):
            try:      
                if (y.iloc[i]>0):
                    y.iloc[i]=1
                else:
                    y.iloc[i]=0
            except:
                y.iloc[i]=0




        txy=closedf.drop(columns=[targetname])

        txy["sscore"]=sendf[stocklist[z]].tolist()

        txy["target"]=y

        txy.dropna(inplace=True)

        Y=txy["target"]

        X=txy.loc[:, txy.columns != 'Date']

        X=X.loc[:, X.columns != 'target']

        totalRows=len(txy)
        test=round(totalRows*0.2)
        train=totalRows-test

        xtrain=X.iloc[0:train]
        xtest=X.iloc[train:]
        ytrain=Y.iloc[0:train]
        ytest=Y.iloc[train:]



        fold=tensplitfolds(xtrain)
        highaccuracylist=[]

        # Cross validation
        svcClass_accuracy=np.zeros(9,dtype=float)
        knnClass_accuracy=np.zeros(9,dtype=float)
        treeClass_accuracy=np.zeros(9,dtype=float)
        reglog_accuracy=np.zeros(9,dtype=float)
        randomForest_accuracy=np.zeros(9,dtype=float)
        for j in range(9):
            X_train = xtrain.iloc[0:fold[j]]
            y_train = ytrain.iloc[0: fold[j]]      
            X_test = xtrain.iloc[fold[j]:fold[j+1]]
            y_test = ytrain.iloc[fold[j]:fold[j+1]]
            try:
                svcClass_accuracy[j]=svcClass(X_train,y_train,X_test,y_test)
                knnClass_accuracy[j]=knnClass(X_train,y_train,X_test,y_test)
                treeClass_accuracy[j]=treeClass(X_train,y_train,X_test,y_test)
                reglog_accuracy[j]=reglog(X_train,y_train,X_test,y_test)
                randomForest_accuracy[j]=randomForest(X_train,y_train,X_test,y_test)
                highAccuracy=highestMeanAccuracy(0,svcClass_accuracy.mean(),
                                knnClass_accuracy.mean(),
                                treeClass_accuracy.mean(),reglog_accuracy.mean(),
                                randomForest_accuracy.mean())
            except:
                continue



        finalModelAccuracy=finalModel(xtrain,ytrain,xtest,ytest,highAccuracy)

        print ("\n\n***************************\n\n")
        print ("Stock:",stocklist[z])
        print ("Final Accuracy of the model=",finalModelAccuracy)
        print ("\n\n***************************\n\n\n\n")








        resultsdf.loc[stocklist[z],"Accuracy"]=finalModelAccuracy

        print (stocklist[z]+" analysis is completed")

outputfilename="output"+str(time.time())

outputfilepath="/home/ec2-user/SageMaker/Jayavignesh_stock_prediction/"+outputfilename+".xlsx"

outputfilepath

resultsdf.to_excel(outputfilepath)

