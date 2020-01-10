import os
import numpy as np
import string
import pandas as  pd
import math
import re
import random
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier


if __name__ == '__main__':

    arg_list = sys.argv
    
    path1 =str(arg_list[1])
    path2 = str(arg_list[2])

    path_train_ham = path1+"\\ham"
    path_train_spam = path1+"\\spam"

    path_test_ham = path2+"\\ham"
    path_test_spam = path2+"\\spam"
    

    files_ham=[]
    path=path_train_ham 
    for r,d,f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files_ham.append(os.path.join(r,file))           
    random.seed(0)
    random.shuffle(files_ham)



    files_spam=[]
    path=path_train_spam
    for r,d,f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files_spam.append(os.path.join(r,file))
    random.seed(0)            
    random.shuffle(files_spam)




    final_files_train=files_ham+files_spam
    



    words=[]
    for f in final_files_train:
        with open(f,'r') as f:
            data=f.read()
            data=data.replace('\n',' ')
            words.extend(data.split())




    x=string.printable
    unwantedchar=list(x)
    prepositions = ['and','the','or','are','in', 'to', 'be','is','as', 'by','if','will','as','for','on','it','we','than','this','an','of']
    words=[i for i in words if i not in unwantedchar]
    words= [i for i in words if i not in prepositions]





    vocabulary_final_train=['weight-zero']
    for x in words:
        if x not in vocabulary_final_train:
            vocabulary_final_train.append(x)



    df= pd.DataFrame(columns =vocabulary_final_train)





    count=0

    for f in final_files_train:
        words_list=[]
        with open(f,'r') as f:
            data=f.read()
            data=data.replace('\n',' ')
            words_list.extend(data.split())

        x=string.printable
        unwantedchars=list(x)
        prepositions = ['and','the','or','are','in', 'to', 'be','is','as', 'by','if','will','as','for','on','it','we','than','this','an','of']
        words_list=[i for i in words_list if i not in unwantedchars]
        words_list= [i for i in words_list if i not in prepositions]
    
    
        row=[0]*len(vocabulary_final_train)
  
        for i in range(len(vocabulary_final_train)):
            if(vocabulary_final_train[i]=='weight-zero'):
                row[i]=1
            else:
                if vocabulary_final_train[i] in words_list:
                    row[i]=1
                else:    
                    row[i]=0
            
        df.loc[count]=row
        count+=1




    train_data=df.values




    labels_train=[]
    for f in final_files_train:
        if f in files_ham:
            labels_train.append(1)
        else:
            labels_train.append(0)
    label_train=np.array(labels_train)




    files_test=[]
    path=path_test_ham
    for r,d,f in os.walk(path):
         for file in f:
            if '.txt' in file:
                files_test.append(os.path.join(r,file))
    path=path_test_spam
    for r,d,f in os.walk(path):
         for file in f:
            if '.txt' in file:
                files_test.append(os.path.join(r,file))




    files_test_ham=[]
    path=path_test_ham
    for r,d,f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files_test_ham.append(os.path.join(r,file))


    files_test_spam=[]
    path=path_test_spam
    for r,d,f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files_test_spam.append(os.path.join(r,file))




    labels_test=[]
    for f in files_test:
        if f in files_test_ham:
            labels_test.append(1)
        else:
            labels_test.append(0)




    df_test= pd.DataFrame(columns =vocabulary_final_train) 
    count=0
    for f in files_test:
        words_list=[]
        with open(f,'r') as f:
            data=f.read()
            data=data.replace('\n',' ')
            words_list.extend(data.split())

        x=string.printable
        unwantedchars=list(x)
        prepositions = ['and','the','or','are','in', 'to', 'be','is','as', 'by','if','will','as','for','on','it','we','than','this','an','of']
        words_list=[i for i in words_list if i not in unwantedchars]
        words_list= [i for i in words_list if i not in prepositions]
    
    
    
        row=[0]*len(vocabulary_final_train)
   
        for i in range(len(vocabulary_final_train)):
            if(vocabulary_final_train[i]=='weight-zero'):
                row[i]=1
            else:
                if vocabulary_final_train[i] in words_list:
                    row[i]=1
                else:    
                    row[i]=0
            
        df_test.loc[count]=row
        count+=1
   
    test_data=df_test.values
    print(test_data)




    parameter_grid={'alpha':[0.01,0.1,0.3]}
    model=SGDClassifier(random_state=0,loss='log',penalty='l2',class_weight='balanced',max_iter=1000)
    model_grid=GridSearchCV(estimator=model,param_grid=parameter_grid,n_jobs=-1,scoring='roc_auc')
    model_grid.fit(train_data,labels_train)
    predictions=model_grid.predict(test_data)

    TrueNegative=0
    TruePositive=0
    FalseNegative=0
    FalsePositive=0
    for i in range(len(labels_test)):
        if labels_test[i]==predictions[i]:
            if labels_test[i]==1:
                TrueNegative = TrueNegative+1
            else:
                TruePositive  = TruePositive+1
        else:        
            if labels_test[i]==1:
                FalseNegative = FalseNegative+1
            else:
                FalsePositive = FalsePositive+1
          


    accuracy=float(TrueNegative+TruePositive)/len(test_data)*100
    precision = float(TruePositive)/(TruePositive+FalsePositive)*100
    recall = float(TruePositive)/(TruePositive+FalseNegative)*100
    F1=2*float(precision*recall)/(precision+recall)
    print("Metrics of Bernoulli Model-SGDC")
    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1:",F1)






