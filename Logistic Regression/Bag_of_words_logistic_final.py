

import os
import numpy as np
import string
import pandas as pd
import math
import re
import random
import sys


def Sigmoid(x):
    S=(1/(1+np.exp(-x)))
    return S

def Accuracy(data,update_w,label):
    TrueNegative=0
    TruePositive=0
    FalseNegative=0
    FalsePositive=0
    accuracy=0
    precision=0
    recall=0
    F1=0
    y=np.array(np.dot(data,update_w),dtype=np.float32)
    #print(y.shape)
    #print(len(label))
    predicted_label=[]
    for i in y:
        if i>0:
            predicted_label.append(1)
        else:
            predicted_label.append(0)
    for i in range(len(label)):
        if label[i]==predicted_label[i]:
            if label[i]==1:
                TrueNegative = TrueNegative+1
            else:
                TruePositive  = TruePositive+1
        else:        
            if label[i]==1:
                FalseNegative = FalseNegative+1
            else:
                FalsePositive = FalsePositive+1
    accuracy=float(TrueNegative+TruePositive)/len(data)*100
    precision = float(TruePositive)/(TruePositive+FalsePositive)*100
    recall = float(TruePositive)/(TruePositive+FalseNegative)*100
    F1=2*float(precision*recall)/(precision+recall)
    return accuracy,precision,recall,F1

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


    #print("spam_len",len(files_spam))
    #print("ham_len",len(files_ham))


    final_files_train=files_ham+files_spam



 
    l=len(files_ham)
    d=int(0.7*l)
    files_ham_train=[]
    files_ham_valid=[]
    for i in range(l):
        if i>d or i==d:
            files_ham_valid.append(files_ham[i])
        else:
            files_ham_train.append(files_ham[i])
 




    l=len(files_spam)
    d=int(0.7*l)
    files_spam_train=[]
    files_spam_valid=[]
    for i in range(l):
        if i>d or i==d:
            files_spam_valid.append(files_spam[i])
        else:
            files_spam_train.append(files_spam[i])





    partial_files_train=files_ham_train+files_spam_train




    words=[]
    for f in partial_files_train:
        with open(f,'r') as f:
            data=f.read()
            data=data.replace('\n',' ')
            words.extend(data.split())




    x=string.printable
    unwantedchar=list(x)
    prepositions = ['and','the','or','are','in', 'to', 'be','is','as', 'by','if','will'
    ,'as','for','on','it','we','than','this','an','of']
    words=[i for i in words if i not in unwantedchar]
    words= [i for i in words if i not in prepositions]





    vocabulary=['weight-zero']
    for x in words:
        if x not in vocabulary:
            vocabulary.append(x)





    df_partial_train= pd.DataFrame(columns =vocabulary)

    count=0
    for f in partial_files_train:
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
        row=[0]*len(vocabulary)
        for i in range(len(vocabulary)):
            if(vocabulary[i]=='weight-zero'):
                row[i]=1
            else:
                if vocabulary[i] in words_list:
                    row[i]=1
                else:
                    feature_count=words_list.count(vocabulary[i])
                    row[i]=feature_count
        df_partial_train.loc[count]=row
        count+=1


    #print("df_train_row_len",len(df_partial_train.index))

    labels_partial_train=[]
    for f in partial_files_train:
        if f in files_ham_train:
            labels_partial_train.append(1)
        else:
            labels_partial_train.append(0)
    label_partial_train=np.array(labels_partial_train)




    partial_train_data=df_partial_train.values



    partial_weights=np.zeros(partial_train_data.shape[1])






    files_valid=files_ham_valid+files_spam_valid





    labels_valid=[]
    for f in files_valid:
        if f in files_ham_valid:
            labels_valid.append(1)
        else:
            labels_valid.append(0)




    df_valid= pd.DataFrame(columns =vocabulary)




    count=0
    for f in files_valid:
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
        row=[0]*len(vocabulary)
        for i in range(len(vocabulary)):
            if(vocabulary[i]=='weight-zero'):
                row[i]=1
            else:
                feature_count=words_list.count(vocabulary[i])
                row[i]=feature_count
        df_valid.loc[count]=row
        count+=1








    Lambda=[0.1,0.5,0.7,0.9]
    UA=0
    for i in Lambda:
        weights=np.zeros(partial_train_data.shape[1])
        for j in range(200):
            x=np.array(np.dot(partial_train_data,partial_weights),dtype=np.float32)
            S=Sigmoid(x)
            L=label_partial_train
            g=np.dot(partial_train_data.T,L-S)
            partial_weights=partial_weights+(0.01*(g))-(0.01*i*partial_weights)
        valid_data=df_valid.values
        A, P, R, F=Accuracy(valid_data,partial_weights,labels_valid)
        if A>UA or A==UA:
            Final_lambda=i
            UA=A




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
                feature_count=words_list.count(vocabulary_final_train[i])
                row[i]=feature_count
        df.loc[count]=row
        count+=1
    labels_train=[]
    for f in final_files_train:
        if f in files_ham:
            labels_train.append(1)
        else:
            labels_train.append(0)
    label_train=np.array(labels_train)
    
    train_data=df.values


    weights=np.zeros(train_data.shape[1])



    for j in range(1000):
        x=np.array(np.dot(train_data,weights),dtype=np.float32)
        S=Sigmoid(x)
        L=label_train
        g=np.dot(train_data.T,L-S)
        weights=weights+(0.01*(g))-(0.01*Final_lambda*weights)



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
                feature_count=words_list.count(vocabulary_final_train[i])
                row[i]=feature_count
        df_test.loc[count]=row
        count+=1



    data_test=df_test.values
    #print("data test shape",data_test.shape)
    A,P,R,F=Accuracy(data_test,weights,labels_test)
    print("Metrics of Multinomial model-Logistic")
    print("Accuracy:",A)
    print("Precesion:",P)
    print("Recall",R)
    print("F1",F)







 
