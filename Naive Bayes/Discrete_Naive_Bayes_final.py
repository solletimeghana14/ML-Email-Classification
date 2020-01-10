import pandas as pd
import numpy as np
import math
import os
import glob
import string
import re
import math
import sys


# In[73]:


def Get_Files_List(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
                
    return files


# In[74]:


def Clean_Words(words_list):
    x =string.printable
    waste = list(x)
    prepositions = ['and','the','or','are','in', 'to', 'be','is','as', 'by','if','will','as','for','on','it','we','than','this','an'
                 ]
    words_list = [i for i in words_list if i not in waste]
    words_list = [i for i in words_list if i.isalnum() is True]
    words_list = [i for i in words_list if i not in prepositions]
    words_list = [i for i in words_list if len(i)!=2]
    
    words_arr = np.array(words_list)
    
    return words_arr


# In[75]:


def Process_train_data(files_train_ham, files_train_spam):
    words_list_total = []
    for f in files_train_ham:
        with open(f,'r') as file:
            data = file.read().replace('\n',' ')
            words_list_total.extend(data.split())
    
    for f in files_train_spam:
        with open(f,'r') as file:
            data = file.read().replace('\n',' ')
            words_list_total.extend(data.split())
            
    words_arr = Clean_Words(words_list_total)
    
    return words_arr


# In[76]:


def Bernoulli_Model(files_train_ham, files_train_spam, words_array_train):
    words_columns, count_words_unique_total = np.unique(words_array_train, return_counts = True)
    Pcolumns = list(words_columns)
    Pcolumns.append('labels_Ans')
    df = pd.DataFrame(columns = Pcolumns)
    
    itr = 0
    for f in files_train_ham:
        words_list_file = []
        with open(f,'r') as file:
            data = file.read().replace('\n',' ')
            words_list_file.extend(data.split())
            
        words_arr_mail = Clean_Words(words_list_file)
        
        presence_words_unique_file = [0]*len(words_columns)
        words_unique_file, count_words_unique_file_temp = np.unique(words_arr_mail, return_counts = True)
    
        common_words,common_indices_total, common_indices_file = np.intersect1d(words_columns, words_unique_file, return_indices=True)
        for i in common_indices_total:
            presence_words_unique_file[i] = 1
        
        p = list(presence_words_unique_file) 
        p.append(0)
        df.loc[itr] = p
        itr += 1
        
    for f in files_train_spam:
        words_list_file = []
        with open(f,'r') as file:
            data = file.read().replace('\n',' ')
            words_list_file.extend(data.split())
            
        words_arr_mail = Clean_Words(words_list_file)
        
        presence_words_unique_file = [0]*len(words_columns)
        words_unique_file, count_words_unique_file_temp = np.unique(words_arr_mail, return_counts = True)
    
        common_words,common_indices_total, common_indices_file = np.intersect1d(words_columns, words_unique_file, return_indices=True)
        for i in common_indices_total:
            presence_words_unique_file[i] = 1
        
        p = list(presence_words_unique_file) 
        p.append(1)
        df.loc[itr] = p
        itr += 1
    
    return df


# In[77]:


def Multivariate_Naive_Bayes(files_test_ham, files_test_spam, df):
    dfSpam = df[df['labels_Ans'] == 1]
    dfHam = df[df['labels_Ans'] == 0]
    ham_mail_count = len(dfHam.index)
    spam_mail_count = len(dfSpam.index)
    PHam = math.log(ham_mail_count,2) - math.log(ham_mail_count+spam_mail_count,2)
    PSpam = math.log(spam_mail_count,2) - math.log(ham_mail_count+spam_mail_count,2)
    
    for i in range(len(df.columns)-1):
        x= 1 - float(1+dfHam[df.columns[i]].sum())/(2+ham_mail_count)
        y = 1 - float(1+dfSpam[df.columns[i]].sum())/(2+spam_mail_count)
        PoHgnotW = math.log(x,2)
        PoSgnotW = math.log(y,2)
        PHam += PoHgnotW
        PSpam += PoSgnotW
    
    TruePositive = 0
    TrueNegative = 0
    FalsePositive = 0
    FalseNegative = 0
    files_test = files_test_ham + files_test_spam
    for f in files_test:
        words_list = []
        with open(f,'r') as file:
            data = file.read().replace('\n',' ')
            words_list.extend(data.split())
    
        words_arr_mail = Clean_Words(words_list)
        
        PHam_Mail= PHam
        PSpam_Mail = PSpam
        
        for i in range(len(words_arr_mail)):
            if words_arr_mail[i] in df.columns:
                PoWgH = math.log(1+dfHam[words_arr_mail[i]].sum(),2)-math.log(2+ham_mail_count,2)
                PoWgS = math.log(1+dfSpam[words_arr_mail[i]].sum(),2)-math.log(2+spam_mail_count,2)
                x= 1 - float(1+dfHam[words_arr_mail[i]].sum())/(2+ham_mail_count)
                y = 1 - float(1+dfSpam[words_arr_mail[i]].sum())/(2+spam_mail_count)
                PoWgH = PoWgH - math.log(x,2)
                PoWgS = PoWgS - math.log(y,2)
                PHam_Mail += PoWgH
                PSpam_Mail += PoWgS
            
            else:
                PoWgH = -math.log(2+ham_mail_count,2)
                PoWgS = -math.log(2+spam_mail_count,2)
                PHam_Mail += PoWgH
                PSpam_Mail += PoWgS
            
    
        
        if(PHam_Mail >= PSpam_Mail):
            if f in files_test_ham:
                TrueNegative += 1
        
            else:
                FalseNegative += 1
            
        #print("mail is not spam")
    
        else:
            if f in files_test_spam:
                TruePositive += 1
        
            else:
                FalsePositive += 1
        #print("mail is spam")
    Accuracy = float(TruePositive + TrueNegative)/len(files_test)
    precision = float(TruePositive)/(TruePositive+FalsePositive)
    recall = float(TruePositive)/(TruePositive+FalseNegative)
    F1=2*float(precision*recall)/(precision+recall)
    
    return Accuracy, precision, recall,F1


# In[7]:
if __name__ == '__main__':

    arg_list = sys.argv
    
    path1 =str(arg_list[1])
    path2 = str(arg_list[2])

    path_train_ham = path1+"\\ham"
    path_train_spam = path1+"\\spam"
    files_train_ham = Get_Files_List(path_train_ham)
    files_train_spam = Get_Files_List(path_train_spam)

   
    words_array_train = Process_train_data(files_train_ham, files_train_spam)
    

    Data_Bernoulli = Bernoulli_Model(files_train_ham, files_train_spam, words_array_train)

    path_test_ham = path2+"\\ham"
    path_test_spam = path2+"\\spam"
    files_test_ham = Get_Files_List(path_test_ham)
    files_test_spam = Get_Files_List(path_test_spam)

    

    Accuracy, Precision, Recall,F1 = Multivariate_Naive_Bayes(files_test_ham, files_test_spam, Data_Bernoulli)


# In[93]:

    print("Metrics in Discrete Naive Bayes-Bernoulli model")
    print("Accuracy:",Accuracy)
    print("Precision:",Precision)
    print("Recall:",Recall)
    print("F1:",F1)

# In[ ]:




