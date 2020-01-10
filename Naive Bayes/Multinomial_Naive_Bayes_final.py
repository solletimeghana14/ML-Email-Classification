

import pandas as pd
import numpy as np
import math
import os
import glob
import string
import re
import math
import sys





def Get_Files_List(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
                
    return files





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




def BagOfWords(files_train_ham, files_train_spam, words_array_train):
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
        count_words_unique_file = [0]*len(count_words_unique_total)
        
        words_unique_file, count_words_unique_file_temp = np.unique(words_arr_mail, return_counts = True)
        common_words,common_indices_total, common_indices_file = np.intersect1d(words_columns, words_unique_file, return_indices=True)
        for i in range(len(common_indices_total)):
            count_words_unique_file[common_indices_total[i]] = count_words_unique_file_temp[common_indices_file[i]]
    
        p = list(count_words_unique_file)
        p.append(0)
        df.loc[itr] = p
        itr += 1
        
    for f in files_train_spam:
        words_list_file = []
        with open(f,'r') as file:
            data = file.read().replace('\n',' ')
            words_list_file.extend(data.split())
    
        words_arr_mail = Clean_Words(words_list_file)
        count_words_unique_file = [0]*len(count_words_unique_total)
        
        words_unique_file, count_words_unique_file_temp = np.unique(words_arr_mail, return_counts = True)
        common_words,common_indices_total, common_indices_file = np.intersect1d(words_columns, words_unique_file, return_indices=True)
        for i in range(len(common_indices_total)):
            count_words_unique_file[common_indices_total[i]] = count_words_unique_file_temp[common_indices_file[i]]
    
        p = list(count_words_unique_file)
        p.append(1)
        df.loc[itr] = p
        itr += 1

    return df



def Multinomial_Naive_Bayes(files_test_ham, files_test_spam, df):
    dfSpam = df[df['labels_Ans'] == 1]
    dfHam = df[df['labels_Ans'] == 0]
    
    ham_mail_count = len(dfHam.index)
    spam_mail_count = len(dfSpam.index)
    PHam = math.log(ham_mail_count,2) - math.log(ham_mail_count+spam_mail_count,2)
    PSpam = math.log(spam_mail_count,2) - math.log(ham_mail_count+spam_mail_count,2)
    
    spam_words_count_arr = dfSpam.sum()
    count_words_spam = spam_words_count_arr.sum()-dfSpam['labels_Ans'].sum()
        
    ham_words_count_arr = dfHam.sum()-dfHam['labels_Ans'].sum()
    count_words_ham = ham_words_count_arr.sum()
    
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
                PoWgH = math.log(1+dfHam[words_arr_mail[i]].sum(),2)-math.log(len(df.columns)-1+count_words_ham,2)
                PoWgS = math.log(1+dfSpam[words_arr_mail[i]].sum(),2)-math.log(len(df.columns)-1+count_words_spam,2)

        
            else:
                PoWgH = -math.log(len(df.columns)-1+count_words_ham,2)
                PoWgS = -math.log(len(df.columns)-1+count_words_spam,2)
            #PWordH =  -math.log(count_words_total,2)
            #PWordS =  -math.log(count_words_total,2)
     
    
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



if __name__ == '__main__':

    arg_list = sys.argv
    
    path1 =str(arg_list[1])
    path2 = str(arg_list[2])

    path_train_ham = path1+"\\ham"
    path_train_spam = path1+"\\spam"
    files_train_ham = Get_Files_List(path_train_ham)
    files_train_spam = Get_Files_List(path_train_spam)

    words_array_train = Process_train_data(files_train_ham, files_train_spam)
   
    Data_BoW = BagOfWords(files_train_ham, files_train_spam, words_array_train)




    path_test_ham = path2+"\\ham"
    path_test_spam = path2+"\\spam"
    files_test_ham = Get_Files_List(path_test_ham)
    files_test_spam = Get_Files_List(path_test_spam)

    

    Accuracy, Precision, Recall,F1 = Multinomial_Naive_Bayes(files_test_ham, files_test_spam, Data_BoW)




    print("Metrics in Multinomial Naive Bayes-Bag of Words")
    print("Accuracy:",Accuracy)
    print("Precision:",Precision)
    print("Recall:",Recall)
    print("F1:",F1)







