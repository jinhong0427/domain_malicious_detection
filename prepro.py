import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
from collections import Counter
import pandas as pd
from pandas import Series,DataFrame
from sklearn.utils import shuffle

class preprosample(object):

    def __init__(self, sample_dir='./data', plk_dir='./plk'):

        self.sample_dir = sample_dir
        self.plk_dir = plk_dir
        
    def load_cvs_evaldata(self,evaldata):
        print ('loading  dataset:',evaldata)
        #ISO-8859-1
        #md = pd.read_csv(evaldata,names=list('ABCDE'),skiprows=1,encoding = "ISO-8859-1")#,encoding = "gbk") #,skiprows=1,columns=list('ABCD')                   
        
        #读取tab 分割的文件
        md = pd.read_table(evaldata,names=list('ABCDE'),skiprows=1,encoding = "ISO-8859-1")#,encoding = "gbk") #,skiprows=1,columns=list('ABCD')                   
    
        
        
        return np.array( md['A'] )
            
        
    def load_txt_sample(self, split='train'):
        '''递归。如果是只有一级。就直接返回。如果是有多级，【根，子目录1，子目录2。。。】
        读取txt文件，一行一行的读，传入一个目录，读取目录下的每一个文件
        '''
        print ('loading sample  dataset..')
        
        alldata = []
        for (dirpath, dirnames, filenames) in os.walk(self.sample_dir):#一级一级的文件夹递归
            print(dirpath,dirnames,filenames)
            sdata = []
            for filename in filenames:
                filename_path = os.sep.join([dirpath, filename])  
                with open(filename_path, 'rb') as f:  
                    for onedata in f: 
                        onedata = onedata.strip(b'\n') 
                        try:
                            #print(onedata.decode('gb2312'))#,onedata.decode('gb2312'))'UTF-8'
                            sdata.append(onedata.decode( 'gb2312' ).lower().replace('\r',''))
                        except (UnicodeDecodeError):
                            print("wrong:",onedata.decode)

            alldata.append(sdata)

                    
        print( len(alldata) )
        if len(alldata)>1:
            return alldata
        return sdata
        
    def do_only_sample(self, alldata):
        '''去重  【【正】【负】【负】】
        '''
        
        alldataset = set(alldata[0] ) 
        dudataset = set(alldata[1] )  
        huangdataset = set(alldata[2] )  
        otherset = set(alldata[3])
        print(len(alldataset))
        yesdataset = (alldataset-dudataset)-huangdataset
        print(len(yesdataset))
        return list(yesdataset),list(dudataset),list(huangdataset),list(otherset)
        
    def make_dictionary(self,alldata):
        all_words = set()  
        for onedata in alldata:  
            #print(label)
            for oneurl in onedata:
                oneset= set(oneurl)
                #print(len(oneset))
                all_words = oneset|all_words
#        counter = Counter(all_words)#词频
#        print(counter)  
        print(len(all_words))
        print(all_words)
        
       
        
        counter = all_words
        words_dic = list(counter)#排序--正向字典
        words_dic.insert(0,'None')#补0用的
        print(words_dic)
        
        words_dic.remove('\x7f') 
        words_dic.remove('\x13')
        words_dic.remove('\x0e')
        words_dic.remove('\x01')
#        words_dic.remove('消')
#        words_dic.remove('息')
#        words_dic.remove('格')
#        words_dic.remove('式')
#        words_dic.remove('错')
#        words_dic.remove('误')
#        words_dic.remove('文')
#        words_dic.remove('本')        
        words_size= len(words_dic)
        words_redic = dict(zip(words_dic, range(words_size))) #反向字典
        print(words_redic)
        print('字表大小:', words_size)
        return words_dic,words_redic

    def wordtov(self,words_redic,word) :
        if words_redic.has_key(word):
            return words_redic[word]
        else:
            return 0 # 字典里没有的就是None
               
    #字符到向量
    def ch_to_v(self,datalist,words_redic,normal = 1):
        
        to_num = lambda word: words_redic[word] if word in words_redic else 0# 字典里没有的就是None

        data_vector =[]
        for ii in datalist:
            one_vector = list(map(to_num, list(ii))) 
            data_vector.append(np.asarray(one_vector))         
        #归一化
        if normal == 1:
            return np.asarray(data_vector)/ (len(words_redic)/2) - 1 
        return data_vector
        

    def pad_sequences(self,sequences, maxlen=None, dtype=np.float32,
                      padding='post', truncating='post', value=0.):
    
        
        lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    
        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)
    
        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
    
        x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)
    
            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))
    
            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x, lengths


    def save_pickle(self,data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print ('Saved %s..' %path)
            
    def load_dic(self,dic):
        print ('loading dic ..')

        #dic_dir = os.path.join(self.plk_dir, dic)
        with open(dic, 'rb') as f:
            dicdata = pickle.load(f)

        print ('finished loading  dic..!')
        return  dicdata
    
    def save_sample(self,sdatasample,slabelsample,maketrain = 1):
        
        if maketrain == 1:
            lendata = int(len(slabelsample)*0.95)
        else:
            lendata = int(len(slabelsample))
        
        train = {'X': sdatasample[:lendata],
                 'y': slabelsample[:lendata]}
        
        test = {'X': sdatasample[lendata:],
                'y': slabelsample[lendata:]}
                
#        if not os.path.exists(self.plk_dir):
#            os.mkdir(self.plk_dir)
            
        # make directory if not exists
        if tf.gfile.Exists(self.plk_dir):
            tf.gfile.DeleteRecursively(self.plk_dir)
        tf.gfile.MakeDirs(self.plk_dir)
            
        self.save_pickle(train, self.plk_dir+'/train.pkl')
        if maketrain == 1:
            self.save_pickle(test, self.plk_dir+'/test.pkl')

                
                


                
def main():

    dosample = preprosample('./data/',plk_dir='./plk/')
    alldata = dosample.load_txt_sample() 
    
    words_dic={}
    words_redic={}
    if tf.gfile.Exists('./words_dic.pkl'):  #########长度50
        words_dic = dosample.load_dic("./words_dic.pkl")
        words_redic = dosample.load_dic("./words_redic.pkl")
    else:
        words_dic,words_redic = dosample.make_dictionary(alldata)
        dosample.save_pickle(words_dic, './words_dic.pkl')
        dosample.save_pickle(words_redic, './words_redic.pkl')
    print(len(words_dic))
    print(words_dic)
    print(words_redic)
    yesdata,dudata,huangdata,otherdata = dosample.do_only_sample(alldata)
    print("yes",len(yesdata))
    print("dudata",len(dudata))
    print("huangdata",len(huangdata))
    print("otherdata",len(otherdata))        


        
if __name__ == '__main__':
    main()   
