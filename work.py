import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
from collections import Counter
import sys
import pandas as pd
from pandas import Series,DataFrame
from tensorflow.python.tools import freeze_graph

from  prepro import preprosample
from sklearn.utils import shuffle
import traceback

class Work(object):

    def __init__(self, model, batch_size=100, pretrain_iter=20000, train_iter=2000, sample_iter=100, 
                 plk_dir='./plk', log_dir='logs', sample_path='sample', 
                 model_save_path='model', test_model='model/dtn-1800'):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.plk_dir = plk_dir
        self.log_dir = log_dir
        self.sample_path = sample_path
        self.model_save_path = model_save_path
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True


            
        
    def load_pkl_sample(self, split='train',plkdir= None):
        print ('loading pkl_sample dataset..')
        sample_file = 'train.pkl' if split=='train' else 'test.pkl'
        if plkdir == None:
            sample_dir = os.path.join(self.plk_dir, sample_file)
        else:
            sample_dir = os.path.join(plkdir, sample_file)
        with open(sample_dir, 'rb') as f:
            sample = pickle.load(f)
        print( np.shape(sample['X']) ,np.shape(sample['X'])[-1])
        print ('finished loading pkl_sample dataset..!')
        return  sample['X'], sample['y']



    def evalfreeze(self,evaldata,dosample):
        words_redic = dosample.load_dic("./words_redic.pkl")
        # load sample dataset
        sample = dosample.load_cvs_evaldata(evaldata)
        print("sample_length:",len(sample))
        
        model = self.model
        model.build_model()
        
               
        
        with tf.Session(graph=model.detection_graph) as sess:
            x = model.detection_graph.get_tensor_by_name('Placeholder:0')
            xlen = model.detection_graph.get_tensor_by_name('Placeholder_1:0')
            result = model.detection_graph.get_tensor_by_name('ArgMax:0')
            print('start eval..!')
            testloop = int(sample.shape[0] / self.batch_size)
           
            badlist = []
            try:
                for i in range(testloop+1):
                    if i == testloop:
                         lastdata = sample.shape[0] % self.batch_size
                         batch_sample = sample[-lastdata:]
                         print("lastdata",lastdata)
                    else:
                        batch_sample = sample[i*self.batch_size:(i+1)*self.batch_size]
                        
                    
                    samplev =  dosample.ch_to_v(batch_sample,words_redic,0)#不归一化
                    samplev,samplelen = dosample.pad_sequences(samplev,  256) 
                    
                    feed_dict = {x: samplev,xlen: samplelen}
                   
                    r = sess.run(result, feed_dict)
                    
                    maliciousname = batch_sample[np.nonzero(r)]


                    if len(maliciousname)>0:
                        badlist = badlist+list(maliciousname)
                        
                        
                    print("No: " ,str(i),"("+str(testloop)+")", 
                         "found:",len(maliciousname),
                         "totalfound:",len(badlist))
                print(badlist)
                np.savetxt("eyn.txt",np.asarray(badlist) ,fmt='%s',newline='\r\n')
            except KeyboardInterrupt:
                print("Ending Eval...")    
                print(badlist)
                np.savetxt("eyn.txt",np.asarray(badlist) ,fmt='%s',newline='\r\n')
           
                
    def doeval(self,evaldata,dosample):
        pass
           
                
    def train(self,evaldata,dosample):
        pass           
                
    def test(self,evaldata,dosample):
        pass             

                
                        
                
                