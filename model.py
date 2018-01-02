import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

class DomainNameModel(object):
    """DomainNameModel
    """
    def __init__(self, mode='train', learning_rate=0.001,dim=64, steps=256,  n_hidden = 64,n_class = 2,wordsize = 50):
        self.mode = mode
        self.learning_rate = learning_rate
        self.steps = steps
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.dim = dim
        self.wordsize = wordsize
        
    def BirnnClassify(self,data,seq_lengths,reuse= False):
        pass
                

                
    def build_model(self):
        
        if self.mode == 'train':        
            pass
            
        elif self.mode == 'eval':
            pass
        elif self.mode =='freeze':

            save_path='BirnnClassify'
            if not os.path.exists(save_path):
                #raise IOError("there is not a model path:"+save_path)
                print("there is not a model path:",save_path)
            
            savedir = save_path
            PATH_TO_CKPT = savedir +'/expert-graph-yes.pb'
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')   
              
            self.detection_graph = tf.get_default_graph() 

