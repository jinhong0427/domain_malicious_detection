import tensorflow as tf
from model import DomainNameModel
from work import Work
from  prepro  import preprosample


import numpy as np


from sklearn.utils import shuffle

def pretain():
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
    
   
    huangv =  dosample.ch_to_v(huangdata,words_redic,0)#不归一化
    otherdatav =  dosample.ch_to_v(otherdata,words_redic,0)
    print(huangv[:5],huangdata[:5])
    duv =  dosample.ch_to_v(dudata,words_redic,0)
    yesv =  dosample.ch_to_v(yesdata,words_redic,0)
        
  
    #共分2类
    labelyes = np.array([0]*len(yesdata))
    labelhuang = np.array([1]*len(huangv))
    #labeldu = np.array([2]*len(duv))
    labeldu = np.array([1]*len(duv))#只有两类
    print(len(huangv),len(duv))
    huangv = list(huangv)
    print(len(huangv))
    labelother = np.array([1]*len(otherdatav))#只有两类 
   
    datasample = np.concatenate((yesv,huangv,duv,otherdatav))#np.array(yesv+ huangv+duv)
    labelsample = np.concatenate((labelyes,labelhuang,labeldu,labelother))
    
    print(len(datasample))

    sdatasample, slabelsample=shuffle (datasample,labelsample)
    
    dosample.save_sample(sdatasample,slabelsample)
    


def main(_):
    
    #mode = "evalfreeze"
    mode = "pretrain"

    evaldata = "20171213142927.xls"
    model = DomainNameModel(mode='freeze', learning_rate=0.0008)
    worker = Work(model, batch_size=1024, pretrain_iter=20000, train_iter=900000, sample_iter=100, 
                     model_save_path='model',
                    sample_path='sample',plk_dir='./plk')

    
    if mode == 'pretrain':
        pretain()

    else:#eval
        dosample = preprosample(None,plk_dir='./plk/')
        worker.evalfreeze(evaldata,dosample)
        
if __name__ == '__main__':

    main(_) 