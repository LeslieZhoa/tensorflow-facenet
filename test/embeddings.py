
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sys
import os
import copy
sys.path.append('../align/')
from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from model import P_Net,R_Net,O_Net
from utils import *
import config
import cv2
import h5py


# In[2]:


def main():
    path='../pictures/embeddings.h5'
    if os.path.exists(path):
        print('生成完了别再瞎费劲了！！！')
        return
    img_arr,class_arr=align_face()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model('../model/')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')

            # 前向传播计算embeddings
            feed_dict = { images_placeholder: img_arr, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
            embs = sess.run(embeddings, feed_dict=feed_dict)
    f=h5py.File('../pictures/embeddings.h5','w')
    class_arr=[i.encode() for i in class_arr]
    f.create_dataset('class_name',data=class_arr)
    f.create_dataset('embeddings',data=embs)
    f.close()


# In[3]:


def align_face(path='../pictures/'):
    thresh=config.thresh
    min_face_size=config.min_face
    stride=config.stride
    test_mode=config.test_mode
    detectors=[None,None,None]
    # 模型放置位置
    model_path=['../align/model/PNet/','../align/model/RNet/','../align/model/ONet']
    batch_size=config.batches
    PNet=FcnDetector(P_Net,model_path[0])
    detectors[0]=PNet


    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet


    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh)
   
    #选用图片
    #获取图片类别和路径
    img_paths=os.listdir(path)
    class_names=[a.split('.')[0] for a in img_paths]
    img_paths=[os.path.join(path,p) for p in img_paths]
    scaled_arr=[]
    class_names_arr=[]
    
    for image_path,class_name in zip(img_paths,class_names):
        
        img = cv2.imread(image_path)
#         cv2.imshow('',img)
#         cv2.waitKey(0)
        try:
            boxes_c,_=mtcnn_detector.detect(img)
        except:
            print('识别不出图像:{}'.format(image_path))
            continue
        #人脸框数量
        num_box=boxes_c.shape[0]
        if num_box>0:
            det=boxes_c[:,:4]
            det_arr=[]
            img_size=np.asarray(img.shape)[:2]
            if num_box>1:
               
                #如果保留一张脸，但存在多张，只保留置信度最大的
                score=boxes_c[:,4]
                index=np.argmax(score)
                det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))
            for i,det in enumerate(det_arr):
                det=np.squeeze(det)
                bb=[int(max(det[0],0)), int(max(det[1],0)), int(min(det[2],img_size[1])), int(min(det[3],img_size[0]))]
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                
                scaled =cv2.resize(cropped,(160, 160),interpolation=cv2.INTER_LINEAR)-127.5/128.0
                scaled_arr.append(scaled)
                class_names_arr.append(class_name)
    
        else:
            print('图像不能对齐 "%s"' % image_path)
    scaled_arr=np.asarray(scaled_arr)
    class_names_arr=np.asarray(class_names_arr)
    return scaled_arr,class_names_arr


# In[4]:


def load_model(model_dir,input_map=None):
    '''重载模型'''
    
    ckpt = tf.train.get_checkpoint_state(model_dir)                         
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')   
    saver.restore(tf.get_default_session(), ckpt.model_checkpoint_path)


# In[ ]:


if __name__=='__main__':
    main()

