
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sys
import os
import copy
from embeddings import load_model
sys.path.append('../align/')
from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from model import P_Net,R_Net,O_Net
import config
import cv2
import h5py
#识别人脸阈值
THRED=0.002


# In[2]:


def main():
    #读取对比图片的embeddings和class_name
    f=h5py.File('../pictures/embeddings.h5','r')
    class_arr=f['class_name'][:]
    class_arr=[k.decode() for k in class_arr]
    emb_arr=f['embeddings'][:]
    cap=cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    path='../output'
    if not os.path.exists(path):
        os.mkdir(path)
    out = cv2.VideoWriter(path+'/out.mp4' ,fourcc,10,(640,480))
    mtcnn_detector=load_align()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model('../model/')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            while True:
                    t1=cv2.getTickCount()
                    ret,frame = cap.read()
                    if ret == True:
                        img,scaled_arr,bb_arr=align_face(frame,mtcnn_detector)
                        if scaled_arr is not None:
                            feed_dict = { images_placeholder: scaled_arr, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
                            embs = sess.run(embeddings, feed_dict=feed_dict)
                            face_num=embs.shape[0]
                            face_class=['Others']*face_num
                            for i in range(face_num):
                                diff=np.mean(np.square(embs[i]-emb_arr),axis=1)
                                min_diff=min(diff)
                                print(min_diff)
                                if min_diff<THRED:
                                    index=np.argmin(diff)
                                    face_class[i]=class_arr[index]
                                    
                            t2=cv2.getTickCount()
                            t=(t2-t1)/cv2.getTickFrequency()
                            fps=1.0/t
                            for i in range(face_num):
                                bbox=bb_arr[i]
                                
                                cv2.putText(img, '{}'.format(face_class[i]), 
                                        (bbox[0], bbox[1] - 2), 
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,(0, 255, 0), 2)
                            
                                #画fps值
                                cv2.putText(img, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        else:
                            img=frame
                        
                        a = out.write(img)
                        cv2.imshow("result", img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        break
            cap.release()
            out.release()
            cv2.destroyAllWindows()


# In[3]:


def load_align():
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
    return mtcnn_detector


# In[4]:


def align_face(img,mtcnn_detector):

    try:
        boxes_c,_=mtcnn_detector.detect(img)
    except:
        print('找不到脸')
        return None,None,None
    #人脸框数量
    num_box=boxes_c.shape[0]
    bb_arr=[]
    scaled_arr=[]
    if num_box>0:
        det=boxes_c[:,:4]
        det_arr=[]
        img_size=np.asarray(img.shape)[:2]
        for i in range(num_box):
            det_arr.append(np.squeeze(det[i]))
            
        for i,det in enumerate(det_arr):
            det=np.squeeze(det)
            bb=[int(max(det[0],0)), int(max(det[1],0)), int(min(det[2],img_size[1])), int(min(det[3],img_size[0]))]
            cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),2)
            bb_arr.append([bb[0],bb[1]])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled =cv2.resize(cropped,(160,160),interpolation=cv2.INTER_LINEAR)
            scaled=cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB)-127.5/128.0
            scaled_arr.append(scaled)
        scaled_arr=np.array(scaled_arr)
        return img,scaled_arr,bb_arr
    else:
        print('找不到脸 ')
        return None,None,None
      


# In[5]:


if __name__=='__main__':
    main()

