
# coding: utf-8

# In[1]:


import sys
from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from model import P_Net,R_Net,O_Net
import cv2
import os
import numpy as np
import config
import random
from tqdm import tqdm
from utils import *

# In[3]:

def main():
    thresh=config.thresh
    min_face_size=config.min_face
    stride=config.stride
    test_mode=config.test_mode
    detectors=[None,None,None]
    # 模型放置位置
    model_path=['model/PNet/','model/RNet/','model/ONet']
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
    out_path=config.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #选用图片
    path=config.test_dir
    #获取图片类别和路径
    dataset = get_dataset(path)
    random.shuffle(dataset)


    # In[4]:


    bounding_boxes_filename = os.path.join(out_path, 'bounding_boxes.txt')

    with open(bounding_boxes_filename, "w") as text_file:
        for cls in tqdm(dataset):
            output_class_dir = os.path.join(out_path, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                #得到图片名字如001
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.jpg')
                if not os.path.exists(output_filename):
                    try:
                        img = cv2.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<3:
                            print('图片不对劲 "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        
                        img = img[:,:,0:3]
                        #通过mtcnn获取人脸框
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
                                if config.detect_multiple_faces:
                                    for i in range(num_box):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
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
                                try:
                                    scaled =cv2.resize(cropped,(config.image_size, config.image_size),interpolation=cv2.INTER_LINEAR)
                                except:
                                    print('识别不出的图像：{}，box的大小{},{},{},{}'.format(image_path,bb[0],bb[1],bb[2],bb[3]))
                                    continue
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if config.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                    cv2.imwrite(output_filename_n,scaled)
                                    text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('图像不能对齐 "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

if __name__=='__main__':
    main()


# In[2]:
