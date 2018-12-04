
# coding: utf-8

# In[1]:


import numpy as np
import os
from tqdm import tqdm
import math
import tensorflow as tf
from scipy import misc

# In[2]:


def IOU(box,boxes):
    '''裁剪的box和图片所有人脸box的iou值
    参数：
      box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
      boxes：图片所有人脸box,[n,4]
    返回值：
      iou值，[n,]
    '''
    #box面积
    box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
    #boxes面积,[n,]
    area=(boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    #重叠部分左上右下坐标
    xx1=np.maximum(box[0],boxes[:,0])
    yy1=np.maximum(box[1],boxes[:,1])
    xx2=np.minimum(box[2],boxes[:,2])
    yy2=np.minimum(box[3],boxes[:,3])

    #重叠部分长宽
    w=np.maximum(0,xx2-xx1+1)
    h=np.maximum(0,yy2-yy1+1)
    #重叠部分面积
    inter=w*h
    return inter/(box_area+area-inter+1e-10)


# In[3]:

def read_annotation(base_dir, label_path):
    '''读取文件的image，box'''
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # 图像地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/images/' + imagepath
        images.append(imagepath)
        # 人脸数目
        nums = labelfile.readline().strip('\n')

        one_image_bboxes = []
        for i in range(int(nums)):

            bb_info = labelfile.readline().strip('\n').split(' ')
            #人脸框
            face_box = [float(bb_info[i]) for i in range(4)]

            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]

            one_image_bboxes.append([xmin, ymin, xmax, ymax])

        bboxes.append(one_image_bboxes)


    data['images'] = images
    data['bboxes'] = bboxes
    return data
def convert_to_square(box):
    '''将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    '''
    square_box=box.copy()
    h=box[:,3]-box[:,1]+1
    w=box[:,2]-box[:,0]+1
    #找寻正方形最大边长
    max_side=np.maximum(w,h)

    square_box[:,0]=box[:,0]+w*0.5-max_side*0.5
    square_box[:,1]=box[:,1]+h*0.5-max_side*0.5
    square_box[:,2]=square_box[:,0]+max_side-1
    square_box[:,3]=square_box[:,1]+max_side-1
    return square_box
class ImageClass():
    '''获取图片类别和路径'''
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(paths):
    dataset = []
    classes = [path for path in os.listdir(paths) if os.path.isdir(os.path.join(paths, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in tqdm(range(nrof_classes)):
        class_name = classes[i]
        facedir = os.path.join(paths, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def split_dataset(dataset,split_ratio,min_nrof_images_per_class):
    '''拆分训练和验证集
    参数：
      dataset:有get_dataset生成的数据集
      split_ratio:留取验证集的比例
      min_nrof_images_per_class：一个类别中最少含有的图片数量，过少舍弃
    返回值：
      train_set,test_set:还有图片类别和路径的训练验证集
    '''
    train_set=[]
    test_set=[]
    for cls in dataset:
        paths=cls.image_paths
        np.random.shuffle(paths)
        #某一种类图片个数
        nrof_images_in_class=len(paths)
        #留取训练的比例
        split=int(math.floor(nrof_images_in_class*(1-split_ratio)))
        if split==nrof_images_in_class:
            split=nrof_images_in_class-1
        if split>=min_nrof_images_per_class and nrof_images_in_class-split>=1:
            train_set.append(ImageClass(cls.name,paths[:split]))
            test_set.append(ImageClass(cls.name,paths[split:]))
    return train_set,test_set

def get_image_paths_and_labels(dataset):
    '''获取所有图像地址和类别'''
    image_paths_flat=[]
    labels_flat=[]
    for i in range(len(dataset)):
        image_paths_flat+=dataset[i].image_paths
        labels_flat+=[i]*len(dataset[i].image_paths)
    return image_paths_flat,labels_flat

def create_input_pipeline(input_queue,image_size,nrof_preprocess_threads,bath_size_placeholder):
    '''由输入队列返回图片和label的batch组合
    参数：
      input_queue:输入队列
      image_size:图片尺寸
      nrof_preprocess_threads:线程数
      batch_size_placeholder:batch_size的placeholder
    返回值：
      image_batch,label_batch:图片和label的batch组合
    '''
    image_and_labels_list=[]
    for _ in range(nrof_preprocess_threads):
        filenames,label=input_queue.dequeue()
        images=[]
        for filename in tf.unstack(filenames):
            file_contents=tf.read_file(filename)
            image=tf.image.decode_image(file_contents,3)
            #随机翻转图像
            image=tf.cond(tf.constant(np.random.uniform()>0.8),
                          lambda:tf.py_func(random_rotate_image,[image],tf.uint8),
                          lambda:tf.identity(image))
            #随机裁剪图像
            image=tf.cond(tf.constant(np.random.uniform()>0.5),
                          lambda:tf.random_crop(image,image_size+(3,)),
                          lambda:tf.image.resize_image_with_crop_or_pad(image,image_size[0],image_size[1]))
            #随机左右翻转图像
            image=tf.cond(tf.constant(np.random.uniform()>0.7),
                          lambda:tf.image.random_flip_left_right(image),
                          lambda:tf.identity(image))
            #图像归一到[-1,1]内
            image=tf.cast(image,tf.float32)-127.5/128.0
            image.set_shape(image_size+(3,))
            images.append(image)
        image_and_labels_list.append([images,label])
    image_batch,label_batch=tf.train.batch_join(image_and_labels_list,
                                  batch_size=bath_size_placeholder,
                                  shapes=[image_size+(3,),()],
                                  enqueue_many=True,
                                  capacity=4*nrof_preprocess_threads*100,
                                  allow_smaller_final_batch=True)
    return image_batch,label_batch

def random_rotate_image(image):
    '''随机翻转图片'''
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')
