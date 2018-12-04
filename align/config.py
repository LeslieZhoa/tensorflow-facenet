
# coding: utf-8

#最小脸大小设定
min_face=20

#生成hard_example的batch
batches=[2048,256,16]
#pent对图像缩小倍数
stride=2
#三个网络的阈值
thresh=[0.6,0.7,0.7]
#最后测试选择的网络
test_mode='ONet'
#测试图片放置位置
test_dir='../data/CASIA-WebFace/'
#测试输出位置
out_path='../data/casia_mtcnn_182/'
#一张图是否获取多张脸
detect_multiple_faces=False
#输出图片大小
image_size=182
