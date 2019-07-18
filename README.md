# tensorflow-facenet
人脸识别算法，结合facenet网络结构和center loss作为损失，基于tensorflow框架，含训练和测试代码，支持从头训练和摄像头测试<br>
## 模型介绍
本代码在训练开始前通过MTCNN网络对数据集的图片进行筛选，筛选出能识别的人脸图像，并通过人脸框将图片裁剪resize成一定尺度用于模型的输入，对于[人脸检测MTCNN算法的讲解](https://github.com/LeslieZhoa/tensorflow-MTCNN)，我的另一篇项目中做了详尽的介绍和代码注释，本代码的实现也是在此基础上简单修改。<br><br>
本代码参考[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)的主体结构，即使用inception_resnet_v1作为模型主体架构,输出一定维度的图片embeddings。facenet的损失函数采用triplet_loss,即对于某张训练图片img,再选取一张同一类别一张图作为pos，选取不同类别的一张图作为neg。img的embeddings与pos的embeddings的平方和作为pos_dist,im的embeddings和neg的embeddings的平方和作为neg_dist，使pos_dist与neg_dist的差总保持一定阈值。实际训练中在每一batch中对于某一张图，选取同一类别图像作为pos，选取embeddings平方和大于pos的不同类别图像作为neg，依次构成三元组来训练模型。triplet_loss的公式如下所示：<br>
![](https://github.com/LeslieZhoa/tensorflow-facenet/blob/master/pictures/loss1.png)<br>
但是这样训练三元组并不能确保每次训练的数据的pos和neg图片都是最难识别的，而且模型收敛极慢。[A Discriminative Feature Learning Approach for Deep Face Recognition](https://link.springer.com/chapter/10.1007%2F978-3-319-46478-7_31)提出center_loss，可以对模型很快收敛。本代码的损失函数就参考center_loss论文中损失函数。损失公式如下所示：<br>
![](https://github.com/LeslieZhoa/tensorflow-facenet/blob/master/pictures/loss2.png)<br>
代码中的损失函数采用softmax交叉熵和center_loss相结合。softmax交叉熵为了使类间距离变大，center_loss是计算某一图片与该类别图片embeddings的均值的损失，为了使类间距离变小。论文中采取minst实验来说明损失函数的功能，如下图所示：<br>
![](https://github.com/LeslieZhoa/tensorflow-facenet/blob/master/pictures/loss2-1.png)
![](https://github.com/LeslieZhoa/tensorflow-facenet/blob/master/pictures/loss2-2.png)<br>
在实际训练中并不是遍历整个数据集来求取某类别center的embeddings，而是只针对每一个batch中出现的类别的center通过batch内相同类别的图像embeddings来进行一定学习率的更新，未出现该批次的类别的center不进行更新。<br><br>
训练过程中是有类别标记的，那在实际视频测试的时候的验证身份是没有在训练集中出现的类别该怎么办？对于实际测试的时候，有身份图片，和待验证图片，主要就是计算两者embeddings之间的平方和均值与阈值进行比较，当待验证图片与身份图片的embeddings的平方和均值小于一定阈值就判定待验证图片的身份。<br><br>
我的代码只训练了15个epoch，如果你有兴趣可以在我的基础上重载继续训练或者重新训练，我在筛选图像的时候花费了好长时间，所以要做好心理准备。<br>
## 代码介绍
本代码参考了[davidsandberg的facenet](https://github.com/davidsandberg/facenet)<br>
### 环境说明
ubuntu16.04<br>
python3.6.5<br>
tensorflow1.8.0<br>
opencv3.4.3<br>
pip install tqdm为了显示进度条<br>
pip install h5py测试时候存储身份图片的embeddings<br>
### 代码介绍
align下放置的是MTCNN的相关代码，align_mtcnn.py是筛选图像的主程序，其他代码具体都取自[我的MTCNN的项目](https://github.com/LeslieZhoa/tensorflow-MTCNN)<br><br>
data下放置的是训练数据以及筛选后的数据<br><br>
pictures下放置的是你需要对比的身份图像，特别强调文件名称要是全英文路径，opencv识别中文路径和添加中文比较费劲我就没加这一功能。<br><br>
test下放置的embeddings.py是为了生成身份图像的embeddings省得每次测试都要重新生成,test.py是通过摄像头来识别人的身份<br><br>
train里放置的config.py是一些参数设定，inception_resnet_v1.py是模型主体结构，train.py是训练代码<br>
### 下载数据
如果你想自己训练，下载[CASIA-WebFace数据](https://pan.baidu.com/s/1hQCOD4Kr66MOW0_PE8bL0w)，提取密码是y3wj，多谢[Yang Fang](https://github.com/Yangel-hide)的提供。将CASIA-WebFace解压到data目录下。<br><br>
如果你想继续我的模型训练或者想纯粹跑一下代码，那么你可以下载我的[训练模型](https://www.jianguoyun.com/p/DdD70RYQv7mYBxjE5YoB)解压里面的模型文件到model目录下。<br>
### 运行
训练:<br><br>
将目录cd到align下，<br>
python align_mtcnn.py生成筛选的人脸图像<br><br>
将目录cd到train下，<br>
python train.py进行训练<br><br>
测试：<br><br>
将你的身份图片放置到pictures目录下<br><br>
将目录cd到test下<br>
python embeddings.py生成身份图像的embeddings并存储<br>
python test.py测试摄像头读取图像的身份，可以自行改变[阈值的选取](https://github.com/LeslieZhoa/tensorflow-facenet/blob/master/test/test.py#L22)<br><br>
如果有什么问题还请留言多多指教！<br>
### 注意
align/utils.py line95 image=tf.cast(image,tf.float32)-127.5/128.0应改为image=(tf.cast(image,tf.float32)-127.5)/128.0<br>
相应test/test.py line152 scaled=cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB)-127.5/128.0应改为scaled=(cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB)-127.5)/128.0<br>
test/embeddings.py的line115的scaled =cv2.resize(cropped,(160, 160),interpolation=cv2.INTER_LINEAR)-127.5/128.0改为scaled =(cv2.resize(cropped,(160, 160),interpolation=cv2.INTER_LINEAR)-127.5)/128.0<br><br>
由于本人失误我的模型应该就是基于此训练的，所以如果要改正需要重新训练模型，不过我想效果应该会更好！<br>
多谢[@ltpjob](https://github.com/ltpjob)的issues7<br>
希望大家继续批评指正！！！
## 结果展示
结果是我通过摄像头的身份验证，其中身份图像来源于百度图片，手机上的图片来源于新浪微博。<br>
![](https://github.com/LeslieZhoa/tensorflow-facenet/blob/master/pictures/out.mp4_20181204_201824.gif)


