
# coding: utf-8

# In[1]:


import os
import random
import tensorflow as tf
import numpy as np
slim=tf.contrib.slim
import inception_resnet_v1 as network
import config
import sys
sys.path.append('../')
from align.utils import *


# In[ ]:


def main():
    image_size=(config.image_size,config.image_size)
    #创建graph和model存放目录
    if not os.path.exists(config.graph_dir):
        os.mkdir(config.graph_dir)
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    #获取图片地址和类别
    dataset=get_dataset(config.data_dir)
    #划分训练验证集
    if config.validation_set_split_ratio>0.0:
        train_set,val_set=split_dataset(dataset,config.validation_set_split_ratio,config.min_nrof_val_images_per_class)
    else:
        train_set,val_set=dataset,[]
    #训练集的种类数量
    nrof_classes=len(train_set)
    with tf.Graph().as_default():
        global_step=tf.Variable(0,trainable=False)
        #获取所有图像位置和相应类别
        image_list,label_list=get_image_paths_and_labels(train_set)
        assert len(image_list)>0, '训练集不能为空'
        val_image_list,val_label_list=get_image_paths_and_labels(val_set)

        labels=tf.convert_to_tensor(label_list,dtype=tf.int32)
        #样本数量
        range_size=labels.get_shape().as_list()[0]
        #每一各epoch的batch数量
        epoch_size=range_size//config.batch_size
        #创建一个队列
        index_queue=tf.train.range_input_producer(range_size,num_epochs=None,
                                                 shuffle=True,seed=None,capacity=32)

        index_dequeue_op=index_queue.dequeue_many(config.batch_size*epoch_size,'index_dequeue')

        batch_size_placeholder=tf.placeholder(tf.int32,name='batch_size')
        phase_train_placeholder=tf.placeholder(tf.bool,name='phase_train')
        image_paths_placeholder=tf.placeholder(tf.string,shape=(None,1),name='image_paths')
        labels_placeholder=tf.placeholder(tf.int32,shape=(None,1),name='label')
        keep_probability_placeholder=tf.placeholder(tf.float32,name='keep_probability')

        nrof_preprocess_threads=4
        #输入队列
        input_queue=tf.FIFOQueue(capacity=2000000,
                                 dtypes=[tf.string,tf.int32],
                                 shapes=[(1,),(1,)],
                                 shared_name=None,name=None)
        enqueue_op=input_queue.enqueue_many([image_paths_placeholder,labels_placeholder],
                                           name='enqueue_op')
        #获取图像，label的batch形式
        image_batch,label_batch=create_input_pipeline(input_queue,
                                                     image_size,
                                                     nrof_preprocess_threads,
                                                     batch_size_placeholder)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        #网络输出
        prelogits,_=network.inference(image_batch,
                                      keep_probability_placeholder,
                                      phase_train=phase_train_placeholder,
                                      bottleneck_layer_size=config.embedding_size,
                                      weight_decay=config.weight_decay)
        #用于计算loss
        logits=slim.fully_connected(prelogits,len(train_set),activation_fn=None,
                                   weights_initializer=slim.initializers.xavier_initializer(),
                                   weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                   scope='Logits', reuse=False)
        #正则化的embeddings主要用于测试，对比两张图片差异
        embeddings=tf.nn.l2_normalize(prelogits,1,1e-10,name='embeddings')
        #计算centerloss
        prelogits_center_loss,_=center_loss(prelogits,label_batch,config.center_loss_alfa,nrof_classes)
        tf.identity(prelogits_center_loss,name='center_loss')
        tf.summary.scalar('center_loss', prelogits_center_loss)
        #学习率
        boundaries = [int(epoch * range_size / config.batch_size) for epoch in config.LR_EPOCH]
        lr_values = [config.learning_rate * (0.1 ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, lr_values)
        tf.identity(learning_rate,name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        #交叉熵损失
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                     logits=logits,
                                                                     name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.identity(cross_entropy_mean,name='cross_entropy_mean')
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
        #l2正则loss
        L2_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #总的loss
        total_loss=cross_entropy_mean+config.center_loss_factor*prelogits_center_loss+L2_loss
        tf.identity( total_loss,name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

        #准确率
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.identity(accuracy,name='accuracy')
        tf.summary.scalar('accuracy',accuracy)

        train_op=optimize(total_loss, global_step,
                        learning_rate,
                        config.moving_average_decay,
                        tf.global_variables())
        saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=3)
        summary_op=tf.summary.merge_all()
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        #训练和验证的graph保存地址
        train_writer=tf.summary.FileWriter(config.graph_dir+'train/',sess.graph)
        val_writer = tf.summary.FileWriter(config.graph_dir+'val/', sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if os.path.exists(config.model_dir):
                model_file=tf.train.latest_checkpoint(config.model_dir)
                if model_file:
                    saver.restore(sess,model_file)
                    print('重载模型训练')

            if not os.path.exists(config.model_dir):
                os.mkdir(config.model_dir)
            for epoch in range(1,config.max_nrof_epochs+1):
                step=sess.run(global_step,feed_dict=None)
                #训练
                batch_number = 0
                #获取image和label
                index_epoch=sess.run(index_dequeue_op)
                label_epoch=np.array(label_list)[index_epoch]
                image_epoch=np.array(image_list)[index_epoch]

                labels_array = np.expand_dims(np.array(label_epoch),1)
                image_paths_array = np.expand_dims(np.array(image_epoch),1)
                #运行输入队列
                sess.run(enqueue_op,{image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
                while batch_number<epoch_size:

                    feed_dict = {phase_train_placeholder:True, batch_size_placeholder:config.batch_size,keep_probability_placeholder:config.keep_probability}
                    tensor_list = [total_loss, train_op, global_step,learning_rate, prelogits,
                                   cross_entropy_mean, accuracy, prelogits_center_loss]
                    #每经过100个batch更新一次graph
                    if batch_number % 100 == 0:

                        loss_, _, step_, lr_,prelogits_, cross_entropy_mean_, accuracy_, center_loss_, summary_str = sess.run(tensor_list + [summary_op],  feed_dict=feed_dict)
                        train_writer.add_summary(summary_str, global_step=step_)
                        saver.save(sess=sess, save_path=config.model_dir+'model.ckpt',global_step=(step_))
                        print('epoch:%d/%d'%(epoch,config.max_nrof_epochs))
                        print("Step: %d/%d, accuracy: %3f, center loss: %4f, cross loss: %4f, Total Loss: %4f ,lr:%f " % (step_,epoch_size*config.max_nrof_epochs, accuracy_, center_loss_, cross_entropy_mean_,loss_, lr_))
                    else:
                        loss_, _, step_, lr_, prelogits_, cross_entropy_mean_, accuracy_, center_loss_,  = sess.run(tensor_list,  feed_dict=feed_dict)
                    batch_number+=1
                train_writer.add_summary(summary_str, global_step=step_)
                #验证
                nrof_val_batches=len(val_label_list)//config.batch_size
                nrof_val_images=nrof_val_batches*config.batch_size

                labels_val_array=np.expand_dims(np.array(val_label_list[:nrof_val_images]),1)
                image_paths_val_array=np.expand_dims(np.array(val_image_list[:nrof_val_images]),1)
                #运行输入队列
                sess.run(enqueue_op, {image_paths_placeholder: image_paths_val_array, labels_placeholder: labels_val_array})
                loss_val_mean=0
                center_loss_val_mean=0
                cross_entropy_mean_val_mean=0
                accuracy_val_mean=0
                for i in range(nrof_val_batches):
                    feed_dict = {phase_train_placeholder:False, batch_size_placeholder:config.batch_size,keep_probability_placeholder:1.0}
                    loss_val,center_loss_val,cross_entropy_mean_val,accuracy_val,summary_val=sess.run ([total_loss,prelogits_center_loss,cross_entropy_mean, accuracy,summary_op], feed_dict=feed_dict)
                    loss_val_mean+=loss_val
                    center_loss_val_mean+=center_loss_val
                    cross_entropy_mean_val_mean+=cross_entropy_mean_val
                    accuracy_val_mean+=accuracy_val
                    if i % 10 == 9:
                        print('.', end='')
                        sys.stdout.flush()
                val_writer.add_summary(summary_val, global_step=epoch)
                loss_val_mean/=nrof_val_batches
                center_loss_val_mean/=nrof_val_batches
                cross_entropy_mean_val_mean/=nrof_val_batches
                accuracy_val_mean/=nrof_val_batches
                print('到这了！')
                print("val: accuracy: %3f, center loss: %4f, cross loss: %4f, Total Loss: %4f " % ( accuracy_val_mean, center_loss_val_mean, cross_entropy_mean_val_mean,loss_val_mean))



# In[2]:


def center_loss(features,label,alfa,nrof_classes):
    '''计算centerloss
    参数：
      features:网络最终输出[batch,512]
      label:对应类别[batch,1]
      alfa:center更新比例
      nrof_classes:类别总数
    返回值：
      loss:center_loss损失值
      centers:中心点embeddings
    '''
    #embedding的维度
    nrof_features=features.get_shape()[1]
    centers=tf.get_variable('centers',[nrof_classes,nrof_features],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0),
                            trainable=False)
    label=tf.reshape(label,[-1])
    #挑选出每个batch对应的centers [batch,nrof_features]
    centers_batch=tf.gather(centers,label)
    diff=(1-alfa)*(centers_batch-features)
    #相同类别会累计相减
    centers=tf.scatter_sub(centers,label,diff)
    #先更新完centers在计算loss
    with tf.control_dependencies([centers]):
        loss=tf.reduce_mean(tf.square(features-centers_batch))
    return loss,centers


# In[3]:


def optimize(total_loss, global_step, learning_rate, moving_average_decay, update_gradient_vars):
    '''优化参数
    参数：
      total_loss:总损失函数
      global_step：全局step数
      learning_rate:学习率
      moving_average_decay：指数平均参数
      update_gradient_vars：需更新的参数
    返回值：
      train_op
    '''

    opt=tf.train.AdamOptimizer(learning_rate ,beta1=0.9, beta2=0.999, epsilon=0.1)
    #梯度计算
    grads=opt.compute_gradients(total_loss,update_gradient_vars)
    #应用更新的梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    #参数和梯度分布图
    for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
    #指数平均
    variable_averages=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


# In[4]:


if __name__ == '__main__':
    main()
