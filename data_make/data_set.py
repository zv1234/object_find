#encoding:utf-8

'''
常用数据集voc2007数据
ananotations  图片的标注信息xml格式
jpeg 所有图片
'''

# tfrecord格式将图像数据和各种标签放在一起的二进制文件文件格式tfrecord 或者tfreocrds
'''
文件格式：.tfrecord or .tfrecords
写入文件内容：使用Example将数据封装成protobuffer协议格式
体积小：消息大小只需要xml的1/10~1/3
解析速度快：解析速度比xml快20~100倍
每个example对应一张图片，其中包括图片的各种信息
'''
import tensorflow as tf
import  os


DIRECTORY_ANNOTATIONS = "Annotations/"

# 每个TFRecords文件的example个数
SAMPLES_PER_FILES = 200

def _get_output_filename(output_dir, name, number):
    return "%s/%s_%03d.tfrecord" % (output_dir, name, number)

def run(dataset_dir,output_dir,name='data'):
    '''
    tfrecords 文件，每个文件固定样本数量
    :param dataset_dir: 数据集目录
    :param output_dir: 输出目录
    :param name: 数据集名字
    :return:
    '''
    #判断数据集路径是否存在，不存在则新建
    if tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    #获取annatations下所有文件名字列表
    path=os.path.join(dataset_dir,DIRECTORY_ANNOTATIONS)
    #排序
    filenames=sorted(os.listdir(path))
    # 3、循环列表中的每个文件
    i=0
    fidx=0
    while i<len(filenames):
        # 新建一个tfrecords文件
        # 构造一个文件名字
        """
        建立TFRecord存储器
        tf.python_io.TFRecordWriter(path)
        写入tfrecords文件
        path: TFRecords文件的路径
        return：写文件
        method
        write(record):向文件中写入一个example
        close():关闭文件写入器
        """
        tf_filename=_get_output_filename(output_dir,name,fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j=0
            #处理200个文件和xml
            while i<len(filenames) and j<SAMPLES_PER_FILES:
                print("转换图片进度 %d/%d" % (i + 1, len(filenames)))
                # 处理图片，读取的此操作
                # 处理每张图片的逻辑
                # 1、读取图片内容以及图片相对应的XML文件
                # 2、读取的内容封装成example, 写入指定tfrecord文件
                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)

                i += 1
                j += 1

                # 当前TFRecords处理结束
            fidx += 1

            print("完成数据集 %s 所有的样本处理" % name)

def _add_to_tfrecord(dataset_dir,img_name,tfrecord_writer):
    pass