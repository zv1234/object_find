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
from .dataset_utils import int64_feature,float_feature,bytes_feature

DIRECTORY_ANNOTATIONS = "Annotations/"
DIRECTORY_IMAGES='Xml/'

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

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
def _process_image(dataset_dir, img_name):
    file_name=dataset_dir+DIRECTORY_IMAGES+img_name+'.'
    #读取图片
    image_data=tf.gfile.FastGFile(file_name,'rb').read()

    #处理xml
    file_name_xml=dataset_dir+DIRECTORY_ANNOTATIONS+img_name+'.'

    #et读取
    tree=ET.parse(file_name)
    root=tree.getroot()
    # 处理每一个标签
    # size:height, width, depth
    # 一张图片只有这三个属性
    size = root.find('size')

    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    # object:name, truncated, difficult, bndbox(xmin,ymin,xmax,ymax)
    # 定义每个属性的列表，装有不同对象
    # 一张图片会有多个对象findall
    bboxes = []
    difficult = []
    truncated = []
    # 装有所有对象名字
    # 对象的名字怎么存储？？？
    labels = []
    labels_text = []
    for obj in root.findall('object'):
        # name
        label = obj.find('name').text
        # 存进对象对应的物体类别数据
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        # difficult
        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)

        # truncated
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        # bndbox  [[12,23,34,45], [56,23,76,9]]
        bbox = obj.find('bndbox')

        # xmin,ymin,xmax,ymax都要进行除以原图片的长宽
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]))

    return image_data, shape, bboxes, difficult, truncated, labels, labels_text

def _convert_to_example(image_data, shape, bboxes, difficult, truncated, labels, labels_text):
    '''

    :param image_data:
    :param shape:
    :param bboxes:
    :param difficult:
    :param truncated:
    :param labels:
    :param labels_text:
    :return:
    '''
    # bboxes的格式转换
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    for b in bboxes:
        ymin.append(b[0])
        xmin.append(b[1])
        ymax.append(b[2])
        xmax.append(b[3])

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))

    return example




def _add_to_tfrecord(dataset_dir,img_name,tfrecord_writer):
    #读取图片内容以及图片对应都xml文件
    #将读取的内容封装为example 写入到tfrecord文件
    '''
    :param dataset_dir:
    :param img_name:
    :param tfrecord_writer:
    :return:
    '''
    image_data, shape, bboxes, difficult, truncated, labels, labels_text = \
        _process_image(dataset_dir, img_name)

    #读取内容封装为example
    example = _convert_to_example(image_data, shape, bboxes, difficult, truncated, labels, labels_text)


