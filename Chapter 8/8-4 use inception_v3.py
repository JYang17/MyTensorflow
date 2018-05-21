# https://www.bilibili.com/video/av20034351/?p=26
import tensorflow as tf 
import os  
import numpy as np 
import re 
#from PIL import Image #https://stackoverflow.com/questions/8863917/importerror-no-module-named-pil pip install Pillow
from PIL import Image
#import matplotlib.pylot as plt
import matplotlib.pyplot as plt

class NodeLookup(object):
    def __init__(self):
        #python单引号和双引号区别 https://blog.csdn.net/woainishifu/article/details/76105667
        #imagenet_2012_challenge_label_map_proto.pbtxt是一个字典，key是1-1000整数，value是n开头后面都是数字的字符串
        label_lookup_path = "inception_pretrain\imagenet_2012_challenge_label_map_proto.pbtxt"
        #imagenet_synset_to_human_label_map.txt也是一个字典，key是n开头后面都是数字的字符串，value是一个或者多个英文单词
        uid_lookup_path = "inception_pretrain\imagenet_synset_to_human_label_map.txt"
        self.node_lookup = self.load(label_lookup_path,uid_lookup_path)

    def load(self,label_lookup_path,uid_lookup_path):
        #加载分类字符串n********对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        for line in proto_as_ascii_lines:
            #去掉换行符
            line = line.strip("\n")
            #按照\t进行分割
            parsed_items = line.split("\t")
            #获取分类编号
            uid = parsed_items[0]
            #获取分类名称
            human_string = parsed_items[1]
            #添加uid和human_string的映射关系到字典
            uid_to_human[uid] = human_string
        
        #加载分类字符串n********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith("  target_class:"):
                #获取分类编号1-1000
                target_class = int(line.split(":")[1])
            if line.startswith("  target_class_string:"):
                #获取n*******
                target_class_string = line.split(": ")[1]#注意这里冒号后需要有个空格，否则[1:-2]去掉了左边的空格，而左边的引号留下了
                node_id_to_uid[target_class] = target_class_string[1:-2]#去掉引号，从第一个字符取到倒数第二个字符
        
        #建立分类编号1-1000到分类名称的字典
        node_id_to_name = {}
        for key,val in node_id_to_uid.items():
            #获取分类名称
            name = uid_to_human[val]
            node_id_to_name[key] = name 

        return node_id_to_name
    
    def id_to_string(self,node_id):
        if node_id not in self.node_lookup:
            return ""
        return self.node_lookup[node_id]

#创建一个图来存放Google训练好的模型
with tf.gfile.FastGFile("inception_pretrain/classify_image_graph_def.pb","rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name="")

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name("softmax:0")
    #遍历目录
    #在inception_pretrain文件夹同一级目录，加文件夹images，在images里放jpg图片
    for root,dirs,files in os.walk("images/"):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file),"rb").read()
            predictions = sess.run(softmax_tensor,{"DecodeJpeg/contents:0":image_data})#图片格式是jpeg，jpeg就是jpg
            predictions = np.squeeze(predictions)#把结果转成1维数据

            #打印图片路径和名称
            image_path = os.path.join(root,file)
            print(image_path)
            #显示图片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()

            #排序：最大的五个值从大到小排序
            #argsort()是从小到大排序
            #[-5:]代表取倒数5个值
            #[::-1]取倒序
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = NodeLookup()
            for node_id in top_k:
                #获取分类和名称
                human_string = node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print("%s(score=%.5f)"%(human_string,score))
            print()


