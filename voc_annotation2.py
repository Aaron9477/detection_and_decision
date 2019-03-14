import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

image_ids1 = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%('2007', 'train')).read().strip().split()
image_ids2 = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%('2007', 'val')).read().strip().split()
image_ids = image_ids1 + image_ids2
image_ids.sort()

list_file = open('%s_%s.txt'%('2007', 'train'), 'w')
for image_id in image_ids:
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, '2007', image_id))
    convert_annotation('2007', image_id, list_file)
    list_file.write('\n')
list_file.close()

image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%('2007', 'test')).read().strip().split()
list_file = open('%s_%s.txt'%('2007', 'test'), 'w')
for image_id in image_ids:
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, '2007', image_id))
    convert_annotation('2007', image_id, list_file)
    list_file.write('\n')
list_file.close()