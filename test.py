import keras
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os

def rand(a=0, b=1):
    # np.random.rand()返回一个或一组服从“0~1”均匀分布的随机样本值
    return np.random.rand()*(b-a) + a

# test for data augmentation
def get_random_data(image_dir, target_dir, image_name, input_shape=(320,320), max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    for i in range(50):
        image = Image.open(image_dir)
        # image.show()
        iw, ih = image.size
        h, w = input_shape

        # # 读取基于原图大小的box
        # # 随机变化图像大小
        # # resize image
        # # w/h比例随机变化
        # new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        # # 以input为基础，输入框的尺度随机变化
        # scale = rand(.25, 2)
        # # 注意，下式首先变化长或宽，之后再以此为基础，再以新的w/h得到另一边的长度
        # if new_ar < 1:
        #     nh = int(scale*h)
        #     nw = int(nh*new_ar)
        # else:
        #     nw = int(scale*w)
        #     nh = int(nw/new_ar)
        # # 将图像resize到新的大小
        # print('nw:', nw, ' nh:', nh)
        # image = image.resize((nw,nh), Image.BICUBIC)
        # # image.show()
        #
        # # place image
        # # 随机寻找一点，作为图像crop的起点，保证crop的图像不会越界
        # dx = int(rand(0, w-nw))
        # dy = int(rand(0, h-nh))
        # print('dx:', dx, ' dy:', dy)
        # # Image.new最后一个参数是对三个通道进行赋值，作为初始化
        # new_image = Image.new('RGB', (w,h), (128,128,128))
        # # 只截取图像的一部分,和上面代码结合相当于随机缩放图像，(dx, dy)定义左上点
        # new_image.paste(image, (dx, dy))
        # image = new_image
        # # image.show()
        #
        # image.save(os.path.join(target_dir, 'scale_' + str(i) + '_' + image_name))


        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        # 这里归一化到[0,1]!!!!!!!!!!!!!!!!!!!!!
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1 ### 还有这种操作！！！！！！！！！！！！！！！
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1  ### 还有这种操作！！！！！！！！！！！！！！！
        x[x<0] = 0
        image_data = hsv_to_rgb(x)
        image_data *= 255.
        image = Image.fromarray(image_data.astype('uint8')).convert('RGB')
        image.save(os.path.join(target_dir, 'hsv_' + str(i) + '_' + image_name))
        #
        # image.show()


if __name__ == '__main__':
    image_dir = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/VOCdevkit/VOC2007/JPEGImages'
    target_dir = '/home/zq610/WYZ/graduation_project/data_augmentation'
    image_name = '000300.jpg'
    get_random_data(os.path.join(image_dir, image_name), target_dir, image_name)


# # test for inherit
# class A():
#     def fun1(self):
#         return self.fun2()
#
# class B(A):
#     def fun2(self):
#         return "AAAA"
#
# # right
# test = B()
# print(test.fun1())
# # wrong
# test2 = A()
# print(test2.fun1())






# test for yield
# def fab(max):
#     n, a, b = 0 ,0, 1
#     while n < max:
#         yield b
#         a, b = b ,a+b
#         n += 1

# right
# for i in fab(10):
#     print(i)

# wrong
# for i in range(10):
#     print(fab(10))

# a = fab(3)
# for i in range(5):
#     print(a.__next__())
#
# for i in range(5):
#     print(fab(5).__next__())
