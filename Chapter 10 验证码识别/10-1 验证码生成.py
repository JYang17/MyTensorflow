#验证码生成库
from captcha.image import ImageCaptcha #pip install captcha
import numpy as np 
from PIL import Image 
import random
import sys

number = ['0','1','2','3','4','5','6','7','8','9']

def random_captcha_text(char_set=number,captcha_size=4):
    #验证码列表
    captcha_text = []
    for i in range(captcha_size):
        #随机选择
        c = random.choice(char_set)
        #加入验证码列表
        captcha_text.append(c)
    return captcha_text

#生成字符串对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    #获取随机生成的验证码
    captcha_text = random_captcha_text()
    #把验证码列表转为字符串
    captcha_text = ''.join(captcha_text)
    #生成验证码
    captcha = image.generate(captcha_text)
    #要自己创建文件夹，不能自动创建文件夹
    image.write(captcha_text,'captcha/images/'+captcha_text+'.jpg')#写到文件

#最终图片数量会少于10000，因为随机生成时会有重复的
num = 10000
if __name__ =='__main__':
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>>Creating image %d%d'%(i+1,num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    print('生成验证码图片完毕')