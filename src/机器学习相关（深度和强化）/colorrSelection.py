import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#读取图片
#img = Image.open('test.png')
image = mpimg.imread('test.png','rb') #必须要加‘rb’，不加错误。要按二进制读取
print(type(image),image.shape)#image.shape是显示出图片的长 宽和三色素（rgb）

ysize = image.shape[0]  #数组
xsize = image.shape[1]
color_select = np.copy(image)

#定义阈值
red_threshold = 200;
green_threshold = 200;
blue_threshold = 200;
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:,:,0]<rgb_threshold[0]) \
        | (image[:,:,1]<rgb_threshold[1])\
        | (image[:,:,2]<rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image
plt.imshow(color_select)
plt.show()




#img.show()
