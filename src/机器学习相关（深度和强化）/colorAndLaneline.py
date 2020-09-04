import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#读取图片
image = mpimg.imread('test.png','rb') #必须要加‘rb’，不加错误。要按二进制读取
print(type(image),image.shape)#image.shape是显示出图片的长 宽和三色素（rgb）

ysize = image.shape[0]  #数组
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

#定义阈值
red_threshold = 200;
green_threshold =200;
blue_threshold = 200;
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

#设置ROI的三角区域
left_bottom = [0,539]
right_bottom = [900,539]
apex = [475,320]

fit_left = np.polyfit((left_bottom[0],apex[0]),(left_bottom[1],apex[1]),1)
fit_right = np.polyfit((right_bottom[0],apex[0]),(right_bottom[1],apex[1]),1)
fit_bottom = np.polyfit((left_bottom[0],right_bottom[0]),(left_bottom[1],right_bottom[1]),1)

#把低于要求门槛的像素全找出来
color_thresholds = (image[:,:,0]<rgb_threshold[0]) \
        | (image[:,:,1]<rgb_threshold[1])\
        | (image[:,:,2]<rgb_threshold[2])

#找到车道线中的像素
XX,YY = np.meshgrid(np.arange(0,xsize),np.arange(0,ysize))
region_thresholds = (YY > (XX*fit_left[0]+fit_left[1])) & \
                    (YY > (XX*fit_right[0]+fit_right[1])) & \
                    (YY > (XX*fit_bottom[0]+fit_bottom[1]))
#把选中的这些像素全赋值为黑色
color_select[color_thresholds] = [0,0,0]
#找到图片中车道线的部分
line_image[~color_thresholds & region_thresholds] = [0,255,0]

# Display the image
plt.imshow(color_select)
plt.imshow(line_image)
plt.show()



#img.show()