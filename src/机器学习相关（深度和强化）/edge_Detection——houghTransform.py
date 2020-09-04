import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#读取图像
image = mpimg.imread( 'exit-ramp.png','rb')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

#定义核的数量和应用高斯平滑
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

#定义canny参数，调用canny函数
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray,low_threshold,high_threshold)

#定义hough变换的参数
#制作一个我们要绘制图像大小的空白
rho = 1
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20
line_image = np.copy(image)*0

#在边沿探测图片上运行hough
lines = cv2.HoughLinesP(edges,rho,theta,threshold,np.array([]),min_line_length,max_line_gap)

#遍历输出的直线，并在空白地方画线
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

#创建一个二值化图像来和line_image结合
color_edges = np.dstack((edges,edges,edges))

#在edges图像上画线
combo = cv2.addWeighted(color_edges,0.8,line_image,1,0)
plt.imshow(combo)
plt.show()