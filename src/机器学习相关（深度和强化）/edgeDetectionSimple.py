import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
#读取图像
image = mpimg.imread('exit-ramp.png','rb')

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray,low_threshold,high_threshold)
plt.imshow(blur_gray)
plt.imshow(edges,cmap='Greys_r')
plt.show()








