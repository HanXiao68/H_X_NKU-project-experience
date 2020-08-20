# H_X-NKU-project-experience
### 本项目主要研究多模态感知融合系统中核心的一些功能，包括图像处理和传感器融合。
### 这个项目记录了笔者在南开大学人工智能学院机器人与信息自动化研究所的经历和个人项目。
### 主要是研究多传感器融合的图像处理和基础平台开发搭建。
### 在期间，参加了与301医院（中国人民解放军总医院），哈工大，浙大，南开AI学院一同合作的人工耳蜗手术项目。我们主要负责虚拟仿真手术的相关方面研究。使用多模态感知融合系统的近景系统做实验。实验内容包括：
#### 1，对人头的耳蜗进行七步解剖，使用近景系统完成realsense数据采集，近景双目相机的数据采集； 
#### 2.进行三维重建。生成动画过程以用于术前讨论和指导，以及实验的教学工作。
#### 3.使用VIO技术进行三维重建的优化，拓展研究SLAM技术和VINS系统。

### 在前期也做了一些基于stm32板子开发全地形车的项目。在windows系统下使用 c/c++开发
    
<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E5%A4%9A%E4%BC%A0%E6%84%9F%E5%99%A8%E4%BB%BF%E7%9C%9F.jpg" width="575"/>

<img src="https://github.com/HanXiao68/H_X-NKU-project-experience/tree/master/image/301.gif" width="575"/>

<img src="https://github.com/HanXiao68/H_X-NKU-project-experience/blob/master/image/%E5%A4%9A.JPG" width="575"/>

<img src="https://github.com/HanXiao68/H_X-NKU-project-experience/blob/master/image/zuobiaoxi-topological.jpg" width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/301_%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA%E4%BB%BF%E7%9C%9F%E5%AE%9E%E9%AA%8C.jpg" width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E5%A4%9A%E4%BC%A0%E6%84%9F%E5%99%A8_V!.0.jpg" width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E5%A4%9A%E4%BC%A0%E6%84%9F%E5%99%A8_V2.1.jpg" width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E5%85%A8%E5%9C%B0%E5%BD%A2%E8%BD%A6%E5%BC%80%E5%8F%91/0e39b69c1619af03134245dce767d3b.jpg" width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E5%85%A8%E5%9C%B0%E5%BD%A2%E8%BD%A6%E5%BC%80%E5%8F%91/185fefa8a50831173715008e0930d85.jpg" width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E5%85%A8%E5%9C%B0%E5%BD%A2%E8%BD%A6%E5%BC%80%E5%8F%91/241e07d2f2a0df900744736abb00dc0.jpg" width="275"/>

<img src=" https://github.com/HanXiao68/upstream/blob/master/image/%E5%85%A8%E5%9C%B0%E5%BD%A2%E8%BD%A6%E5%BC%80%E5%8F%91/%E8%B0%83%E8%AF%95apollo.jpg" width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E7%AC%AC%E4%B8%89%E5%B1%8A%E6%99%BA%E8%83%BD%E9%A9%BE%E9%A9%B6%E6%8C%91%E6%88%98%E8%B5%9B/02194416e1ccf723d9ec8cdaaf9050d.jpg " width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E7%AC%AC%E4%B8%89%E5%B1%8A%E6%99%BA%E8%83%BD%E9%A9%BE%E9%A9%B6%E6%8C%91%E6%88%98%E8%B5%9B/119bf52fb407373bba67e7d9a15d8d4.jpg " width="575"/>

<img src="https://github.com/HanXiao68/upstream/blob/master/image/%E7%AC%AC%E4%B8%89%E5%B1%8A%E6%99%BA%E8%83%BD%E9%A9%BE%E9%A9%B6%E6%8C%91%E6%88%98%E8%B5%9B/a2feb732d0c3ffd02b4466697bef570.jpg " width="575"/>



## 客户端:
### 1\建立代码
### 1.1初始化ros结点
### 1.2创建client实例
### 1.3发布服务请求数据
### 1.4等待server处理之后的结果
### 2设置编译规则
### 3catkin_make执行

## 服务端:
### 1代码实现
### 1.1\初始化ros结点
### 1.2\创建server实例
### 1.3\循环等待服务请求,进入回调函数
### 1.4\处理回调函数,返回应答数据
### 2设置编译规则
### 3运行

