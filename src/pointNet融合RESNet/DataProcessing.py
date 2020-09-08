import numpy as np
import sys
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input, Add, concatenate
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
from keras.models import load_model
import h5py
from matplotlib.pyplot import imshow
import cv2
import glob
import math
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

#loading files and paths
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(BASE_DIR)

def load_pc_from_bin(bin_path):
    """Load a bin file to a np array"""
    num_select = 2048
    selected_points =[]
    obj = np.fromfile(bin_path, dtype = np.float32).reshape(-1,4)
    pc = filter_camera_angle(obj)
    index = np.random.choice(pc.shape[0], num_select, replace=False)
    for i in range(len(index)):
        selected_points.append(pc[index[i]][0:3])
    selected_points = np.array(selected_points).reshape(1,-1,3)# return N*3 array
    return selected_points



def read_calib_file(calib_path):
    """Read a calibration file."""
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def proj_to_velo(calib_data):
    """Projection matrix to 3D axis for 3D Label"""
    rect = calib_data["R0_rect"].reshape(3, 3)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    inv_rect = np.linalg.inv(rect)
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)


def read_label_from_txt(label_path):
    """Read label from txt file."""
    text = np.fromfile(label_path)
    bounding_box = []
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            if not label:
                continue
            label = label.split(" ")
            if (label[0] == "DontCare"):
                continue
            y_class = [int(label[0]=="Car"), int(label[0]=="Van"), int(label[0]=="Pedestrian")]
            if label[0] == ("Car") or label[0]=="Van" or label[0]=="Pedestrian": #  or "Truck"
                y_labels = list(np.hstack((label[8:15], y_class)))
                bounding_box.append(y_labels)
                break

    if bounding_box:
        data = np.array(bounding_box, dtype=np.float32)
        return data[:, 3:6], data[:, :3], data[:, 6], data[:,7:] 
    else:
        return None, None, None, None

def read_labels(label_path, label_type, calib_path=None, is_velo_cam=False, proj_velo=None):
    """Read labels from txt file.
    Original Label value is shifted about 0.27m from object center.
    So need to revise the position of objects.
    """
    if label_type == "txt": #TODO
        places, size, rotates, y_class = read_label_from_txt(label_path)
        if places is None:
            return None, None, None, None
        rotates = np.pi / 2 - rotates
        dummy = np.zeros_like(places)
        dummy = places.copy()
        if calib_path:
            places = np.dot(dummy, proj_velo.transpose())[:, :3]
        else:
            places = dummy
        if is_velo_cam:
            places[:, 0] += 0.27

    data_combined = []
    for p, r, s , cl in zip(places, rotates, size, y_class):
        ps = np.hstack((cl[:],p[:],s[:]))
        data_combined.append(list(np.append(ps, r)))
        
    return places, rotates, size, y_class

def get_boxcorners(places, rotates, size):
    """Create 8 corners of bounding box from bottom center."""
    corners = []
    for place, rotate, sz in zip(places, rotates, size):
        x, y, z = place
        h, w, l = sz
        if l > 10:
            continue

        corner = np.array([
            [x - l / 2., y - w / 2., z],
            [x + l / 2., y - w / 2., z],
            [x - l / 2., y + w / 2., z],
            [x - l / 2., y - w / 2., z + h],
            [x - l / 2., y + w / 2., z + h],
            [x + l / 2., y + w / 2., z],
            [x + l / 2., y - w / 2., z + h],
            [x + l / 2., y + w / 2., z + h],
        ])

        corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), -np.sin(rotate), 0],
            [np.sin(rotate), np.cos(rotate), 0],
            [0, 0, 1]
        ])

        a = np.dot(corner, rotate_matrix.transpose())
        a += np.array([x, y, z])
        corners.append(a)
    return np.array(corners)

def filter_camera_angle(places):
    """Filter camera angles for KiTTI Datasets"""
    bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    # bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[bool_in]

def center_to_sphere(places, size, resolution=0.50, min_value=np.array([0., -50., -4.5]), scale=4, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert object label to Training label for objectness loss"""
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] < z[1]), (places[:, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2.
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)
    return sphere_center

def sphere_to_center(p_sphere, resolution=0.5, scale=4, min_value=np.array([0., -50., -4.5])):
    """from sphere center to label center"""
    center = p_sphere * (resolution*scale) + min_value
    return center

#derived from yukitsuji/3D_CNN_tensorflow
def process(velodyne_path, label_path=None, calib_path=None, dataformat="bin", label_type="txt", is_velo_cam=False):
    p = []
    pc = None
    bounding_boxes = None
    places = None
    rotates = None
    size = None
    proj_velo = None
    
    filenames_velo = [d for d in sorted(os.listdir(velodyne_path)) if d[0]!='.']
    train_points = None
    train_labels = None
    train_classes = None
    for d in filenames_velo:
        value = d[0:6]
        print(value)
        velo_path = velodyne_path + value + '.bin'
        cal_path = calib_path + value + '.txt'
        lab_path = label_path + value + '.txt'
        cur_points = load_pc_from_bin(velo_path)
        cur_calib = read_calib_file(cal_path)
        proj_velo = proj_to_velo(cur_calib)[:, :3]
        places, rotates, size, classes= read_labels(lab_path, label_type, cal_path , is_velo_cam=is_velo_cam, proj_velo=proj_velo)
        corners = get_boxcorners(places, rotates, size)
        if places is None:
            continue
        if train_points is None:
            train_labels = corners
            train_points = cur_points
            train_classes = classes
        else:
            train_points = np.concatenate((train_points, cur_points), axis = 0)
            train_labels = np.concatenate((train_labels, corners), axis = 0)
            train_classes = np.concatenate((train_classes, classes), axis =0)
            
    return train_points, train_labels, train_classes

train_points, train_labels, train_classes = process(pcd_path, label_path, calib_path=calib_path, dataformat="bin", is_velo_cam=True)

print(train_classes.shape)
np.save('train_classes.npy', train_classes)
np.save('train_points.npy', train_points)
np.save('train_labels.npy', train_labels)

print(train_labels.reshape((7481,24)).shape)


#run this cell only if images need to be processed again
filenames_imgs = [img_path+d for d in sorted(os.listdir(img_path)) if d[0]!='.']
batch_images = None
for d in filenames_imgs:
    print(d)
    original = load_img(d, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch_cur = np.expand_dims(numpy_image, axis=0)
    if batch_images is None:
        batch_images = image_batch_cur
    else:
        batch_images = np.concatenate((batch_images,image_batch_cur ), axis = 0)

from keras.applications import resnet50
resnet_model = resnet50.ResNet50(weights='imagenet')

#run this cell only if images need to be processed again by resnet. Otherwise, use the saved file in next cell
processed_images = np.load('processed_images.npy')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#extract final layer called flatten_1 with size 1 by 2048
layer_name = 'flatten_1'
layer = resnet_model.get_layer(layer_name)

#layer = resnet_model.layers[layer_num]
intermediate_layer_model = Model(inputs=resnet_model.input,
                                 outputs=layer.output)
intermediate_output = intermediate_layer_model.predict(processed_images)
print("Shape of Processed Images: ", processed_images.shape)
print("Layer used: ", layer.name)
print("Shape of layer output:", intermediate_output.shape)

np.save('intermediate_output.npy', intermediate_output)

intermediate_output = np.load('intermediate_output.npy')
print(intermediate_output.shape)





