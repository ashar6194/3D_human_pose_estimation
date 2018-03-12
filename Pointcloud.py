from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import math as math
import cv2 as cv


def find_rot_matrix(radx, rady, radz):
    rotx = np.matrix([[1, 0, 0], [np.cos(radx), -np.sin(radx), 0], [np.sin(radx), np.cos(radx), 0]])
    roty = np.matrix([[np.cos(rady), 0, -np.sin(rady)], [0, 1, 0], [np.sin(rady), 0, np.cos(rady)]])
    rotz = np.matrix([[np.cos(radz), -np.sin(radz), 0], [np.sin(radz), np.cos(radz), 0], [0, 0, 1]])
    matrix = rotx * roty * rotz
    return matrix


def find_SE3(matrix, translation):
    homogeneous = np.append(matrix, [[0, 0, 0]], axis=0)
    final = np.append(translation, [[1]])
    final = np.reshape(final, (4, 1))
    SE3 = np.append(homogeneous, final, axis=1)
    return SE3

data = loadmat('/home/mcao/Documents/deconvnets/input/UBC_easy/train/1/groundtruth.mat')
fig = plt.figure()
plt.ion()

for i in range(0, 1):
    Wc = []
    for j in range(1, 4):
        print 'Picture %d, Camera %d' % (i, j)
        #Translation for Camera
        trans_for_cam2 = data['cameras'][0, 0]['Cam%d' % j][0, 0]['frames'][0, i]['translate'][0, 0]
        #Rotation for Camera
        rot_for_cam2 = data['cameras'][0, 0]['Cam%d' % j][0, 0]['frames'][0, i]['rotate'][0, 0]
        print trans_for_cam2
        matrix = find_rot_matrix(rot_for_cam2[0,0], rot_for_cam2[0,1], rot_for_cam2[0,2])
        SE3 = find_SE3(matrix, trans_for_cam2)
        image = cv.imread('input/UBC_easy/train/1/images/depthRender/Cam%d/mayaProject.%06d.png' % (j, (i + 1)))
        asd = np.array(image)
        print image.shape
        Img = asd.mean(axis=2)
        Img = (np.double(Img)/255) * (800-50) + 50
        x, y = np.meshgrid(np.arange(Img.shape[1]), np.arange(Img.shape[0]))
        I = Img.reshape(Img.shape[0] * Img.shape[1])
        X = x.reshape(Img.shape[0] * Img.shape[1])
        Y = y.reshape(Img.shape[0] * Img.shape[1])
        args = (X, Y, I)

        #Intrinsic Parameters of
        K= [[525.0, 0.0, 319.5],
            [0.0, 525.0, 239.5],
            [0.0, 0.0, 1.0]]

        Kinv = np.linalg.inv(K)
        p = np.column_stack(args)
        p = p[p[:, 2] > 51]
        p2 = np.transpose(p)
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(p2[1, :], p2[0, :], p2[2, :], c='b')
        plt.pause(0.5)
        pmod = (p2[0, :],  p2[1, :], np.ones(p2.shape[1]))
        pmod = np.transpose(np.column_stack(pmod))
        p33 = p2[2, :]
        Xc = np.matmul(Kinv, pmod) * p33

        #Make it Homogenous for SE3 Multiplication
        HomV = np.append(Xc, np.ones(p2.shape[1])).reshape(4, Xc.shape[1])
        #print HomV[:, 0:5]
        Wcf = SE3 * HomV
        if j != 1:
            argf = (Wc, Wcf)
            Wc = np.concatenate((Wc, Wcf), 1)
            print Wc.shape
        else:
            Wc = Wcf
            print Wc.shape

    #fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([Wc[0, :]], [Wc[1, :]],  [Wc[2, :]], c='r')
    plt.grid()
    plt.pause(0.5)

while True:
     plt.pause(0.5)