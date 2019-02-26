import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.spatial.distance import euclidean
from utils import plot_basic_object
import time

def dist_ponto_cj(ponto, lista):
  return [euclidean(ponto, lista[j]) for j in range(len(lista))]


def ponto_mais_longe(lista_ds):
  ds_max = max(lista_ds)
  idx = lista_ds.index(ds_max)
  return pts[idx]


def op(pts, N, K):
    farthest_pts = [0] * K

    P0 = pts[np.random.randint(0, N)]
    farthest_pts[0] = P0
    ds0 = dist_ponto_cj(P0, pts)

    ds_tmp = ds0
    for i in range(1, K):
        farthest_pts[i] = ponto_mais_longe(ds_tmp)
        ds_tmp2 = dist_ponto_cj(farthest_pts[i], pts)
        ds_tmp = [min(ds_tmp[j], ds_tmp2[j]) for j in range(len(ds_tmp))]
        print ('P[%d]: %s' % (i, farthest_pts[i]))
    return farthest_pts


if __name__ == "__main__":
    pcl_list = sorted(glob.glob('/data/MHAD/train_pc/*.pkl'))
    gt_list = sorted(glob.glob('/data/MHAD/train_pcgt/*.pkl'))
    K = 2048

    for pcl in tqdm(pcl_list):
      pts = pickle.load(open(pcl, 'rb'))
      pts = pts.T
      N = pts.shape[0]

      farthest_pts = np.array(op(pts, N, K))

      a = 1


    # plt.figure(1)
    # a = 1
    # plt.scatter(pts[:, 0], pts[:, 1], color='r')
    # plt.scatter(farthest_pts[:, 0], farthest_pts[:, 1], color='b')
    # plt.show()
    #
