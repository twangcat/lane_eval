import numpy as np
from transformations import euler_matrix
from transformations import translation_matrix
import pickle
# defines a set of perspective transforms that correspond to shifts/rotations in the real world

def setPersp(scale_factor = 1, distort_location = '/afs/cs.stanford.edu/u/twangcat/scratch/sail-car-log/process/'):    
    distort = { }
    distort['M_1'] = [np.eye(3)] # perspective distrotion in cam1 image space
    distort['M_2'] = [np.eye(3)] # perspective distrotion in cam2 image space
    distort['T'] = [np.eye(4)] # corresponding real-world transformation


    distort['T'].insert(0,translation_matrix(np.array([1.0,0,0])))
    distort['T'].insert(0,translation_matrix(np.array([-1.0,0,0])))
    distort['T'].insert(0,euler_matrix(0, 0.06, 0))
    distort['T'].insert(0,euler_matrix(0, -0.06, 0))
    distort['T'].insert(0,euler_matrix(0, 0.12, 0))
    distort['T'].insert(0,euler_matrix(0, -0.12, 0))


    distort['M_1'].insert(0,pickle.load(open(distort_location+'M_shift1.0_cam1.pickle'))) # shift 1.0
    distort['M_1'].insert(0,pickle.load(open(distort_location+'M_shift-1.0_cam1.pickle'))) # shift -1.0
    distort['M_1'].insert(0,pickle.load(open(distort_location+'M_rot0.06_cam1.pickle','r'))) # rotate 0.06
    distort['M_1'].insert(0,pickle.load(open(distort_location+'M_rot-0.06_cam1.pickle','r'))) # rotate -0.06
    distort['M_1'].insert(0,pickle.load(open(distort_location+'M_rot0.12_cam1.pickle','r'))) # rotate 0.12
    distort['M_1'].insert(0,pickle.load(open(distort_location+'M_rot-0.12_cam1.pickle','r'))) # rotate -0.12




    distort['M_2'].insert(0,pickle.load(open(distort_location+'M_shift1.0_cam2.pickle'))) # shift 1.0
    distort['M_2'].insert(0,pickle.load(open(distort_location+'M_shift-1.0_cam2.pickle'))) # shift -1.0
    distort['M_2'].insert(0,pickle.load(open(distort_location+'M_rot0.06_cam2.pickle','r'))) # rotate 0.06
    distort['M_2'].insert(0,pickle.load(open(distort_location+'M_rot-0.06_cam2.pickle','r'))) # rotate -0.06
    distort['M_2'].insert(0,pickle.load(open(distort_location+'M_rot0.12_cam2.pickle','r'))) # rotate 0.12
    distort['M_2'].insert(0,pickle.load(open(distort_location+'M_rot-0.12_cam2.pickle','r'))) # rotate -0.12

    assert len(distort['M_1']) == len(distort['M_2']) and len(distort['M_1'])==len(distort['T'])
    if scale_factor != 1:
      for i in xrange(len(distort['M_1'])-1): # last one is always identity, no need change.
        distort['M_1'][i][:,0:2]/=scale_factor
        distort['M_1'][i][2,:]/=scale_factor
        distort['M_2'][i][:,0:2]/=scale_factor
        distort['M_2'][i][2,:]/=scale_factor


    return distort

