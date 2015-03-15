from sklearn.cluster import DBSCAN
import numpy as np
import copy
__all__=['dbscanCluster','jlinkCluster']

def jaccard(im1, im2):
    """
    Computes the Jaccard metric, a measure of set similarity.
 
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
 
    Returns
    -------
    jaccard : float
        Jaccard metric returned is a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
    
    Notes
    -----
    The order of inputs for `jaccard` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
 
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    return intersection.sum() / float(union.sum())



def segmentDist(seg1, seg2):
  rW = 1
  xDistW = 36
  X = np.array([[seg1[0],1], [seg1[2],1], [seg2[0],1], [seg2[2],1],])
  y = np.array([seg1[1], seg1[3], seg2[1], seg2[3]])
  sol = np.linalg.lstsq(X,y)
  R = sol[1]
  if len(R)==0:
    R=0
  
  dist1 = (xDistW*(seg1[0]-seg2[0])**2+(seg1[1]-seg2[1])**2)**0.5
  dist2 = (xDistW*(seg1[2]-seg2[2])**2+(seg1[3]-seg2[3])**2)**0.5
  dist3 = (xDistW*(seg1[0]-seg2[2])**2+(seg1[1]-seg2[3])**2)**0.5
  dist4 = (xDistW*(seg1[2]-seg2[0])**2+(seg1[3]-seg2[1])**2)**0.5
  dist=min([dist1, dist2, dist3, dist4])
  dist+=R*rW
  return dist
def segmentDist3(seg1, seg2, eps=0.0001):
  angleW = 100#150
  xDistW = 64
  grad1 = (seg1[2]-seg1[0])/(seg1[3]-seg1[1]+eps)
  grad2 = (seg2[2]-seg2[0])/(seg2[3]-seg2[1]+eps)
  dist1 = ((seg1[0]-seg2[2])**2+(seg1[1]-seg2[3])**2)**0.5
  dist2 = ((seg1[2]-seg2[0])**2+(seg1[3]-seg2[1])**2)**0.5
  if dist1<dist2:
    dist=(xDistW*(seg1[0]-seg2[2])**2+(seg1[1]-seg2[3])**2)**0.5
    grad = (seg1[0]-seg2[2])/(seg1[1]-seg2[3]+eps)
  else:
    dist=(xDistW*(seg1[2]-seg2[0])**2+(seg1[3]-seg2[1])**2)**0.5
    grad = (seg2[0]-seg1[2])/(seg2[1]-seg1[3]+eps)
  dist+=abs(grad-(grad1+grad2)/2)*angleW
  return dist
def segmentDist2(seg1, seg2):
  xDistW = 64
  angleW = 30
  dist1 = (xDistW*(seg1[0]-seg2[0])**2+(seg1[1]-seg2[1])**2)**0.5
  dist2 = (xDistW*(seg1[2]-seg2[2])**2+(seg1[3]-seg2[3])**2)**0.5
  dist3 = (xDistW*(seg1[0]-seg2[2])**2+(seg1[1]-seg2[3])**2)**0.5
  dist4 = (xDistW*(seg1[2]-seg2[0])**2+(seg1[3]-seg2[1])**2)**0.5
  dist=min([dist1, dist2, dist3, dist4])
  dist+=abs((seg1[2]-seg1[0])/(seg1[3]-seg1[1])-(seg2[2]-seg2[0])/(seg2[3]-seg2[1]))*angleW
  return dist

def orderLabels(labels, segs, mode='topDown'):
  # reorder the lane ids so that 0 corresponds to the left lane boundary of ego-lane, 1 is right boundary of ego-lane, 2 is left lane boundary of left lane, etc
  labels2 = copy.deepcopy(labels)
  if mode=='topDown':
    selfX = 0
  elif mode=='2D':
    selfX =320
  else:
    assert False,'unrecognized mode! must be topDown or 2D'
  numLanes = np.max(labels)+1
  nearValues = np.empty(numLanes) # x coords of the 'nearest' pt of each lane
  for n in range(numLanes):
    if np.sum(labels==n)==0:
      nearValues[n]=float('nan')
      continue
    currLane = segs[labels==n,:]
    if mode=='topDown':
      # use linear interpolation to find intersection with z=0 plane.
      nearSeg = currLane[np.argmin(currLane[:, 1]), :]
      slope=(nearSeg[2]-nearSeg[0])/(nearSeg[3]-nearSeg[1])
      nearValues[n] = nearSeg[0]-slope*nearSeg[1]
      #nearValues[n] = currLane[np.argmin(currLane[:, 1]), 0]
    else:
      # fixme. Should use interpolation to intersect bottom of image.
      nearValues[n] = currLane[np.argmax(currLane[:, 1]), 0]-selfX
  lIds = np.where(nearValues<0)[0]
  rIds = np.where(nearValues>=0)[0]
  if len(lIds)>0:
    lXs = nearValues[lIds]
    newlIds = lIds[np.argsort(-lXs)]
    for l in range(len(lIds)):
      labels2[labels==newlIds[l]]=l*2
  if len(rIds)>0:
    rXs = nearValues[rIds]
    newrIds = rIds[np.argsort(rXs)]
    for r in range(len(rIds)):
      labels2[labels==newrIds[r]]=r*2+1
  return labels2



def dbscanCluster(segs, eps=16, min_samples=3, mode='topDown'):
  # cluster 2d line segments into lanes, returns cluster id for every segment
  num_segs = segs.shape[0]
  dist_mat = np.zeros([num_segs, num_segs], dtype='f4') # initialize distance matrix
  for cnt1 in range(num_segs):
    for cnt2 in range(num_segs):
      dist_mat[cnt1, cnt2]=segmentDist3(segs[cnt1,:], segs[cnt2,:])
  db = DBSCAN(eps=eps, metric='precomputed', min_samples=min_samples).fit(dist_mat)
  labels = orderLabels(db.labels_, segs, mode)
  return labels

def jlinkCluster(segs, ids, deg=3, tol=0.25):
  # use j-linkage cluster algorithm to refine the dbscan clusters, returning a list of polynomial models
  # ids are initialized as the 'labels' returned by dbscanCluster.
  # initial guess of number of lanes
  numSegs = segs.shape[0]
  numClusters = np.unique(ids).shape[0]
  pts = np.reshape(segs[:,0:4], [numSegs*2,2])
  ids = np.repeat(ids,2)
  numPts = pts.shape[0]
  '''
  Input: the set of data points, each point represented by its preference set (PS)
  Output: clusters of points belonging to the same model
  1. Put each point in its own cluster.
  2. Define the PS of a cluster as the intersection of the PSs of its points.
  3. Among all current clusters, pick the two clusters with the smallest Jaccard
  distance between the respective PSs.
  4. Replace these two clusters with the union of the two original ones.
  5. Repeat from step 3 while the smallest Jaccard distance is lower than 1.
  '''
  pointPs = np.zeros([numPts,numClusters],dtype=bool)
  clusterPs =np.zeros([numClusters,numClusters],dtype=bool) 
  # compute Ps for all pts
  for l in range(numClusters):
    currIds = np.where(ids==l) # points associated with this cluster
    currPts = pts[currIds,:]
    coef,residuals,_,_,_ = np.polyfit(currPts[:,2], currPts[:,0], deg,full=True) # fit x as a function of z
    polynomial = numpy.poly1d(coef)
    # find errors from all pts to the current model
    predX = polynomial(pts[:,2])
    errs = np.abs(pts[:,0]-predX)
    pointPs[:,l] = np.logical_or(pointPs[:,l], errs<tol)
  # compute Ps for all clusters
  for l in range(numClusters):
    currIds = np.where(ids==l) # points associated with this cluster
    clusterPs[l,:] = np.prod(pointPs[currIds,:], axis=0)
    
  
  while True:
    jDist =np.ones([numClusters,numClusters],dtype='f4')
    # fill distance matrix
    for l1 in range(numClusters):
      for l2 in range(l1):
        jDist[l1,l2] = jaccard(clusterPs[l1,:], clusterPs[l2,:])
    np.argmin(jDist)
    if minJDsit>=1:
      break
    else:
      # merge two clusters with the smallest jaccard dist
      numClusters-=1
      clusterPs[l1,:] = np.logical_or(clusterPs[l1,:], clusterPs[l2,:])
      clusterPs = np.delete(clusterPs, [l2], axis=0)
      clusterPs[:,l1] = np.logical_or(clusterPs[:,l1], clusterPs[:,l2])
      clusterPs = np.delete(clusterPs, [l2], axis=1)
      ids[ids==l2]=l1
      ids[ids>l2] -= 1
  return ids
    
  '''
  confs = pred[candidates[:,0], candidates[:,1]]
  candidates = np.c_[ candidates, confs ] 
  #candidates = np.c_[ candidates, np.zeros(candidates.shape[0]) ] 
  sortidx = np.argsort(-candidates[:,2]) # sort in decreasing confidence
  candidates=candidates[sortidx,:]
  
  cnt=0
  for c in candidates:
    curr_reg = reg_pred[c[0], c[1],:]
    all_reg =reg_pred[candidates[:,0].astype('i4'), candidates[:,1].astype('i4'),:]
    dist1 = np.sqrt((all_reg[:,0]-curr_reg[0])**2+(all_reg[:,1]-curr_reg[1])**2)
    dist2 = np.sqrt((all_reg[:,2]-curr_reg[2])**2+(all_reg[:,3]-curr_reg[3])**2)
    dist3 = np.sqrt((all_reg[:,0]-curr_reg[2])**2+(all_reg[:,1]-curr_reg[3])**2)
    dist4 = np.sqrt((all_reg[:,2]-curr_reg[0])**2+(all_reg[:,3]-curr_reg[1])**2)
    all_angle = np.arctan((all_reg[:,3]-all_reg[:,1])/(all_reg[:,2]-all_reg[:,0]))
    curr_angle = np.arctan((curr_reg[3]-curr_reg[1])/(curr_reg[2]-curr_reg[0]))
    curr_adj1 = np.logical_or(dist1<joinWin, dist2<joinWin)
    curr_adj2 = np.logical_or(dist3<joinWin, dist4<joinWin)
    curr_adj = np.logical_or(curr_adj1, curr_adj2)
    adj_mat[cnt,:] = np.logical_and(curr_adj, all_angle-curr_angle<0.1)
    cnt+=1
  return adj_mat*254
  lane_cnt=1
  for seed in candidates:
    if seed[2]<highThresh: # all following candidates will not be seeds.
      break
    if seed[3]>0: # already attached to a lane, cannot be a seed.
      continue
    # seeding a lane
    seed[3]=lane_cnt
    for c in candidates:
      pass        
  '''  
