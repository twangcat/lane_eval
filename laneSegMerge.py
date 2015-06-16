from sklearn.cluster import DBSCAN
import numpy as np
import copy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


__all__=['simpleAggCluster2','simpleAggCluster','dbscanCluster','dbscanJLinkCluster','jLinkCluster', 'ransacCluster','ransacFit']

def jaccard(im1, im2):
    """
    Computes the Jaccard metric, a distance measure of set similarity.
 
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
        Maximum similarity = 0
        No similarity = 1
    
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
    return (float(union.sum())-intersection.sum()) / float(union.sum())



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
  angleW = 100
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


def simpleAggCluster2(segs, eps=16, min_samples=3, mode='topDown'):
  # Doesn't work when far end of two lanes are touching
  # cluster 2d line segments into lanes, returns cluster id for every segment
  num_segs = segs.shape[0]
  dist_mat = 1e10*np.ones([num_segs, num_segs], dtype='f4') # initialize distance matrix
  for cnt1 in range(num_segs):
    for cnt2 in range(num_segs):
      if cnt1!=cnt2:
        dist_mat[cnt1, cnt2]=segmentDist3(segs[cnt1,:], segs[cnt2,:])
 
  labels=-2*np.ones(num_segs,dtype='i4') # -2 means not assigned yet.
  sortidx = np.argsort(segs[:,1])[::-1] # sort according descending z distance
  lane_id=0
  for idx in sortidx:
    if labels[idx]!=-2: # this segment already in one of the clusters, or labelled as noise
      continue
    neighbors = np.zeros(num_segs, dtype=bool)
    neighbors[idx]=True
    while np.sum(neighbors)>0:
      labels[neighbors] = lane_id
      new_neighbors = np.logical_and(np.sum(dist_mat[neighbors,:]<eps, axis=0)>0, labels==-2)
      new_neighbors = np.logical_and(new_neighbors, labels==-2)
      neighbors = np.logical_and(new_neighbors, segs[:,1]<=np.median(segs[neighbors,1]))
    if np.sum(labels==lane_id)<min_samples:
      labels[labels==lane_id]=-1
    else:
      lane_id+=1
  labels2 = orderLabels(labels, segs, mode)
  return labels2

def simpleAggCluster(segs, eps=16, min_samples=3, mode='topDown'):
  # Doesn't work when near end of two lanes are touching
  # cluster 2d line segments into lanes, returns cluster id for every segment
  num_segs = segs.shape[0]
  dist_mat = 1e10*np.ones([num_segs, num_segs], dtype='f4') # initialize distance matrix
  for cnt1 in range(num_segs):
    for cnt2 in range(num_segs):
      if cnt1!=cnt2:
        dist_mat[cnt1, cnt2]=segmentDist3(segs[cnt1,:], segs[cnt2,:])
 
  labels=-2*np.ones(num_segs,dtype='i4') # -2 means not assigned yet.
  sortidx = np.argsort(segs[:,1]) # sort according ascending z distance
  lane_id=0
  seed_size = 10000
  for idx in sortidx:
    if labels[idx]!=-2: # this segment already in one of the clusters, or labelled as noise
      continue
    '''
    neighbors = np.zeros(num_segs, dtype=bool)
    neighbors[idx]=True
    iter=0
    while np.sum(neighbors)>0:
      labels[neighbors] = lane_id
      new_neighbors = np.logical_and(np.sum(dist_mat[neighbors,:]<eps, axis=0)>0, labels==-2)
      print 'seg idx',
      print idx
      print np.sum(new_neighbors),
      if iter<seed_size:
        new_neighbors_seed = np.sum(dist_mat[neighbors,:]==np.min(dist_mat[neighbors,:]), axis=0)>0
        new_neighbors = np.logical_and(new_neighbors, new_neighbors_seed)
      print np.where(new_neighbors)
      #print np.sum(new_neighbors)
      #print labels[new_neighbors]
      #new_neighbors = np.logical_and(new_neighbors, labels==-2)
      neighbors = new_neighbors
      #neighbors = np.logical_and(new_neighbors, segs[:,1]>np.median(segs[neighbors,1]))
      iter+=1
    '''

    neighbor = idx

    while True:
      labels[neighbor] = lane_id
      if np.all(labels!=-2):
        break
      remaining_idx = np.where(labels==-2)[0]
      min_idx = np.argmin(dist_mat[neighbor,remaining_idx])
      new_neighbor = remaining_idx[min_idx]
      if dist_mat[neighbor,new_neighbor]>eps:
        break
      neighbor = new_neighbor
    if np.sum(labels==lane_id)<min_samples:
      labels[labels==lane_id]=-1
    else:
      lane_id+=1
  labels2 = orderLabels(labels, segs, mode)
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

def dbscanCluster2(segs, eps=1, min_samples=1, mode='topDown'):
  # cluster 2d line segments into lanes, returns cluster id for every segment
  # often clusters 2 near lanes into one cluster.
  num_segs = segs.shape[0]
  dist_mat = np.zeros([num_segs, num_segs], dtype='f4') # initialize distance matrix
  for cnt1 in range(num_segs):
    for cnt2 in range(num_segs):
      dist_mat[cnt1, cnt2]=segmentDist2(segs[cnt1,:], segs[cnt2,:])
  db = DBSCAN(eps=eps, metric='precomputed', min_samples=min_samples).fit(dist_mat)
  #labels = splitLabels(db.labels_, segs)
  #labels2 = orderLabels(db.labels_, segs, mode)
  labels2 = db.labels_
  return labels2

def dbscanJLinkCluster(segs, eps=10, min_samples=10, mode='topDown'):
  # cluster 2d line segments into lanes, returns cluster id for every segment
  # Then for each cluster, further split it using Jlink clustering.
  num_segs = segs.shape[0]
  segs_xz = segs[:,[0,2,3,5]] # x-z elements of the segments
  model=quadModel
  dist_mat = np.zeros([num_segs, num_segs], dtype='f4') # initialize distance matrix
  for cnt1 in range(num_segs):
    for cnt2 in range(num_segs):
      dist_mat[cnt1, cnt2]=segmentDist2(segs_xz[cnt1,:], segs_xz[cnt2,:])
  db = DBSCAN(eps=eps, metric='precomputed', min_samples=min_samples).fit(dist_mat)
  #labels = splitLabels(db.labels_, segs)
  ids = db.labels_
  pts = (segs[:,0:3]+segs[:,3:6])/2.0
  
  inlier_thresh=0.3
  unique_ids = np.sort(np.unique(ids))
  for i in range(unique_ids.shape[0]):
    id = unique_ids[i]
    if id<0:
      continue
    idx = np.where(ids==id)[0]
    if len(idx)<min_samples:
      continue
    params, pcov = curve_fit(model, pts[idx,1], pts[idx,0])
    lat_vals = model(pts[idx,1], params[0],params[1],params[2])
    sqr_errs = (lat_vals-pts[idx,0])**2
    if np.mean(sqr_errs)<inlier_thresh:
      continue
    
    sub_ids = jLinkCluster(segs_xz[idx,:])
    unique_sub_ids = np.sort(np.unique(sub_ids))
    unique_ids[unique_ids>id]+=(len(unique_sub_ids))-1
    ids[ids>id]+=(len(unique_sub_ids))-1
    ids[idx] += sub_ids

  ids = orderLabels(ids, segs_xz, mode)
  # for each cluster, fit curve.
  new_segs = np.array(segs)
  unique_ids = np.sort(np.unique(ids))
  for i in range(unique_ids.shape[0]):
    id = unique_ids[i]
    if id<0:
      continue
    idx = ids==id
    if np.sum(idx)<5:
      ids[idx]=-1
      continue
    '''
    # if less than 6 pts, fit linear model
    if np.sum(idx)<6:
      # fit curve of x vs z
      params, pcov = curve_fit(linearModel, pts[idx,2], pts[idx,0])
      new_segs[idx,0] = linearModel(segs[idx,2], params[0],params[1])
      new_segs[idx,3] = linearModel(segs[idx,5], params[0],params[1])
      # fit curve of y vs z
      params, pcov = curve_fit(linearModel, pts[idx,2], pts[idx,1])
      new_segs[idx,1] = linearModel(segs[idx,2], params[0],params[1])
      new_segs[idx,4] = linearModel(segs[idx,5], params[0],params[1])
      continue
    '''
    # Otherwise fit quadratic curve
    # fit curve of x vs z
    params, pcov = curve_fit(model, pts[idx,2], pts[idx,0])
    new_segs[idx,0] = model(segs[idx,2], params[0],params[1],params[2])
    new_segs[idx,3] = model(segs[idx,5], params[0],params[1],params[2])
    # fit curve of y vs z
    params, pcov = curve_fit(model, pts[idx,2], pts[idx,1])
    new_segs[idx,1] = model(segs[idx,2], params[0],params[1],params[2])
    new_segs[idx,4] = model(segs[idx,5], params[0],params[1],params[2])
    
  return ids, new_segs








def splitLabels(labels, segs):
  labels=np.array(labels,dtype='i4')
  model=quadModel
  num_labels = np.max(labels)+1
  pts = (segs[:,0:2]+segs[:,2:4])/2.0
  grads = (segs[:,2]-segs[:,0])/(segs[:,3]-segs[:,1]) # dx/dz
  for l in range(num_labels):
    ids = np.where(labels==l)[0]
    sub_labels = ransacCluster(segs[ids])
    labels[ids[sub_labels==-1]]=-1
    for nl in range(1,np.max(sub_labels)+1):
      labels[ids[sub_labels==nl]]=num_labels+nl-1
  return labels
    


def linearModel(x, a, b): # linear model
  return a*x+b


def quadModel(x, a, b, c): # quadratic curve model
  return a*x*x+b*x+c


def meanshiftFit(pts, min_pts = 20, inlier_thresh=0.2, max_iter=20, tol=0.1):
  # Often find quadratic curves that crosses 2 lanes
  model=quadModel
  inliers = np.zeros(pts.shape[0],dtype=bool)
  max_err = 1e10
  best_err = 1e10
  num_pts = pts.shape[0]
  best_model=[]
  iter=0
  inlier_ids = np.random.choice(num_pts, min_pts)
  maybeinliers = np.zeros(num_pts,dtype=bool)
  maybeinliers[inlier_ids]=1
  params, pcov = curve_fit(model, pts[maybeinliers,1], pts[maybeinliers,0])
  while iter<max_iter and best_err>tol:
    x_vals = pts[:,0]
    z_vals = pts[:,1]
    lat_vals = model(z_vals, params[0],params[1],params[2])
    sqr_errs = (lat_vals-x_vals)**2
    inliers = sqr_errs<inlier_thresh
    #plt.plot(pts[:,1], pts[:,0],'go')
    #plt.plot(pts[inliers,1], pts[inliers,0],'ro')
    #aa = np.sort(pts[inliers,1])
    #plt.plot(aa, model(aa, params[0],params[1],params[2]),'b')
    ##plt.savefig(str(iter)+'.png')
    #plt.ion()
    #plt.show()
    #plt.pause(3)
    #plt.clf()
    if np.sum(inliers)>min_pts:
      params, pcov = curve_fit(model, pts[inliers,1], pts[inliers,0])
      lat_vals = model(pts[inliers,1], params[0],params[1],params[2])
      sqr_errs = (lat_vals-pts[inliers,0])**2
      max_err = np.max(sqr_errs)
      if max_err<best_err:
        best_err=max_err
        best_model=params
    else:
      inlier_ids = np.random.choice(num_pts, min_pts)
      maybeinliers = np.zeros(num_pts,dtype=bool)
      maybeinliers[inlier_ids]=1
      params, pcov = curve_fit(model, pts[maybeinliers,1], pts[maybeinliers,0])
    iter+=1
  return best_err, best_model, inliers


def ransacFit(pts, min_pts = 20, gd_pts=40, inlier_thresh=1, max_iter=100, tol=0.1):
  model=quadModel
  inliers = np.zeros(pts.shape[0],dtype=bool)
  max_err = 1e10
  best_err = 1e10
  num_pts = pts.shape[0]
  best_model=[]
  iter=0
  while iter<max_iter and best_err>tol:
    inlier_ids = np.random.choice(num_pts, min_pts)
    maybeinliers = np.zeros(num_pts,dtype=bool)
    maybeinliers[inlier_ids]=1
    params, pcov = curve_fit(model, pts[maybeinliers,1], pts[maybeinliers,0])
    x_vals = pts[:,0]
    z_vals = pts[:,1]
    lat_vals = model(z_vals, params[0],params[1],params[2])
    sqr_errs = (lat_vals-x_vals)**2
    #alsoinliers = np.logical_and(sqr_errs<inlier_thresh, np.logical_not(maybeinliers))
    alsoinliers = sqr_errs<inlier_thresh
    if np.sum(alsoinliers)>gd_pts:
      #inliers = np.logical_or(alsoinliers, maybeinliers)
      inliers = alsoinliers
      params, pcov = curve_fit(model, pts[inliers,1], pts[inliers,0])
      lat_vals = model(pts[inliers,1], params[0],params[1],params[2])
      sqr_errs = (lat_vals-pts[inliers,0])**2
      max_err = np.max(sqr_errs)
      if max_err<best_err:
        best_err=max_err
        best_model=params
    iter+=1
  return best_err, best_model, inliers

def ransacCluster(segs, min_pts=20):
  # Often find quadratic curves that crosses 2 lanes
  pts = (segs[:,0:2]+segs[:,2:4])/2.0
  grads = (segs[:,2]-segs[:,0])/(segs[:,3]-segs[:,1]) # dx/dz
  labels = -np.ones(pts.shape[0], dtype='i4')
  remaining = np.arange(pts.shape[0])
  iter=0
  while remaining.shape[0]>min_pts:
    err,model,inliers = meanshiftFit(pts[remaining])
    if np.sum(inliers)==0:
      break
    labels[remaining[inliers]]=iter
    remaining = remaining[np.logical_not(inliers)]
    iter+=1
  return labels.tolist()



# numerically stable sum
def kahan_sum(input):
  sum = 0.0
  c = 0.0 #A running compensation for lost low-order bits.
  for i in range(input.shape[0]):
    y = input[i] - c     # So far, so good: c is zero.
    t = sum + y      # Alas, sum is big, y small, so low-order digits of y are lost.
    c = (t - sum) - y # (t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
    sum = t           # Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
    # Next time around, the lost low part will be added to y in a fresh attempt.
  return sum




def jLinkSample(segs, M=200, inlier_thresh=0.3):
  numSegs = segs.shape[0]
  PM = np.zeros([numSegs, M], dtype=bool) # preference matrix
  pts = (segs[:,0:2]+segs[:,2:4])/2.0
  x_vals = pts[:,0]
  z_vals = pts[:,1]
  grads = (segs[:,2]-segs[:,0])/(segs[:,3]-segs[:,1]) # dx/dz
  model=quadModel
  poly_order = 2
  sigma = 1 
  dist_mat = 1e10*np.ones([numSegs, numSegs], dtype='f4') # initialize distance matrix
  for cnt1 in range(numSegs):
    for cnt2 in range(numSegs):
      if cnt1!=cnt2:
        dist_mat[cnt1, cnt2]=segmentDist3(segs[cnt1,:], segs[cnt2,:])
  for m in range(M):
    rand_pts = np.empty(poly_order+1, dtype='i4')
    rand_pts[0] = np.random.choice(numSegs)
    prob = np.exp(-dist_mat[rand_pts[0],:]/sigma).astype('f8')
    prob[rand_pts[0]]=0
    prob /= np.sum(prob)
    #prob /= kahan_sum(prob)
    #print "somme sur p:", repr(kahan_sum(prob))
    #rand_pts[1] = np.random.choice(numSegs, p=prob)
    rand_pts[1] = np.random.choice(numSegs)
    prob = (np.exp(-dist_mat[rand_pts[0],:]/sigma)+np.exp(-dist_mat[rand_pts[1],:]/sigma)).astype('f8')
    prob[rand_pts[0]]=0
    prob[rand_pts[1]]=0
    prob /= np.sum(prob)
    #prob /= kahan_sum(prob)
    #rand_pts[2] = np.random.choice(numSegs, p=prob)
    rand_pts[2] = np.random.choice(numSegs)
    params, pcov = curve_fit(model, pts[rand_pts,1], pts[rand_pts,0])
    lat_vals = model(z_vals, params[0],params[1],params[2])
    sqr_errs = (lat_vals-x_vals)**2
    PM[:,m] = sqr_errs<inlier_thresh
  return PM
    

def jLinkSample2(segs, M=200, inlier_thresh=0.3):
  numSegs = segs.shape[0]
  PM = np.zeros([numSegs, M], dtype=bool) # preference matrix
  pts = (segs[:,0:2]+segs[:,2:4])/2.0
  x_vals = pts[:,0]
  z_vals = pts[:,1]
  grads = (segs[:,2]-segs[:,0])/(segs[:,3]-segs[:,1]) # dx/dz
  model=quadModel
  poly_order = 2
  sigma = 1 
  dist_mat = 1e10*np.ones([numSegs, numSegs], dtype='f4') # initialize distance matrix
  for cnt1 in range(numSegs):
    for cnt2 in range(numSegs):
      if cnt1!=cnt2:
        dist_mat[cnt1, cnt2]=segmentDist3(segs[cnt1,:], segs[cnt2,:])
  for m in range(M):
    rand_pts = np.empty(poly_order+1, dtype='i4')
    rand_pts[0] = np.random.choice(numSegs)
    prob = np.exp(-dist_mat[rand_pts[0],:]/sigma).astype('f8')
    prob[rand_pts[0]]=0
    prob /= np.sum(prob)
    #prob /= kahan_sum(prob)
    #print "somme sur p:", repr(kahan_sum(prob))
    rand_pts[1] = np.random.choice(numSegs, p=prob)
    prob = (np.exp(-dist_mat[rand_pts[0],:]/sigma)+np.exp(-dist_mat[rand_pts[1],:]/sigma)).astype('f8')
    prob[rand_pts[0]]=0
    prob[rand_pts[1]]=0
    prob /= np.sum(prob)
    #prob /= kahan_sum(prob)
    rand_pts[2] = np.random.choice(numSegs, p=prob)
    params, pcov = curve_fit(model, pts[rand_pts,1], pts[rand_pts,0])
    lat_vals = model(z_vals, params[0],params[1],params[2])
    sqr_errs = (lat_vals-x_vals)**2
    PM[:,m] = sqr_errs<inlier_thresh
  return PM



def jLinkCluster(segs, deg=3):
  # This version over-segments the detections using aggressive dbscan first, then use the clusters as initial clusters for j-linkage
  # issues: 1. Sometimes the clusters found by jlink can still go across different lanes. Usually when inlier threshold is large
  #         2. when inlier threshold is small, an initial cluster can have all-0 preference set and that results in error.
  
  pointPs = jLinkSample(segs) # preference matrix of points
  numSegs = segs.shape[0]
  numClusters = numSegs
  clusterPs = np.zeros([numClusters,pointPs.shape[1]], dtype=bool)
  ids = np.arange(numClusters, dtype='i4') # place each seg in its own cluster
  
    
  
  while True:
    #print numClusters
    # compute Ps for all clusters
    for l in range(numClusters):
      currIds = np.where(ids==l)[0] # points associated with this cluster
      clusterPs[l,:] = np.prod(pointPs[currIds,:], axis=0)
    # merge clusters using JDist
    jDist =np.ones([numClusters,numClusters],dtype='f4')
    # fill distance matrix
    for l1 in range(numClusters):
      union_sum = np.sum((clusterPs[l1]+clusterPs), axis=1)
      jDist[l1,:] = (union_sum.astype('f4')-np.sum(clusterPs[l1,:]*clusterPs, axis=1))/union_sum.astype('f4')
      jDist[l1,l1] = 1
      #for l2 in range(l1):
      #  jDist[l1,l2] = jaccard(clusterPs[l1,:], clusterPs[l2,:])
    minJDist = np.min(jDist)
    l1 = np.where(jDist==minJDist)[0][0]
    l2 = np.where(jDist==minJDist)[1][0]
    if minJDist>=1:
      break
    else:
      # merge two clusters with the smallest jaccard dist
      numClusters-=1
      clusterPs[l1,:] = np.logical_or(clusterPs[l1,:], clusterPs[l2,:])
      clusterPs = np.delete(clusterPs, [l2], axis=0)
      #clusterPs[:,l1] = np.logical_or(clusterPs[:,l1], clusterPs[:,l2])
      #clusterPs = np.delete(clusterPs, [l2], axis=1)
      ids[ids==l2]=l1
      ids[ids>l2] -= 1
  unique_ids = np.sort(np.unique(ids))
  for i in range(unique_ids.shape[0]):
    ids[ids==unique_ids[i]]=i
  return ids


def jLinkClusterDBSCAN(segs, deg=3):
  # This version over-segments the detections using aggressive dbscan first, then use the clusters as initial clusters for j-linkage
  # issues: 1. Sometimes the clusters found by jlink can still go across different lanes. Usually when inlier threshold is large
  #         2. when inlier threshold is small, an initial cluster can have all-0 preference set and that results in error.
  
  pointPs = jLinkSample(segs) # preference matrix of points
  numSegs = segs.shape[0]
  ids = dbscanCluster(segs) 
  unique_ids = np.sort(np.unique(ids))
  for i in range(unique_ids.shape[0]):
    ids[ids==unique_ids[i]]=i
  numClusters = len(unique_ids)
  clusterPs = np.zeros([numClusters,pointPs.shape[1]], dtype=bool)
  #ids = np.arange(numClusters, dtype='i4') # place each seg in its own cluster
  
    
  
  while True:
    print numClusters
    # compute Ps for all clusters
    for l in range(numClusters):
      currIds = np.where(ids==l)[0] # points associated with this cluster
      clusterPs[l,:] = np.prod(pointPs[currIds,:], axis=0)
    # merge clusters using JDist
    jDist =np.ones([numClusters,numClusters],dtype='f4')
    # fill distance matrix
    for l1 in range(numClusters):
      union_sum = np.sum((clusterPs[l1]+clusterPs), axis=1)
      jDist[l1,:] = (union_sum.astype('f4')-np.sum(clusterPs[l1,:]*clusterPs, axis=1))/union_sum.astype('f4')
      jDist[l1,l1] = 1
      #for l2 in range(l1):
      #  jDist[l1,l2] = jaccard(clusterPs[l1,:], clusterPs[l2,:])
    minJDist = np.min(jDist)
    l1 = np.where(jDist==minJDist)[0][0]
    l2 = np.where(jDist==minJDist)[1][0]
    if minJDist>=1:
      break
    else:
      # merge two clusters with the smallest jaccard dist
      numClusters-=1
      clusterPs[l1,:] = np.logical_or(clusterPs[l1,:], clusterPs[l2,:])
      clusterPs = np.delete(clusterPs, [l2], axis=0)
      #clusterPs[:,l1] = np.logical_or(clusterPs[:,l1], clusterPs[:,l2])
      #clusterPs = np.delete(clusterPs, [l2], axis=1)
      ids[ids==l2]=l1
      ids[ids>l2] -= 1
  unique_ids = np.sort(np.unique(ids))
  for i in range(unique_ids.shape[0]):
    ids[ids==unique_ids[i]]=i
  return ids







def jLinkClusterOLD(segs, ids, deg=3, tol=0.25):
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
