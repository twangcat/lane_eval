from transformations import euler_matrix
from copy import deepcopy
import numpy as np
import logging
from math import isnan
import cv,cv2
__all__=['MultilaneLabelEvaluator']

# class for evaluating multilane detection performance using various metrics.
# lane id convention: even numbers are for lane boundaries to the left
# odd numbers are for lane boundaries to the right.
# eg: lane 0 is the left boundary of ego lane, lane 3 is the right boundary of the first lane to the right, etc.

class MultilaneLabelEvaluator():
    def __init__(self, depths=np.arange(15,90,5), numLanes=6,mbly=False):
      self.depths=depths # depths to evaluate lane detection
      self.tp=np.zeros([len(depths), numLanes], dtype='f4') # true positive counts by each lane, each depth
      self.fp_cnt=np.zeros([len(depths), numLanes], dtype='f4') # false positive counts
      self.fn_cnt=np.zeros([len(depths), numLanes], dtype='f4') # false negative counts
      self.pd=np.zeros([len(depths), numLanes], dtype='f4') # confident prediction counts 
      self.lat_err=np.zeros([len(depths), numLanes], dtype='f4') # lateral sqrerr, for confident predictions only
      self.mbly = mbly # whether it's evaluating mbly predictions
      assert(numLanes%2==0),'numLanes must be an even number!'
      self.numLanes = numLanes # maximum number of lane boundaries we care about, starting from ego lane.
      self.tol = np.arange(0.1,1.8,0.1)
    def sync(self, root):
      comm.Bcast([self.tp, np.prod(self.tp.shape), MPI.FLOAT], root=root)
      comm.Bcast([self.fp_cnt, np.prod(self.fp_cnt.shape), MPI.FLOAT], root=root)
      comm.Bcast([self.fn_cnt, np.prod(self.fn_cnt.shape), MPI.FLOAT], root=root)
      comm.Bcast([self.pd, np.prod(self.pd.shape), MPI.FLOAT], root=root)
      comm.Bcast([self.lat_err, np.prod(self.lat_err.shape), MPI.FLOAT], root=root)
    def dist2Id(self, dist, laneWidth=3.6576): # default lane width for US highways is 12 ft
      return np.floor(np.abs(dist)/laneWidth)*2+int(dist>0)
    
    def findDist(self, pt1, pt2, origin, direction):
      vec1 = (pt2-pt1)/np.sqrt(np.sum(((pt2-pt1)**2))) # unit vec of first line
      diff = origin-pt1
      s = (np.dot(diff,vec1)*np.dot(vec1,direction)-np.dot(diff,direction))/(1-np.dot(vec1,direction)**2)
      return s 

    def setPred(self,pred):
      # go through each lane. Define lane id based on lateral dist at fixed longitudinal distances.
      center2 = self.trajectory['center'].transpose()
      sideways = self.trajectory['sideways']
      num_anchors = center2.shape[0]
      #id_change=False # if the lane id changed during the course of this lane in the current frame
      # define the anchor pts on this lane
      anchors = np.empty([0,5])
      for pt in pred['pts']:
        for n in range(num_anchors):
          diff=pt-center2[n:n+1,:]
          minid = np.argmin((diff**2).sum(1))
          cross = np.cross(diff[minid,:],sideways[n,:])
          sine = np.sqrt(np.sum(cross**2))
          #if (minid==pt.shape[0]-1 or minid==0) and np.abs(sine)<0.05:
          if np.abs(sine)<0.02:
            mindist = np.dot(diff[minid,:],sideways[n,:]) # length projected to perpendicular vector
          else:
            if (cross[1]<0 and minid==pt.shape[0]-1) or (cross[1]>0 and minid==0):
              continue
            #if cross[1]>0:
            #  minid2 = minid-1
            #else:
            #  minid2 = minid+1
            #mindist = self.findDist(pt[minid,:], pt[minid2,:], center2[n,:], sideways[n,:])
            mindist = np.dot(diff[minid,:],sideways[n,:]) # length projected to perpendicular vector
          anchor_id = self.dist2Id(mindist)
          temp = np.empty([1,5])
          temp[0,0:3]=center2[n,:]+mindist*sideways[n,:]
          temp[0,3]=n
          temp[0,4]=anchor_id
          anchors = np.r_[anchors, temp]
      pred['anchors']=anchors
      self.pred=pred
 
    def setLabels(self,lanes):
      # assign lane ids to the labels so that 0 corresponds to the left lane boundary of ego-lane, 1 is right boundary of ego-lane, 2 is left lane boundary of left lane, etc
      total_num_lanes = len(lanes)
      nearValues = np.empty(total_num_lanes) # x coords of the 'nearest' pt of each lane
      for n in range(total_num_lanes):
        l = lanes[n].transpose()
        # use linear interpolation to find intersection with z=0 plane.
        if l.shape[0]>1:
          slope=(l[1,0]-l[0,0])/(l[1,2]-l[0,2])
          nearValues[n] = l[0,0]-slope*l[0,2]
        else: 
          nearValues[n]=float('nan')
      # fill in self.lanes with data as well as id.
      # find the lane ids to left and right of car according to convention.
      lIds = np.where(nearValues<0)[0]
      rIds = np.where(nearValues>=0)[0]
      newlIds=None
      if len(lIds)>0:
        lXs = nearValues[lIds]
        newlIds = lIds[np.argsort(-lXs)]
      newrIds=None
      if len(rIds)>0:
        rXs = nearValues[rIds]
        newrIds = rIds[np.argsort(rXs)]
      self.labels  = {'pts':[],'id':[]}
      cnt=0
      while cnt*2<self.numLanes:
        if newlIds is not None and cnt<len(newlIds):
          self.labels['pts'].append(lanes[newlIds[cnt]].transpose())
          self.labels['id'].append(cnt*2)
        if newrIds is not None and cnt<len(newrIds):
          self.labels['pts'].append(lanes[newrIds[cnt]].transpose())
          self.labels['id'].append(cnt*2+1)
        cnt+=1

    # accumulate counts for evaluation
    def accEvalCounts(self, pred, labels, trajectory):
      #self.setLabels(labels)
      self.labels=labels
      self.trajectory=trajectory
      self.setPred(pred)
      
      matched_pairs, fps, fns = self.matchAnchors()
      for pair in matched_pairs:
        p=pair['pred']
        l=pair['label']
        # need to define depth_idx, and lane id.
        d_idx=l[3]
        id = min(l[4],self.numLanes-1)
        self.pd[d_idx, id]+=1
        self.tp[d_idx,id]+=1
        self.lat_err[d_idx, id]+=np.sum((p[0:3]-l[0:3])**2)
      
      # collect false positive counts from unmatched predictions
      for fp in fps:
        self.fp_cnt[fp[3],min(fp[4], self.numLanes-1)]+=1
      # collect false negative counts from unmatched labels
      for fn in fns:
        self.fn_cnt[fn[3],min(fn[4], self.numLanes-1)]+=1
      '''
      matched_pairs, fps, fns = self.matchLanes()
      # collect true and false positives from matched pairs, based on their lateral distance
      for pair in matched_pairs:
        p=pair['pred']
        l=pair['label']
        for d_idx in range(len(self.depths)):
          d=self.depths[d_idx]
          # for a certain depth, using linear interpolation to find the X value of the predicted lane crossing this depth.
          predX = self.xByDepth(p['pts'],d)
          labelX = self.xByDepth(l['pts'],d)
          if not isnan(predX): # has prediction at this depth
            self.pd[d_idx, l['id']]+=1
            if not isnan(labelX): # has label at this depth
              if abs(predX-labelX)<self.tol: # smaller than tol, counted as tp
                self.tp[d_idx,l['id']]+=1
              else:
                self.fp_cnt[d_idx,l['id']]+=1
                self.fn_cnt[d_idx,l['id']]+=1
              # accumulate sqrerr
              self.lat_err[d_idx, l['id']]+=(predX-labelX)**2
            else:  # no label at this depth
              self.fp_cnt[d_idx,l['id']]+=1
          else:
            if not isnan(labelX): # has label at this depth, but no prediction
              self.fn_cnt[d_idx,l['id']]+=1 # increase false neg count

      # collect false positive counts from unmatched predictions
      for fp_id in range(len(fps['id'])):
        for d_idx in range(len(self.depths)):
          d=self.depths[d_idx]
          if not isnan(self.xByDepth(fps['pts'][fp_id],d)):
            self.fp_cnt[d_idx,fps['id'][fp_id]]+=1
      # collect false negative counts from unmatched labels
      for fn_id in range(len(fns['id'])):
        for d_idx in range(len(self.depths)):
          d=self.depths[d_idx]
          if not isnan(self.xByDepth(fns['pts'][fn_id],d)):
            self.fn_cnt[d_idx,fns['id'][fn_id]]+=1
      '''
    def getPrecision(self):
      return self.tp/(self.tp+self.fp_cnt)

    def getRecall(self):
      return self.tp/(self.tp+self.fn_cnt)

    def getFscore(self,beta=1):
      a = (1+beta**2)*self.tp
      f = a/(a+beta**2*self.fn_cnt+self.fp_cnt)
      return f
   
    def getSqrErr(self,):
      return np.sqrt(self.lat_err)/self.pd

    def xByDepth(self,pts, depth):
      idx = np.argmin(np.abs(pts[:,2]-depth))
      if pts[idx,2]==depth:
        return pts[idx,0]
      elif idx==0:
        return float('nan')
      elif idx==pts.shape[0]-1:
        return float('nan')
      elif pts[idx,2]<depth:
        idx2 = idx+1
      elif pts[idx,2]>depth:
        idx2 = idx-1

      return pts[idx,0]+(pts[idx2,0]-pts[idx,0])*(depth-pts[idx,2])/(pts[idx2,2]-pts[idx,2])

    def matchAnchors(self):
      i=0
      matched_pairs = []
      while i<self.pred['anchors'].shape[0] and self.labels['anchors'].shape[0]>0:
        unmatched=True
        # try to match current prediction anchor with the nearest label anchor
        diff = self.labels['anchors'][:,0:3]-self.pred['anchors'][i:i+1,0:3]
        dist = (diff[:,[0,2]]**2).sum(1)
        match_idx = np.argmin(dist)
        tol = (self.labels['anchors'][match_idx,2]-self.depths[0])/100.0+0.1 # variable tolerance with z distance
        if self.mbly:
          tol = (self.labels['anchors'][match_idx,2]-self.depths[0])/100.0+0.2 # variable tolerance with z distance
        if dist[match_idx]<tol:
          l=self.labels['anchors'][match_idx,:]
          p=self.pred['anchors'][i,:]
          matched_pairs.append({'pred':p, 'label':l})
          self.labels['anchors']=np.delete(self.labels['anchors'],[match_idx],axis=0) 
          self.pred['anchors']=np.delete(self.pred['anchors'],[i],axis=0)
        else:  
          i+=1
      fp = self.pred['anchors'] # any predictions left unmatched are false positives
      fn = self.labels['anchors'] # any labels left unmatched are false negatives
      self.matched_pairs = matched_pairs
      self.fp = fp
      self.fn = fn
      return matched_pairs, fp, fn
 
    def matchLanes(self):
      # matching predicted lanes to ground truth lane labels.
      sortidx = np.argsort(-np.array(self.pred['conf'])) # sort in decreasing confidence.
      self.pred['pts']=[ self.pred['pts'][i] for i in sortidx]
      self.pred['id']=[ self.pred['id'][i] for i in sortidx]
      self.pred['conf']=[ self.pred['conf'][i] for i in sortidx]
      i=0
      matched_pairs = []
      while i<len(self.pred['pts']) and len(self.labels['pts'])>0:
        #p=self.pred['pts'][i]
        unmatched=True
        for j in range(len(self.labels['pts'])):
          #l=self.labels['pts'][j]
          if self.pred['id'][i]==self.labels['id'][j]:
            l = {'pts':self.labels['pts'].pop(j), 'id':self.labels['id'].pop(j)}
            p = {'pts':self.pred['pts'].pop(i),'conf':self.pred['conf'].pop(i),'id':self.pred['id'].pop(i)}
            matched_pairs.append({'pred':p, 'label':l})
            unmatched=False
            break
            
        ''' 
        dist = np.zeros(len(self.labels))
        for j in len(self.labels):
          l=self.labels[j]['pts']
          dist[j] = distDTW(p,l)/p.shape[0] # average distance btw pred and label, computed with dynamic time warping
        match_id = np.argmin(dist)
        # add the matched pair to matched list and deleting them from pred and labels
        matched_pairs.append({'pred':p, 'label':self.labels[match_id]})
        '''
        if unmatched:
          i+=1
      fp = self.pred # any predictions left unmatched are false positives
      fn = self.labels # any labels left unmatched are false negatives
      return matched_pairs, fp, fn
    

    def drawOnImage(self,image):
      for pair in self.matched_pairs:
        p=pair['pred'][0:3]
        l=pair['label'][0:3]
        cv2.circle(image, (int(l[0]*6+240), int(479-(l[2]-4)*6+6)), 3, (0,255,0),-1)
        cv2.circle(image, (int(p[0]*6+240), int(479-(p[2]-4)*6+6)), 3, (255,0,0),-1)
      
      # collect false positive counts from unmatched predictions
      for fp in self.fp:
        cv2.circle(image, (int(fp[0]*6+240), int(479-(fp[2]-4)*6+6)), 3, (0,0,255),-1)
      # collect false negative counts from unmatched labels
      for fn in self.fn:
        cv2.circle(image, (int(fn[0]*6+240), int(479-(fn[2]-4)*6+6)), 3, (0,255,255),-1)
      return image
