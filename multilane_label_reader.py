from transformations import euler_matrix
from copy import deepcopy
import collections
import os
import numpy as np
from threading import Thread
from Queue import Queue
import pickle
from GPSReader import GPSReader
from GPSTransforms import *
from GPSReprojection import *
from WarpUtils import warpPoints
from Q50_config import *
from ArgParser import *
from numpy import array, dot, zeros, around, divide, ones
from scipy.io import loadmat
import string
import time
from SetPerspDist import setPersp
import cv,cv2
import os
import time
import copy
__all__=['MultilaneLabelReader']


blue = np.array([255,0,0])
green = np.array([0,255,0])
red = np.array([0,0,255])

def dist2color(dist, max_dist = 90.0):
  # given a distance and a maximum distance, gives a color code for the distance.
  # red being closest, green is mid-range, blue being furthest
  alpha = (dist/max_dist)
  if alpha<0.5:
    color = red*(1-alpha*2)+green*alpha*2
  else:
    beta = alpha-0.5
    color = green*(1-beta*2)+blue*beta*2
  return color.astype(np.int)

def colorful_line(img, start, end, start_color, end_color, thickness):
  # similar to cv.line, but draws a line with gradually (linearly) changing color. 
  # allows starting and ending color to be specified. 
  # implemented using recursion.
  if ((start[0]-end[0])**2 + (start[1]-end[1])**2)**0.5<=thickness*2:
    cv2.line(img, start, end ,start_color,thickness)
    return img
  mid = (int((start[0]+end[0])/2),int((start[1]+end[1])/2))
  mid_color = [int((start_color[0]+end_color[0])/2),int((start_color[1]+end_color[1])/2),int((start_color[2]+end_color[2]))/2]
  img = colorful_line(img, start, mid, start_color, mid_color, thickness)
  img = colorful_line(img, mid, end, mid_color, end_color, thickness)
  return img

def cropScaleLabels(labels, upper_left, scale):
  # scale the labels linearly according to the upper_left offset (x,y)
  # and the scales (x_scale, y_scale)
  return (labels-upper_left)*scale



class MultilaneLabelReader():
    def __init__(self,buffer_size, imdepth, imwidth, imheight, markingWidth=0.07, distortion_file='/scail/group/deeplearning/driving_data/perspective_transforms.pickle', pixShift=0, label_dim = [160,120], new_distort=False, predict_depth = False, readVideo=False):
      self.q = Queue(buffer_size) # queue holding actual data
      self.bq = Queue() # queue holding filenames, frame nums, etc.
      self.p = Thread(target=self.readBatches)
      self.new_distort = new_distort
      if new_distort:
        self.Ps = setPersp()
      else:
        self.Ps = pickle.load(open(distortion_file, 'rb'))
      self.buffer_size = buffer_size
      self.lane_values = dict()
      self.gps_values1 = dict()
      self.gps_values2 = dict()
      self.count=0
      self.markingWidth = markingWidth
      self.pixShift = pixShift
      self.labelw = label_dim[0]
      self.labelh = label_dim[1]
      self.labeld = 6 if predict_depth else 4
      self.imwidth = imwidth
      self.imheight = imheight
      self.imdepth = imdepth
      self.label_scale = None
      self.img_scale = None
      self.predict_depth = predict_depth
      self.visualize = readVideo
      self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,255),(128,255,128),(255,128,128),(128,128,0),(128,0,128),(0,128,128),(0,128,255),(0,255,128),(128,0,255),(128,255,0),(255,0,128),(255,128,0)]
    def start(self):
      self.p.start()

    def push_batch(self, batch):
      self.bq.put(batch)

    def pop_batch(self):
      item = self.q.get()
      return item

    def readBatches(self):
      while(True):
        # get a batch (filename, frame_numbers, perspective_ids) when available
        batch = self.bq.get()
        self.runLabelling(batch)

    def outputDistances(self, distances, framenum, meters_per_point, points_fwd, start_offset):
        output = []
        point_num = 1
        dist = 0

        framenum += 1
        while framenum < distances.size and point_num <= points_fwd:
            dist += distances[framenum]
            if point_num * meters_per_point <= dist - start_offset:
                output.append(framenum)
                point_num += 1
            else:
                framenum += 1
            
        return output

    def zDistances(self, distances, global_frame, starting_point, meters_per_point, points_fwd):
        output = []
        point_num = 1
        dist = 0
        for pt in xrange(points_fwd):
          dist = pt * meters_per_point+starting_point
          output.append((np.abs(distances-dist)).argmin()+global_frame)
        return output
    

    def dist2Id(self, dist, laneWidth=3.6576): # default lane width for US highways is 12 ft
      return np.floor(np.abs(dist)/laneWidth)*2+int(dist>0)

    def setLaneIDs(self, lane, center2, sideways):
      # go through each lane. Define lane id based on lateral dist at fixed longitudinal distances.
      num_anchors = center2.shape[1]
      #id_change=False # if the lane id changed during the course of this lane in the current frame
      # define the anchor pts on this lane
      anchors = np.empty([0,5])
      id=float('nan')
      for n in range(num_anchors):
        diff=lane-center2[:,n:n+1]
        minid = np.argmin((diff**2).sum(0))
        mindist = np.dot(diff[:,minid],sideways[n,:]) # length projected to perpendicular vector
        if (minid==0 or minid==lane.shape[1]-1) and np.abs(mindist/np.sqrt(np.sum(diff[:,minid]**2)))<0.95:
          continue
        else:
          anchor_id = self.dist2Id(mindist)
          temp = np.empty([1,5])
          #temp[0,0:3]=lane[:,minid]
          temp[0,0:3]=center2[:,n]+mindist*sideways[n,:]
          temp[0,3]=n
          temp[0,4]=anchor_id
          anchors = np.r_[anchors, temp]
          if not np.isnan(id):
            id = anchor_id
      return lane,id, anchors
          
        

    def runBatch(self, vid_name, gps_dat, gps_times1, gps_times2, frames, start_frame, final_frame, lanes, tr1,Pid, split_num, cam_num, params):
        if self.visualize:
          print 'warning: reading videos in labeller...'
          cap = cv2.VideoCapture(vid_name)
        cam = params['cam'][cam_num-1]#self.cam[cam_num - 1]
        lidar_height = params['lidar']['height']
        T_from_l_to_i = params['lidar']['T_from_l_to_i']
        T_from_i_to_l = np.linalg.inv(T_from_l_to_i) 
        starting_point = 4#12
        meters_per_point = 80#24#12#6
        points_fwd = 2#6#12
        starting_point2 = 15#12
        meters_per_point2 = 5#24#12#6
        points_fwd2 = 15#6#12
        scan_range = starting_point + (points_fwd-1)*meters_per_point
        seconds_ahead=120
        output_num = 0
        batchSize = frames.shape[0]
        labels= np.zeros([self.labelh, self.labelw, 2, batchSize],dtype=np.uint8,order='F')
        reg_labels= np.zeros([self.labelh, self.labelw, self.labeld, batchSize],dtype='f4',order='F')
        weight_labels= np.ones([self.labelh, self.labelw, 1, batchSize],dtype='f4',order='F')
        labels_3d= []
        trajectory_3d= []
        count = 0
        #print 'reading labels... ',
        labelling_time=0
        imgs = [] # raw images
        trs = [] # transformations wrt 0-th frame
        fnum1s = []
        timestamps = []
        lane_keys = copy.copy(lanes.keys())
        for key in lanes.keys():
          if key[0:4]!='lane':
            lane_keys.remove(key)
        for idx in xrange(batchSize):
            frame = frames[idx]
            video_frame = frames[idx]
            #fnum2 =frame*10+split_num-1 # global video frame. if split0, *10+9; if split1, *10+0; if split 2, *10+1 .... if split9, *10+8
            fnum2 =frame # global video frame. if split0, *10+9; if split1, *10+0; if split 2, *10+1 .... if split9, *10+8
            #if cam_num>2:
            #  fnum2 *=2 # wideview cams have half the framerate
            t = gps_times2[fnum2] # time stamp for the current video frame (same as gps_mark2)
            fnum1 = Idfromt(gps_times1,t) # corresponding frame in gps_mark1
            if self.new_distort:
              T = self.Ps['T'][Pid[idx]]
              P=self.Ps['M_'+str(cam_num)][Pid[idx]]
            else:
              T = np.eye(4)
              P = self.Ps[int(Pid[idx])]
            if frame < start_frame or (final_frame != -1 and frame >= final_frame):
                continue

            # car trajectory in current camera frame
            local_pts = MapPos(tr1[fnum1:fnum1+290,0:3,3], tr1[fnum1,:,:], cam, T_from_i_to_l)
            local_pts[1,:]+=lidar_height # subtract height to get point on ground
            # transform according to real-world distortion
            local_pts = np.vstack((local_pts, np.ones((1,local_pts.shape[1]))))
            local_pts = np.dot(T, local_pts)[0:3,:]
            # pick start and end point frame ids
            ids = np.where(np.logical_and(gps_times1>t-seconds_ahead*1000000, gps_times1<t+seconds_ahead*1000000))[0]
            ids = range(ids[0], ids[-1]+1)
            # ids for computing lateral ordering of lanes.
            anchor_ids = (self.zDistances(local_pts[2,:], fnum1, starting_point2, meters_per_point2, points_fwd2))
            velocities = gps_dat[anchor_ids,4:7]
            velocities[:,[0, 1]] = velocities[:,[1, 0]]
            vel_start = ENU2IMUQ50(np.transpose(velocities), gps_dat[0,:])
            vel_current = ENU2IMUQ50(np.transpose(velocities), gps_dat[fnum1,:])
            sideways_start = np.cross(vel_start.transpose(), tr1[anchor_ids,0:3,2], axisa=1, axisb=1, axisc=1) # sideways vector wrt starting imu frame
            sideways_start /= np.sqrt((sideways_start ** 2).sum(1))[...,np.newaxis]
            sideways_curr = np.transpose(MapVec( sideways_start, tr1[fnum1,:,:], cam, T_from_i_to_l))
            center = MapPosTrajectory(tr1[ids,:,:], tr1[fnum1,:,:], cam, T_from_i_to_l,height=lidar_height)
            center2 = MapPosTrajectory(tr1[anchor_ids,:,:], tr1[fnum1,:,:], cam, T_from_i_to_l,height=lidar_height)
            temp_label = np.zeros([self.labelh, self.labelw])
            if self.predict_depth:
              temp_reg1 = np.zeros([self.labelh, self.labelw, self.labeld/2],dtype='f4')
              temp_reg2 = np.zeros([self.labelh, self.labelw, self.labeld/2],dtype='f4')
            else:
              temp_reg = np.zeros([self.labelh, self.labelw, self.labeld],dtype='f4')
            temp_weight = np.ones([self.labelh, self.labelw, 1],dtype='f4')
            Lane3d = {'pts':[],'id':[],'anchors':np.empty([0,5])}
            Trajectory = {'center':center2,'sideways':sideways_curr}
            for l in range(lanes['num_lanes']):
              #lane_key = 'lane'+str(l)
              lane_key = lane_keys[l]
              lane = lanes[lane_key]
              # find the appropriate portion on the lane (close to the position of car, in front of camera, etc)
              # find the closest point on the lane to the two end-points on the trajectory of car. ideally this should be done before-hand to increase efficiency.
              local_displacement = np.dot(np.linalg.inv(tr1[fnum1,:,:]), np.concatenate((lane, np.ones([lane.shape[0], 1])), axis=1)
.T).T
              lane = lane[np.argsort(local_displacement[:,0]),:]
              dist_near = np.sum((lane-tr1[ids[0],0:3,3])**2, axis=1) # find distances of lane to current 'near' position.
              dist_far = np.sum((lane-tr1[ids[-1],0:3,3])**2, axis=1) # find distances of lane to current 'far' position.
              dist_self = np.sum((lane-tr1[fnum1,0:3,3])**2, axis=1) # find distances of lane to current self position.
              dist_mask = np.where(dist_self<=(scan_range**2))[0]# only consider points to be valid within scan_range from the 'near' position
              if len(dist_mask)==0:
                continue
              nearid = np.argmin(dist_near[dist_mask]) # for those valid points, find the one closet to 'near' position.
              farid = np.argmin(dist_far[dist_mask])  #and far position
              lids = range(dist_mask[nearid], dist_mask[farid]+1) # convert back to global id and make it into a consecutive list.
              lane3d = MapPos(lane[lids,:], tr1[fnum1,:,:], cam,T_from_i_to_l) # lane markings in current camera frame
              
              if np.all(lane3d[2,:]<=0):
                continue
              lane3d = lane3d[:,lane3d[2,:]>0] # make sure in front of camera
              depths = lane3d[2,:]
              # project into 2d image
              (c, J)  = cv2.projectPoints(lane3d[0:3,:].transpose(), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cam['KK'], cam['distort'])
              # need to define lane id. If necessary split current lane based on lateral distance. 
              lane3d,lane_id, anchors=self.setLaneIDs(lane3d, center2, sideways_curr)
              Lane3d['pts'].append(lane3d)
              Lane3d['id'].append(lane_id)
              Lane3d['anchors'] = np.r_[Lane3d['anchors'],anchors]
              c= warpPoints(P, c[:,0,:].transpose()[0:2,:])
              # scale down to the size of the label mask 
              labelpix = np.transpose(np.round(c*self.label_scale))
              # scale down to the size of the actual image 
              #imgpix = c#*self.img_scale  #scaling is done in producer node for now
              imgpix = cropScaleLabels(c, np.array([[107],[0]]), np.array([[self.imwidth/1067.0],[self.imheight/800.0]])) 
              # find unique indices to be marked in the label mask
              #lu = np.ascontiguousarray(labelpix).view(np.dtype((np.void, labelpix.dtype.itemsize * labelpix.shape[1])))
              #_, l_idx = np.unique(lu, return_index=True)
              #l_idx = np.sort(l_idx) 
              labelpix = (np.transpose(labelpix)).astype('i4')
              # draw labels on temp masks
              if self.visualize: # if need to visualize, make the lines more colorful!
                mask_color = 1#l+1
              else:
                mask_color=1
              for ii in range(1,imgpix.shape[1]-1):
                ip = ii-1
                ic = ii
                xp = labelpix[0,ip]
                yp = labelpix[1,ip]
                xc = labelpix[0,ic]
                yc = labelpix[1,ic]
               
                if np.abs(xp-xc)>1 or np.abs(yp-yc)>1:
                  x1 = xp
                  y1 = yp
                else:
                  x1 = xc
                  y1 = yc
                x2 = xc
                y2 = yc 
                if yc>-1 and yc<self.labelh and xc>-1 and xc<self.labelw:# and np.abs(yp-yc)<5:
                  # only update info for the first pt if nothing has been drawn for this grid. otherwise keep the first point and update the second point.
                  if temp_label[yc,xc]<1:
                    regx1 = imgpix[0,ip]
                    regy1 = imgpix[1,ip]
                    depth1 = depths[ip]
                  else:
                    if self.predict_depth:
                      regx1 = float(temp_reg1[yc,xc,0])
                      regy1 = float(temp_reg1[yc,xc,1])
                      depth1 = float(temp_reg2[yc,xc,1])
                    else:
                      regx1 = float(temp_reg[yc,xc,0])
                      regy1 = float(temp_reg[yc,xc,1])
                  regx2 = imgpix[0,ii+1]
                  regy2 = imgpix[1,ii+1]
                  depth2 = depths[ii+1]
                  if self.predict_depth:
                    cv2.line(temp_reg1, (x1,y1), (x2,y2) , [regx1, regy1, regx2], thickness=1 )
                    cv2.line(temp_reg2, (x1,y1), (x2,y2), [regy2, depth1, depth2], thickness=1 )
                  else:
                    cv2.line(temp_reg, (x1,y1), (x2,y2) , [regx1,regy1,regx2,regy2], thickness=1 )
                  # draw mask label
                  cv2.line(temp_label, (x1, y1), (x2, y2), mask_color, thickness=1 )
            
            # fill temp masks into actual batch labels
            labels[:,:,0,idx] = temp_label
            if self.predict_depth:
              reg_labels[:,:,0:3,idx] = temp_reg1
              reg_labels[:,:,3:,idx] = temp_reg2    
            else:
              reg_labels[:,:,:,idx] = temp_reg    
            weight_labels[:,:,:,idx] = temp_weight
            labels_3d.append(Lane3d)
            trajectory_3d.append(Trajectory)
            # code to visualize the read labels. Not run during actual training/testing
            if self.visualize:
              mask_scale = 8#opts.bb_mask_size/opts.mask_dim                                                        
              ms2 = mask_scale/2 
              cap.set(cv.CV_CAP_PROP_POS_FRAMES, video_frame)
              success, img = cap.read()
              img = img.astype('f4')
              reg_label = reg_labels[:,:,:,idx]
              #cv2.putText(img, str(global_frame), (100,100), cv2.FONT_HERSHEY_PLAIN, 2.0, self.colors[0],thickness=2)
              '''
              for ii in xrange(temp_label.shape[0]):
                for jj in xrange(temp_label.shape[1]):
                    xx = ii*mask_scale
                    yy = jj*mask_scale
                    #img[xx-ms2:xx+ms2,yy-ms2:yy+ms2,1]+=temp_label[ii,jj]*255
                    if temp_label[ii,jj]>0.5:
                      #cv2.putText(img, str(int(temp_label[ii,jj]-1)), (reg_label[ii,jj,0],reg_label[ii,jj,1]), cv2.FONT_HERSHEY_PLAIN, 1.0, self.colors[int(temp_label[ii,jj]-1)%len(self.colors)],thickness=1)
                      
                      if self.predict_depth:
                        cv2.line(img, (reg_label[ii,jj,0]*self.img_scale[0],reg_label[ii,jj,1]*self.img_scale[0]), (reg_label[ii,jj,2]*self.img_scale[1], reg_label[ii,jj,3]*self.img_scale[1]), dist2color((reg_label[ii,jj,4]+reg_label[ii,jj,5])/2).tolist(), thickness=2 )
                      else:
                        cv2.line(img, (reg_label[ii,jj,0],reg_label[ii,jj,1]), (reg_label[ii,jj,2], reg_label[ii,jj,3]), self.colors[int(temp_label[ii,jj]-1)%len(self.colors)], thickness=2 )
              '''
              imgs.append(img)
              trs.append(tr1[fnum1,:,:])
              fnum1s.append(fnum1)
              timestamps.append(t)
              #cv2.imwrite('/scr/twangcat/lane_detect_results/test2/label_'+str(self.count)+'.png', np.clip(img, 0,255).astype('u1'))
            self.count+=1
        labels[:,:,1,:] = 1-labels[:,:,0,:]
        # push a batch of data to the data queue
        self.q.put([imgs, labels.astype(np.float32),reg_labels,weight_labels,Pid, cam, labels_3d, trajectory_3d, trs,params,fnum1s, os.path.split(vid_name)[-1],timestamps])

    def runLabelling(self, batch):
        f = batch[0]
        #cam_num = int(f[-5])
        cam_num = 601+1
        splitidx = string.index(f,'split_')
        split_num = int(f[splitidx+6])
        if split_num==0:
          split_num=10
        frames = batch[1] # frame numbers
        Pid = batch[2] # type of transformation
        path, fname = os.path.split(f)
        fname = fname[8:] # remove 'split_?'
        args = parse_args(path, fname)
        prefix = path + '/' + fname
        



        params = args['params'] 
        cam = params['cam'][cam_num-1]
        self.label_scale = np.array([[float(self.labelw) / cam['width']], [float(self.labelh) / cam['height']]])
        self.img_scale = np.array([[float(self.imwidth) / cam['width']], [float(self.imheight) / cam['height']]])
        if os.path.isfile(args['gps_mark2']):
          gps_key1='gps_mark1'
          gps_key2='gps_mark2'
          postfix_len = 13
        else:
          gps_key1='gps'
          gps_key2='gps'
          postfix_len=8
        
        # gps_mark2 
        gps_filename2= args[gps_key2]
        if not (gps_filename2 in self.gps_values2): # if haven't read this gps file before, cache it in dict.
          gps_reader2 = GPSReader(gps_filename2)
          self.gps_values2[gps_filename2] = gps_reader2.getNumericData()
        gps_data2 = self.gps_values2[gps_filename2]
        gps_times2 = utc_from_gps_log_all(gps_data2)
    
        # gps_mark1
        gps_filename1= args[gps_key1]
        if not (gps_filename1 in self.gps_values1): # if haven't read this gps file before, cache it in dict.
          gps_reader1 = GPSReader(gps_filename1)
          self.gps_values1[gps_filename1] = gps_reader1.getNumericData()
        gps_data1 = self.gps_values1[gps_filename1]
        tr1 = IMUTransforms(gps_data1)
        gps_times1 = utc_from_gps_log_all(gps_data1)

        prefix = gps_filename2[0:-postfix_len]
        
        #lane_filename = prefix+'_multilane_points_done.npz'
        lane_filename = prefix+'_multilane_points_planar_done.npz'
        if not (lane_filename in self.lane_values):
          self.lane_values[lane_filename] = np.load(lane_filename)
        lanes = self.lane_values[lane_filename] # these values are alread pre-computed and saved, now just read it from dictionary

        start_frame = 0 #frames to skip  #int(sys.argv[3]) if len(sys.argv) > 3 else 0
        final_frame = -1  #int(sys.argv[4]) if len(sys.argv) > 4 else -1
        self.runBatch(f, gps_data1, gps_times1, gps_times2, frames, start_frame, final_frame, lanes, tr1, Pid, split_num, cam_num, params)



if __name__ == '__main__':
  label_reader = MultilaneLabelReader(400, 0, 3, 640, 480, label_dim = [80,60], readVideo = True,predict_depth=True)
  fid = open('/scail/group/deeplearning/driving_data/twangcat/schedules/q50_test_schedule_4-2-14-monterey-17S_a2.avi_96.pickle')
  batches = pickle.load(fid)
  fid.close()
  label_reader.start()
  for b in batches:
    label_reader.push_batch(b)

