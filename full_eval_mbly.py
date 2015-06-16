#!/usr/bin/env python
import sys
import numpy as np
import os
import cPickle as pickle
import copy
import re
import pdb
from laneSegMerge import *
from scipy.io import savemat
import cv,cv2
import string
from CameraReprojection import recover3d
from GPSReprojection import MapPos, MapPosInv
# import the protobuf format defined in caffe
sys.path.append('/afs/cs.stanford.edu/u/twangcat/scratch/caffe/src/caffe/proto')
import caffe_pb2
from multilane_label_reader import MultilaneLabelReader
from multilane_label_evaluator import MultilaneLabelEvaluator


sys.path.append('/afs/cs.stanford.edu/u/twangcat/scratch/sail-car-log/process')
from MblyTransforms import MblyLoader, T_from_mbly_to_lidar
from ArgParser import parse_args
from transformations import euler_from_matrix, euler_matrix
from LidarTransforms import R_to_c_from_l

blue = np.array([255,0,0])
green = np.array([0,255,0])
red = np.array([0,0,255])
colors = [(255,255,0),(255,0,255),(0,255,255),(128,128,255),(128,255,128),(255,128,128),(128,128,0),(128,0,128),(0,128,128),(0,128,255),(0,255,128),(128,0,255),(128,255,0),(255,0,128),(255,128,0),(255,255,128),(128,255,255),(255,128,255),(128,0,0),(0,128,0),(0,0,128)]
label_color = (50,50,50)
mbly_rot = [0.0, -0.005, -0.006]
mbly_T = [5.4, 0.0, -1.9]

def cropScaleLabels(labels, upper_left, scale):
  # scale the labels linearly according to the upper_left offset (x,y)
  # and the scales (x_scale, y_scale)
  return (labels-upper_left)*scale


def recoverScaledPred(pred, upper_left, scale):
  # recover the predictions linearly according to the upper_left offset (x,y)
  # and the scales (x_scale, y_scale)
  return (pred/scale+upper_left)

 
  
def check_for_nans(W):
    assert not np.any(np.isnan(W.get()))


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


def addLaneToCam(lane_wrt_mbly,args):
        mbly_R = euler_matrix(*mbly_rot)[:3, :3]
        mbly_lane_pts_3d = []
        for i in xrange(lane_wrt_mbly.shape[0]):
            lane_pts_wrt_mbly = getLanePointsFromModel(lane_wrt_mbly[i, :])
            if lane_pts_wrt_mbly is None:
                continue
            
            pts_wrt_cam = xformMblyToCam(lane_pts_wrt_mbly, args, mbly_T,mbly_R)
            mbly_lane_pts_3d.append(pts_wrt_cam)
        return mbly_lane_pts_3d


def xformMblyToCam(mbly_data, args, T, R):
    """ Projects mobileye points into the camera's frame
        Args: mbly_data, the output from loadMblyWindow
              args, the output from parse_args
    """
    params = args['params']
    cam_num = 601
    cam = params['cam'][cam_num]

    # Move points to the lidar FoR
    pts_wrt_lidar = T_from_mbly_to_lidar(mbly_data, T, R)

    # Move the points to the cam FoR
    pts_wrt_cam = pts_wrt_lidar +\
      cam['displacement_from_l_to_c_in_lidar_frame']
    pts_wrt_cam = np.dot(R_to_c_from_l(cam), pts_wrt_cam.transpose()).T
    return pts_wrt_cam



def getLanePointsFromModel(lane_wrt_mbly):
        view_range = lane_wrt_mbly[6]
        if view_range == 0:
            return None
        num_pts = view_range * 3
        X = np.linspace(0, view_range, num=num_pts)
        # from model: Y = C3*X^3 + C2*X^2 + C1*X + C0.
        X = np.vstack((np.ones(X.shape), X, np.power(X, 2), np.power(X, 3)))
        # Mbly uses Y-right as positive, we use Y-left as positive
        Y = -1 * np.dot(lane_wrt_mbly[:4], X)
        lane_pts_wrt_mbly = np.vstack((X[1, :], Y, np.zeros((1, num_pts)))).T
        return lane_pts_wrt_mbly


def mblyLaneAsNp( mbly_lane):
        """Turns a mobileye lane into a numpy array with format:
        [C0, C1, C2, C3, lane_id, lane_type, view_range]

        Y = C3*X^3 + C2*X^2 + C1*X + C0.
        X is longitudinal distance from camera (positive right!)
        Y is lateral distance from camera

        lane_id is between -2 and 2, with -2 being the farthest left,
        and 2 being the farthest right lane. There is no 0 id.

        """
        lanes_wrt_mbly = []
        for l in mbly_lane:
            lane_wrt_mbly = [l.C0, l.C1, l.C2, l.C3, l.lane_id, l.lane_type, \
                             l.view_range]
            lanes_wrt_mbly.append(lane_wrt_mbly)
        return np.array(lanes_wrt_mbly)




def main(args=None):
    # parse command-line args
    from optparse import OptionParser
    bpv=caffe_pb2.BlobProtoVector()
    parser = OptionParser()
    parser.add_option('--date', dest='date', default='',
                      help="Date on which video is taken, format: (d)d-mm-yy-location")
    parser.add_option('--video_name', dest='video_name', default='',
                      help="Name of video to test")
    parser.add_option('--batch_size', dest='batch_size', default=5,
                      help="Batch size")
    parser.add_option('--split_num', dest='split_num', default=10,
                      help="video split number")
    (opts,args)=parser.parse_args(args)

    unique_name = opts.date+'-'+opts.video_name
    split_num=opts.split_num
    #schedule_file = '/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_test_schedule_'+unique_name+'.avi_batch'+str(opts.batch_size)+'_split'+str(split_num)+'.txt'
    #schedule_file = '/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_test_schedule_'+unique_name+'.avi_batch'+str(opts.batch_size)+'.txt'
    schedule_file = '/scail/group/deeplearning/driving_data/twangcat/schedules/q50_HDR_multilane_planar_test_schedule_4-17-15-280_batch5.txt'
    print 'loading schedule file: '+ schedule_file
    if os.path.isfile(schedule_file):
      sid = open(schedule_file, 'r')
    else:
      print 'file not found!'
      return 1
    num_batches = 650#sum(1 for line in sid)
    print 'total of '+str(num_batches)+' batches'
    draw_raw = True
    depth_only = False#True
    imwidth = 640
    imheight = 480
    # label dimensions
    #label_width = 160
    #label_height = 120
    label_width = 80
    label_height = 60
    reg_scale = np.array([[640/1067.0,480/800.0]])
    upper_left = np.array([[107,0]])
    labelReader = MultilaneLabelReader(buffer_size=10, imdepth=3, imwidth=imwidth, imheight=imheight, label_dim = [label_width,label_height], predict_depth = True, readVideo=True)
    evaluator = MultilaneLabelEvaluator(mbly=True)
    batch_num =0
    starting_batch=1
    with open(schedule_file, 'r') as sid:
      for line in sid:
        if batch_num>=starting_batch:
          words = string.split(line, ',')
          batch = []
          batch.append(words[0])
          batch.append(np.fromstring(words[1], dtype=int, sep=' '))
          batch.append(np.fromstring(words[2], dtype=float, sep=' '))
          labelReader.push_batch(batch)
        batch_num+=1
    labelReader.start()
    #evaluator = MultilaneLabelEvaluator()
    scaling = imwidth/label_width # ratio of orig image dimension to label dimension
    
    patch_size = 32 # size of central patch
    count = 0
    lowThresh = 0.5
    # read a single file first to find out the dimensions
    #proto_fname = '/deep/group/driving_data/twangcat/caffe_results/proto/raw_outputs/'+unique_name+'_split'+str(split_num)+'_batch0.proto'
    proto_fname = '/deep/group/driving_data/twangcat/caffe_results/proto/raw_outputs/'+unique_name+'_batch0.proto'
    fid = open(proto_fname,'r')
    bpv.ParseFromString(fid.read())
    fid.close()
    print bpv.blobs[0].shape
    grid_length = bpv.blobs[0].shape.dim[1]
    grid_dim = int(np.sqrt(grid_length)) # number of quadrants inside the central patch, along 1 dimension
    batch_size = bpv.blobs[0].shape.dim[0]
    assert batch_size==opts.batch_size
    quad_height = bpv.blobs[0].shape.dim[2]
    quad_width = bpv.blobs[0].shape.dim[3]
    num_regressions = bpv.blobs[1].shape.dim[1]/bpv.blobs[0].shape.dim[1]
    if not depth_only:
      # matrix of RF offsets. to be added to regression results
      x_adj = (np.floor(np.arange(label_width)/grid_dim)*grid_dim+grid_dim/2)*imwidth/label_width
      y_adj = (np.floor(np.arange(label_height)/grid_dim)*grid_dim+grid_dim/2)*imheight/label_height
      y_adj = np.array([y_adj]).transpose()
      adj = np.zeros([2, quad_height*grid_dim, quad_width*grid_dim])
      for qy in range(quad_height):
        for qx in range(quad_width):
          adj[0,qy*grid_dim:(qy+1)*grid_dim, qx*grid_dim:(qx+1)*grid_dim] = (qx*grid_dim + grid_dim / 2) * scaling;
          adj[1,qy*grid_dim:(qy+1)*grid_dim, qx*grid_dim:(qx+1)*grid_dim] = (qy*grid_dim + grid_dim / 2) * scaling;
    # predictions of normal flat shape
    All_Lane_Pred = []
    Global_Frames = [] 
    for it in range(starting_batch,num_batches):

      imgs,labels,reg_labels,weights,Pid,cam,labels_3d,trajectory,trs,params,global_frames,videoname,timestamps = labelReader.pop_batch()
      foldername = '/deep/group/driving_data/q50_data/4-17-15-280/' 
      videoname = videoname.replace('split_0_','')
      args = parse_args(foldername, videoname)
      mbly_loader = MblyLoader(args)



      labels = np.transpose(labels, [3,2,1,0]) #[numimgs, channels, x, y]
      reg_labels = np.transpose(reg_labels, [3,2,1,0])
      #reg_labels[:,0:4,:,:]*=reg_scale
      T_from_l_to_i = params['lidar']['T_from_l_to_i']
      T_from_i_to_l = np.linalg.inv(T_from_l_to_i)
      pix_label_all = bpv.blobs[2]
      pix_label_all = np.array(bpv.blobs[2].data, order='C')
      pix_label_all = np.reshape(pix_label_all, [batch_size,grid_length,quad_height,quad_width], order='C')
      reg_label_all = bpv.blobs[3]
      reg_label_all = np.array(bpv.blobs[3].data, order='C')
      reg_label_all = np.reshape(reg_label_all, [batch_size,grid_length*num_regressions,quad_height,quad_width], order='C')
      # turn the z-stacked output into normal flat shape.
      grid_cnt = 0
      pix_label_full = np.zeros([batch_size, 1, quad_height*grid_dim, quad_width*grid_dim], order='C')
      reg_label_full = np.zeros([batch_size, num_regressions, quad_height*grid_dim, quad_width*grid_dim], order='C')
      for qy in range(quad_height):
        for qx in range(quad_width):
          pix_label_full[:, 0, qy*grid_dim:(qy+1)*grid_dim, qx*grid_dim:(qx+1)*grid_dim] = np.reshape(pix_label_all[:, :, qy,qx], [batch_size, grid_dim,grid_dim], order='C')
          reg_label_full[:, :, qy*grid_dim:(qy+1)*grid_dim, qx*grid_dim:(qx+1)*grid_dim] = np.reshape(reg_label_all[:, :, qy,qx], [batch_size, num_regressions, grid_dim, grid_dim],order='C')
          grid_cnt+=1

      if not depth_only:
        reg_label_full[:,[0,2],:,:] +=x_adj
        reg_label_full[:,[1,3],:,:] +=y_adj
      pix_label_full = np.transpose(pix_label_full, [0,1,3,2]) #[numimgs, channels, x, y]
      reg_label_full = np.transpose(reg_label_full, [0,1,3,2])
      #mask_scale = opts.bb_mask_size/opts.mask_dim
      #      ms2 = mask_scale/2
      for i in range(batch_size):
                img = imgs[i].astype('u1')
                if draw_raw:  
                  img_raw = np.array(img)
                topDown = np.zeros([480,480,3], dtype='u1') 
                label = labels[i,:,:,:]
                #reg_label = reg_labels[i,:,:,:]
                #reg_pred = reg_pred_full[i,:,:,:]
                mbly_lanes = mbly_loader.loadLane(timestamps[i])
                lane_wrt_mbly = mblyLaneAsNp(mbly_lanes)
                lane_pred_3d = addLaneToCam(lane_wrt_mbly,args)
                label2 = pix_label_full[i,:,:,:]
                reg_label = reg_label_full[i,:,:,:]
                print '**************************************'
                if len(trajectory)>0:
                  Trajectory=trajectory[i]
                if len(labels_3d)>0:
                  Lane3d = labels_3d[i]
                  for lane3d in Lane3d['pts']:
                    for ll in range(1, lane3d.shape[1]-1):
                      ip = ll-1
                      ic = ll
                      cv2.line(topDown, (int(lane3d[0,ip]*6+240), int(479-(lane3d[2,ip]-4)*6+6)) , (int(lane3d[0, ll+1]*6+240), int(479-(lane3d[2,ll+1]-4)*6+6)) ,label_color,1)
                #Pos = np.reshape(recover3d(pix, cam, Z=zs), [num_pts, 6], order='C')
                  
                #(c1, J)  = cv2.projectPoints(Pos[:,0:3].astype('f8'), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cam['KK'], cam['distort'])
                #(c2, J)  = cv2.projectPoints(Pos[:,3:6].astype('f8'), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cam['KK'], cam['distort'])
                #all_reg[:,0:2]=np.squeeze(cropScaleLabels(c1, upper_left, reg_scale))
                #all_reg[:,2:4]=np.squeeze(cropScaleLabels(c2, upper_left, reg_scale))
                #for rid in range(laneids.shape[0]):
                  #if laneids[rid]>=0:
                    #cv2.line(img, (all_reg[rid, 0],all_reg[rid,1]), (all_reg[rid,2], all_reg[rid,3]), colors[laneids[rid]], thickness=2)
                    #cv2.line(topDown, (int(Pos[rid,0]*6+240), int(479-(Pos[rid,2]-4)*6+6)) , (int(Pos[rid,3]*6+240), int(479-(Pos[rid,5]-4)*6+6)) ,colors[laneids[rid]],1)
                # group segments belonging to the same lane together 
                #unique_laneids = np.sort(np.unique(laneids))
                #lane_pred_3d = []
                #lane_pred_frame0 = [] # 3d lane pts wrt frame 0
                lane_pred =[]
                lane_conf = [1]*len(lane_pred_3d)
                lane_ids = range(len(lane_pred_3d))
                for lane_3d in lane_pred_3d:
                  (c, J)  = cv2.projectPoints(lane_3d, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cam['KK'], cam['distort'])
                  lane_pred.append(np.squeeze(cropScaleLabels(c, upper_left, reg_scale)))
                #lane_ids = []
                #maxNumLanes=4
                #for id in unique_laneids:
                #  if id !=-1 and id<maxNumLanes:
                #    npts = np.sum(laneids==id)
                #    r = np.reshape(all_reg[laneids==id, 0:4], [npts*2,2],order='C')
                #    p = np.reshape(Pos[laneids==id,:], [npts*2,3],order='C')
                #    c = candidates[laneids==id,:]
                
                # draw each lane
                for lid in range(len(lane_conf)):
                  l = lane_pred[lid]
                  l3d = lane_pred_3d[lid]
                  for rid in range(1,l3d.shape[0]):
                    cv2.line(img, (int(l[rid-1, 0]),int(l[rid-1,1])), (int(l[rid,0]), int(l[rid,1])), colors[lane_ids[lid]%(len(colors))], thickness=2)
                    cv2.line(topDown, (int(l3d[rid-1,0]*6+240), int(479-(l3d[rid-1,2]-4)*6+6)) , (int(l3d[rid,0]*6+240), int(479-(l3d[rid,2]-4)*6+6)) ,colors[lane_ids[lid]%len(colors)],1)
                    
                pred_lanes = {'pts':lane_pred_3d, 'conf':lane_conf, 'id':lane_ids}
                evaluator.accEvalCounts(pred_lanes,Lane3d,Trajectory)
                topDown = evaluator.drawOnImage(topDown)
                #cv2.putText(img, str(count)+' from '+videoname+' '+str(global_frames[i]), (10,20), cv2.FONT_HERSHEY_PLAIN, 1.6, colors[-1],thickness=1)
                imgname = '/scr/twangcat/caffenet_results/test/'+str(count)+'.png' 
                cv2.imwrite(imgname, np.concatenate((np.clip(img,0,255).astype(np.uint8),topDown), axis=1))
                #if save_outputs:
                #  allMaskLabel[vis_count-1,:,:] = label[0,:,:] 
                #  allRegLabel[vis_count-1,:,:,:] = reg_label
                #  allMaskPred[vis_count-1,:,:] = pred[0,:,:]
                #  allRegPred[vis_count-1,:,:,:] = reg_pred
                count +=1
                All_Lane_Pred.append(lane_pred_3d)
                #Global_Frames.append(global_frames[i])
      print it
      it+=1
    print 'saving...'
    tp=evaluator.tp
    fp=evaluator.fp_cnt
    fn=evaluator.fn_cnt
    pd=evaluator.pd
    lat_err=evaluator.lat_err
    laneResults={'tp':tp,'fp':fp,'fn':fn,'pd':pd,'lat_err':lat_err}
    acc_name = '/deep/group/driving_data/twangcat/caffe_results/lane_acc/'+unique_name+'_mbly.mat' 
    savemat(acc_name,laneResults)
    aa = {'Lanes':All_Lane_Pred, 'Frames':Global_Frames}
    lane_outname = '/deep/group/driving_data/twangcat/caffe_results/lane_acc/'+unique_name+'_mbly_lanepred.pickle'
    #lane_outname = '/deep/group/driving_data/twangcat/caffe_results/lane_acc/'+unique_name+'_split'+str(split_num)+'_lanepred.pickle'
    fid = open(lane_outname, 'wb')
    pickle.dump(aa, fid)
    fid.close()

    '''
      if save_outputs: 
        allOutputs = dict()
        allOutputs['mask_pred'] = allMaskPred
        allOutputs['mask_label'] = allMaskLabel
        allOutputs['reg_pred'] = allRegPred
        allOutputs['reg_label'] = allRegLabel
        print 'saving...'
        pickle.dump(allOutputs, open('/scail/group/deeplearning/driving_data/twangcat/mask_reg_with_depth.pickle','wb'))
    '''
 


if __name__ == "__main__":
    main()
    

