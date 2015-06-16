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




blue = np.array([255,0,0])
green = np.array([0,255,0])
red = np.array([0,0,255])
colors = [(255,255,0),(255,0,255),(0,255,255),(128,128,255),(128,255,128),(255,128,128),(128,128,0),(128,0,128),(0,128,128),(0,128,255),(0,255,128),(128,0,255),(128,255,0),(255,0,128),(255,128,0),(255,255,128),(128,255,255),(255,128,255),(128,0,0),(0,128,0),(0,0,128)]
label_color = (50,50,50)

  
  
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


def main(args=None):
    # parse command-line args
    from optparse import OptionParser
    bpv=caffe_pb2.BlobProtoVector()
    parser = OptionParser()
    parser.add_option('--date', dest='date', default='',
                      help="Date on which video is taken, format: (d)d-mm-yy-location")
    parser.add_option('--video_name', dest='video_name', default='',
                      help="Name of video to test")
    parser.add_option('--batch_size', dest='batch_size', default=10,
                      help="Batch size")
    parser.add_option('--split_num', dest='split_num', default=10,
                      help="video split number")
    (opts,args)=parser.parse_args(args)

    unique_name = opts.date
    split_num=opts.split_num
    opts.batch_size = int(opts.batch_size)
    #schedule_file = '/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_test_schedule_'+unique_name+'.avi_batch'+str(opts.batch_size)+'_split'+str(split_num)+'.txt'
    schedule_file = '/scail/group/deeplearning/driving_data/twangcat/schedules/q50_multilane_planar_test_schedule_'+unique_name+'_batch'+str(opts.batch_size)+'_2cam.txt'
    print 'loading schedule file: '+ schedule_file
    if os.path.isfile(schedule_file):
      sid = open(schedule_file, 'r')
    else:
      print 'file not found!'
      return 1
    num_batches = sum(1 for line in sid)
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
    reg_scale = 0.5
    labelReader = MultilaneLabelReader(buffer_size=10, imdepth=3, imwidth=imwidth, imheight=imheight, label_dim = [label_width,label_height], predict_depth = True, readVideo=True)
    evaluator = MultilaneLabelEvaluator()
    batch_num =0
    starting_batch=0
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
    #lowThresh = 0.2
    lowThresh = 0.4
    # read a single file first to find out the dimensions
    #proto_fname = '/deep/group/driving_data/twangcat/caffe_results/proto/raw_outputs/'+unique_name+'_split'+str(split_num)+'_batch0.proto'
    proto_fname = '/deep/group/driving_data/twangcat/caffe_results/proto/raw_outputs/'+unique_name+'_batch0.proto'
    fid = open(proto_fname,'r')
    bpv.ParseFromString(fid.read())
    fid.close()
    print bpv.blobs[0].shape
    grid_length = bpv.blobs[0].shape.dim[1]
    grid_dim = int(np.sqrt(grid_length)) # number of quadrants inside the central patch, along 1 dimension
    batch_size = int(bpv.blobs[0].shape.dim[0])
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
      imgs,labels,reg_labels,weights,Pid,cam,labels_3d,trajectory,trs,params,global_frames,videoname = labelReader.pop_batch()
      labels = np.transpose(labels, [3,2,1,0]) #[numimgs, channels, x, y]
      reg_labels = np.transpose(reg_labels, [3,2,1,0])
      reg_labels[:,0:4,:,:]*=reg_scale
      T_from_l_to_i = params['lidar']['T_from_l_to_i']
      T_from_i_to_l = np.linalg.inv(T_from_l_to_i)
      #proto_fname = '/deep/group/driving_data/twangcat/caffe_results/proto/raw_outputs/old_iter72000/'+unique_name+'_batch'+str(it)+'.proto'
      #proto_fname = '/deep/group/driving_data/twangcat/caffe_results/proto/raw_outputs/'+unique_name+'_split'+str(split_num)+'_batch'+str(it)+'.proto'
      proto_fname = '/deep/group/driving_data/twangcat/caffe_results/proto/raw_outputs/'+unique_name+'_batch'+str(it)+'.proto'
      fid = open(proto_fname,'r')
      bpv.ParseFromString(fid.read())
      fid.close()
      pix_pred_all = np.array(bpv.blobs[0].data, order='C')
      pix_pred_all = np.reshape(pix_pred_all, [batch_size,grid_length,quad_height,quad_width], order='C')
      pix_pred_all = 1.0/ (1.0 + np.exp(-pix_pred_all))
      reg_pred_all = bpv.blobs[1]
      reg_pred_all = np.array(bpv.blobs[1].data, order='C')
      reg_pred_all = np.reshape(reg_pred_all, [batch_size,grid_length*num_regressions,quad_height,quad_width], order='C')
      pix_label_all = bpv.blobs[2]
      pix_label_all = np.array(bpv.blobs[2].data, order='C')
      pix_label_all = np.reshape(pix_label_all, [batch_size,grid_length,quad_height,quad_width], order='C')
      reg_label_all = bpv.blobs[3]
      reg_label_all = np.array(bpv.blobs[3].data, order='C')
      reg_label_all = np.reshape(reg_label_all, [batch_size,grid_length*num_regressions,quad_height,quad_width], order='C')
      # turn the z-stacked output into normal flat shape.
      grid_cnt = 0
      pix_pred_full = np.zeros([batch_size, 1, quad_height*grid_dim, quad_width*grid_dim], order='C')
      reg_pred_full = np.zeros([batch_size, num_regressions, quad_height*grid_dim, quad_width*grid_dim], order='C')
      pix_label_full = np.zeros([batch_size, 1, quad_height*grid_dim, quad_width*grid_dim], order='C')
      reg_label_full = np.zeros([batch_size, num_regressions, quad_height*grid_dim, quad_width*grid_dim], order='C')
      for qy in range(quad_height):
        for qx in range(quad_width):
          pix_pred_full[:, 0, qy*grid_dim:(qy+1)*grid_dim, qx*grid_dim:(qx+1)*grid_dim] = np.reshape(pix_pred_all[:, :, qy,qx], [batch_size, grid_dim,grid_dim], order='C')
          reg_pred_full[:, :, qy*grid_dim:(qy+1)*grid_dim, qx*grid_dim:(qx+1)*grid_dim] = np.reshape(reg_pred_all[:, :, qy,qx], [batch_size, num_regressions, grid_dim, grid_dim],order='C')
          pix_label_full[:, 0, qy*grid_dim:(qy+1)*grid_dim, qx*grid_dim:(qx+1)*grid_dim] = np.reshape(pix_label_all[:, :, qy,qx], [batch_size, grid_dim,grid_dim], order='C')
          reg_label_full[:, :, qy*grid_dim:(qy+1)*grid_dim, qx*grid_dim:(qx+1)*grid_dim] = np.reshape(reg_label_all[:, :, qy,qx], [batch_size, num_regressions, grid_dim, grid_dim],order='C')
          grid_cnt+=1

      if not depth_only:
        reg_pred_full[:,[0,2],:,:] +=x_adj
        reg_pred_full[:,[1,3],:,:] +=y_adj
        reg_label_full[:,[0,2],:,:] +=x_adj
        reg_label_full[:,[1,3],:,:] +=y_adj
      pix_pred_full = np.transpose(pix_pred_full, [0,1,3,2]) #[numimgs, channels, x, y]
      pix_label_full = np.transpose(pix_label_full, [0,1,3,2]) #[numimgs, channels, x, y]
      reg_pred_full = np.transpose(reg_pred_full, [0,1,3,2])
      reg_label_full = np.transpose(reg_label_full, [0,1,3,2])
      #mask_scale = opts.bb_mask_size/opts.mask_dim
      #      ms2 = mask_scale/2
      '''
      for i in xrange(batch_size):
        image = imgs[i].astype('u1')
        for y in range(label_height):
          for x in range(label_width):
            pix_pred = pix_pred_full[i, 0, y, x]
            # draw pixel label/pred
            x1 = 0 if x-0.5<0 else x-0.5
            y1 = 0 if y-0.5<0 else y-0.5
            w = scaling - (0.5-x if x<0.5 else 0) - (x1+scaling-imwidth if x1+scaling>imwidth else 0)
            h = scaling - (0.5-y if y<0.5 else 0) - (y1+scaling-imheight if y1+scaling>imheight else 0)
            roi = image[y1*scaling:y1*scaling+h, x1*scaling:x1*scaling+w,:]
            image[y1*scaling:y1*scaling+h, x1*scaling:x1*scaling+w,:] = green*pix_pred+roi*(1.0 - pix_pred)
            if pix_pred > thresh:
              # draw reg label/pred
              x_min = np.round(reg_pred_full[i, 0, y, x])
              y_min = np.round(reg_pred_full[i, 1, y, x])
              x_max = np.round(reg_pred_full[i, 2, y, x])
              y_max = np.round(reg_pred_full[i, 3, y, x])
              
              min_depth = reg_pred_full[i, 4, y, x]
              max_depth = reg_pred_full[i, 5, y, x]
              lineColor = dist2color((min_depth+max_depth)/2.);
              #draw label and predictions on image.
                
              cv2.line(image,(int(x_min), int(y_min)),(int(x_max), int(y_max)),lineColor.tolist(), thickness=2);
        cv2.imwrite('/scr/twangcat/caffenet_results/test/'+str(count)+'.png', np.clip(image, 0,255).astype('u1')) 
        count+=1
      '''
      for i in range(batch_size):
                img = imgs[i].astype('u1')
                if draw_raw:  
                  img_raw = np.array(img)
                topDown = np.zeros([480,480,3], dtype='u1') 
                pred = pix_pred_full[i,:,:,:] 
                label = labels[i,:,:,:]
                reg_label = reg_labels[i,:,:,:]
                reg_pred = reg_pred_full[i,:,:,:]
                label2 = pix_label_full[i,:,:,:]
                reg_label2 = reg_label_full[i,:,:,:]
                print '**************************************'
                for ii in xrange(label.shape[1]):                                                                
                  for jj in xrange(label.shape[2]): 
                    if label[0,ii,jj]>0.5:
                      cv2.line(img, (reg_label[0,ii,jj],reg_label[1,ii,jj]), (reg_label[2,ii,jj], reg_label[3,ii,jj]), label_color, thickness=1 )
                    if draw_raw:
                      pix_pred = pred[0, ii, jj]
                      x1 = 0 if ii-0.5<0 else ii-0.5
                      y1 = 0 if jj-0.5<0 else jj-0.5
                      w = scaling - (0.5-ii if ii<0.5 else 0) - (x1+scaling-imwidth if x1+scaling>imwidth else 0)
                      h = scaling - (0.5-jj if jj<0.5 else 0) - (y1+scaling-imheight if y1+scaling>imheight else 0)
                      roi = img_raw[y1*scaling:y1*scaling+h, x1*scaling:x1*scaling+w,:]
                      if depth_only:
                        if pix_pred > lowThresh:
                          maskcolor = dist2color(reg_pred_full[i,0,ii,jj]);
                          img_raw[y1*scaling:y1*scaling+h, x1*scaling:x1*scaling+w,:] = maskcolor*pix_pred+roi*(1.0 - pix_pred)
                      else: 
                        img_raw[y1*scaling:y1*scaling+h, x1*scaling:x1*scaling+w,:] = green*pix_pred+roi*(1.0 - pix_pred)
                        if pix_pred > lowThresh:
                          x_min = np.round(reg_pred_full[i, 0, ii, jj])
                          y_min = np.round(reg_pred_full[i, 1, ii, jj])
                          x_max = np.round(reg_pred_full[i, 2, ii, jj])
                          y_max = np.round(reg_pred_full[i, 3, ii, jj])
                          min_depth = reg_pred_full[i, 4, ii, jj]
                          max_depth = reg_pred_full[i, 5, ii, jj]
                          lineColor = dist2color((min_depth+max_depth)/2.);
                          cv2.line(img_raw,(int(x_min), int(y_min)),(int(x_max), int(y_max)),lineColor.tolist(), thickness=2);
                if 1==1:
                  if len(trajectory)>0:
                    Trajectory=trajectory[i]
                  if len(labels_3d)>0:
                    Lane3d = labels_3d[i]
                    for lane3d in Lane3d['pts']:
                      for ll in range(1, lane3d.shape[1]-1):
                        ip = ll-1
                        ic = ll
                        cv2.line(topDown, (int(lane3d[0,ip]*6+240), int(479-(lane3d[2,ip]-4)*6+6)) , (int(lane3d[0, ll+1]*6+240), int(479-(lane3d[2,ll+1]-4)*6+6)) ,label_color,1)
                  candidates = np.transpose(np.array(np.where(pred[0,:,:]>lowThresh))) # coord of candidates in label mask
                  confs = pred[0,candidates[:,0], candidates[:,1]] # candidate confidences
                  all_reg =np.transpose(reg_pred[:,candidates[:,0].astype('i4'), candidates[:,1].astype('i4')])
                  num_pts = all_reg.shape[0]
                  pix = np.reshape(all_reg[:,0:4], [num_pts*2,2],order='C')/reg_scale # all start and end pts of line segments in pixels
                  zs = np.reshape(all_reg[:,4:6], num_pts*2, order='C')
                  Pos = np.reshape(recover3d(pix, cam, Z=zs), [num_pts, 6], order='C')
                  # assign a lane label to each segments
                  #laneids = ransacCluster(Pos[:,[0,2,3,5]])
                  laneids = dbscanCluster(Pos[:,[0,2,3,5]])
                  #laneids, Pos2 = dbscanJLinkCluster(Pos[:,[0,2,3,5]])
                  #laneids, Pos = dbscanJLinkCluster(Pos)
                  
                  (c1, J)  = cv2.projectPoints(Pos[:,0:3].astype('f8'), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cam['KK'], cam['distort'])
                  (c2, J)  = cv2.projectPoints(Pos[:,3:6].astype('f8'), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cam['KK'], cam['distort'])
                  all_reg[:,0:2]=np.squeeze(c1*reg_scale)
                  all_reg[:,2:4]=np.squeeze(c2*reg_scale)




                  #for rid in range(laneids.shape[0]):
                    #if laneids[rid]>=0:
                      #cv2.line(img, (all_reg[rid, 0],all_reg[rid,1]), (all_reg[rid,2], all_reg[rid,3]), colors[laneids[rid]], thickness=2)
                      #cv2.line(topDown, (int(Pos[rid,0]*6+240), int(479-(Pos[rid,2]-4)*6+6)) , (int(Pos[rid,3]*6+240), int(479-(Pos[rid,5]-4)*6+6)) ,colors[laneids[rid]],1)
                  # group segments belonging to the same lane together 
                  unique_laneids = np.sort(np.unique(laneids))
                  lane_pred_3d = []
                  lane_pred_frame0 = [] # 3d lane pts wrt frame 0
                  lane_pred =[]
                  lane_conf = []
                  lane_ids = []
                  maxNumLanes=8
                  for id in unique_laneids:
                    if id !=-1 and id<maxNumLanes:
                      npts = np.sum(laneids==id)
                      r = np.reshape(all_reg[laneids==id, 0:4], [npts*2,2],order='C')
                      p = np.reshape(Pos[laneids==id,:], [npts*2,3],order='C')
                      c = candidates[laneids==id,:]
                      conf = np.sum(pred[0,c[:,0], c[:,1]])
                      sortidx = np.argsort(p[:,2])  # sort according to Z
                      lane_pred.append(r[sortidx,:])
                      lane_pred_3d.append(p[sortidx,:])
                      lane_pred_frame0.append(MapPosInv(p[sortidx,:].transpose(), trs[i], cam, T_from_i_to_l))
                      lane_conf.append(conf)
                      lane_ids.append(id)
                  # draw each lane
                  for lid in range(len(lane_conf)):
                    l = lane_pred[lid]
                    l3d = lane_pred_3d[lid]
                    for rid in range(1,l.shape[0]):
                      cv2.line(img, (int(l[rid-1, 0]),int(l[rid-1,1])), (int(l[rid,0]), int(l[rid,1])), colors[lane_ids[lid]%(len(colors))], thickness=2)
                      cv2.line(topDown, (int(l3d[rid-1,0]*6+240), int(479-(l3d[rid-1,2]-4)*6+6)) , (int(l3d[rid,0]*6+240), int(479-(l3d[rid,2]-4)*6+6)) ,colors[lane_ids[lid]%len(colors)],1)
                    
                  pred_lanes = {'pts':lane_pred_3d, 'conf':lane_conf, 'id':lane_ids}
                
                  evaluator.accEvalCounts(pred_lanes,Lane3d,Trajectory)
                  topDown = evaluator.drawOnImage(topDown)
                #cv2.putText(img, str(count)+' from '+videoname+' '+str(global_frames[i]), (10,20), cv2.FONT_HERSHEY_PLAIN, 1.6, colors[-1],thickness=1)
                imgname = '/scr/twangcat/caffenet_results/test/'+str(count)+'.png' 
                cv2.imwrite(imgname, np.concatenate((np.clip(img,0,255).astype(np.uint8),topDown), axis=1))
                if draw_raw:  
                  imgname_raw = '/scr/twangcat/caffenet_results/test/raw_'+str(count)+'.png' 
                  cv2.imwrite(imgname_raw, np.clip(img_raw,0,255).astype(np.uint8))
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
    acc_name = '/deep/group/driving_data/twangcat/caffe_results/lane_acc/'+unique_name+'.mat' 
    savemat(acc_name,laneResults)
    aa = {'Lanes':All_Lane_Pred, 'Frames':Global_Frames}
    lane_outname = '/deep/group/driving_data/twangcat/caffe_results/lane_acc/'+unique_name+'_lanepred.pickle'
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
    

