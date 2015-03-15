from transformations import euler_matrix
import numpy as np
from WGS84toENU import *
from numpy import array, dot, zeros, around, divide, ones
from LidarTransforms import *
from Q50_config import *
from cv2 import projectPoints
# only works for new Q50 data. Maps positions wrt imu_0 to cam_t
def MapPos(map_pos, imu_transforms_t, cam, T_from_i_to_l):
    pts_wrt_imu_0 = array(map_pos).transpose()
    pts_wrt_imu_0 = np.vstack((pts_wrt_imu_0, np.ones((1,pts_wrt_imu_0.shape[1]))))
    # transform points from imu_0 to imu_t
    pts_wrt_imu_t = np.dot( np.linalg.inv(imu_transforms_t), pts_wrt_imu_0)
    # transform points from imu_t to lidar_t
    pts_wrt_lidar_t = np.dot(T_from_i_to_l, pts_wrt_imu_t);
    # transform points from lidar_t to camera_t
    pts_wrt_camera_t = pts_wrt_lidar_t.transpose()[:, 0:3] + cam['displacement_from_l_to_c_in_lidar_frame']
    pts_wrt_camera_t = dot(R_to_c_from_l(cam), pts_wrt_camera_t.transpose())
    pts_wrt_camera_t = np.vstack((pts_wrt_camera_t,
        np.ones((1,pts_wrt_camera_t.shape[1]))))
    pts_wrt_camera_t = dot(cam['E'], pts_wrt_camera_t)
    pts_wrt_camera_t = pts_wrt_camera_t[0:3,:]
    return pts_wrt_camera_t
# only works for new Q50 data. Maps positions wrt cam_t to imu_0
def MapPosInv(pts_wrt_camera_t, imu_transforms_t, cam, T_from_i_to_l):
    if pts_wrt_camera_t.shape[0]==3:
      pts_wrt_camera_t = np.vstack((pts_wrt_camera_t,
        np.ones((1,pts_wrt_camera_t.shape[1]))))
    else:
      assert pts_wrt_camera_t.shape[0] == 4
    pts_wrt_camera_t = dot(np.linalg.inv(cam['E']), pts_wrt_camera_t)
    pts_wrt_camera_t = pts_wrt_camera_t[0:3,:]
    pts_wrt_camera_t = dot(np.linalg.inv(R_to_c_from_l(cam)), pts_wrt_camera_t).transpose()
    pts_wrt_lidar_t = (pts_wrt_camera_t - cam['displacement_from_l_to_c_in_lidar_frame']).transpose()
    pts_wrt_lidar_t = np.vstack((pts_wrt_lidar_t,
      np.ones((1,pts_wrt_lidar_t.shape[1]))))
    pts_wrt_imu_t = np.dot(np.linalg.inv(T_from_i_to_l), pts_wrt_lidar_t);
    pts_wrt_imu_0 = np.dot(imu_transforms_t, pts_wrt_imu_t)
    map_pos = pts_wrt_imu_0[0:3,:].transpose()
    return map_pos

# only works for new Q50 data. Maps vectors wrt imu_0 to cam_t
def MapVec(map_vec, imu_transforms_t, cam, T_from_i_to_l):
    vec_wrt_imu_0 = array(map_vec).transpose()
    # transform vectors from imu_0 to imu_t
    vec_wrt_imu_t = np.dot(imu_transforms_t[0:3,0:3].transpose(), vec_wrt_imu_0)
    # transform vectors from imu_t to lidar_t
    vec_wrt_lidar_t = np.dot(T_from_i_to_l[0:3,0:3], vec_wrt_imu_t);
    # transform vectors from lidar_t to camera_t
    vec_wrt_camera_t = dot(R_to_c_from_l(cam), vec_wrt_lidar_t)
    vec_wrt_camera_t = dot(cam['E'][0:3,0:3], vec_wrt_camera_t)
    return vec_wrt_camera_t

# same as MapPos, but takes the imu transforms of the car and computes the local trajectory on ground.
def MapPosTrajectory(imu_tr, imu_transforms_t, cam, T_from_i_to_l, height):
    height_array = np.zeros([3,3])
    height_array[2,0]=-height
    aa = np.dot(imu_tr[:,0:3,0:3], height_array)[:,:,0] # shift down in the self frame
    pts_wrt_imu_0 = (array(imu_tr[:,0:3,3])+aa).transpose()
    pts_wrt_imu_0 = np.vstack((pts_wrt_imu_0, np.ones((1,pts_wrt_imu_0.shape[1]))))
    # transform points from imu_0 to imu_t
    pts_wrt_imu_t = np.dot( np.linalg.inv(imu_transforms_t), pts_wrt_imu_0)
    # transform points from imu_t to lidar_t
    pts_wrt_lidar_t = np.dot(T_from_i_to_l, pts_wrt_imu_t);
    # transform points from lidar_t to camera_t
    pts_wrt_camera_t = pts_wrt_lidar_t.transpose()[:, 0:3] + cam['displacement_from_l_to_c_in_lidar_frame']
    pts_wrt_camera_t = dot(R_to_c_from_l(cam), pts_wrt_camera_t.transpose())
    pts_wrt_camera_t = np.vstack((pts_wrt_camera_t,
        np.ones((1,pts_wrt_camera_t.shape[1]))))
    pts_wrt_camera_t = dot(cam['E'], pts_wrt_camera_t)
    pts_wrt_camera_t = pts_wrt_camera_t[0:3,:]
    return pts_wrt_camera_t



def IMUPosTrajectory(imu_tr, imu_transforms_t, cam, height):
    height_array = np.zeros([3,3])
    height_array[2,0]=-height
    aa = np.dot(imu_tr[:,0:3,0:3], height_array)[:,:,0] # shift down in the self frame
    pts_wrt_imu_0 = (array(imu_tr[:,0:3,3])+aa).transpose()
    pts_wrt_imu_0 = np.vstack((pts_wrt_imu_0, np.ones((1,pts_wrt_imu_0.shape[1]))))
    # transform points from imu_0 to imu_t
    pts_wrt_imu_t = np.dot( np.linalg.inv(imu_transforms_t), pts_wrt_imu_0)
    return pts_wrt_imu_t[0:3]

def ENU2IMUQ50(world_coordinates, start_frame):
    roll_start = deg2rad(start_frame[8]);
    pitch_start = deg2rad(start_frame[7]);
    yaw_start = -deg2rad(start_frame[9]);

    psi = pitch_start; 
    cp = cos(psi);
    sp = sin(psi);
    theta = roll_start;
    ct = cos(theta);
    st = sin(theta);
    gamma = yaw_start;
    cg = cos(gamma);
    sg = sin(gamma);

    R_to_i_from_w = \
            array([[cg*cp-sg*st*sp, -sg*ct, cg*sp+sg*st*cp],
                  [sg*cp+cg*st*sp, cg*ct, sg*sp-cg*st*cp],
                  [-ct*sp, st, ct*cp]]).transpose()
    pos_wrt_imu = dot(R_to_i_from_w, world_coordinates);
    return pos_wrt_imu



def GPSVelocities(GPSData):
   return (np.apply_along_axis(np.linalg.norm, 1, GPSData[:,4:7]))

def GPSPos(GPSData, Camera, start_frame):
    roll_start = -deg2rad(start_frame[7]);
    pitch_start = deg2rad(start_frame[8]);
    yaw_start = -deg2rad(start_frame[9]+90);

    psi = pitch_start; 
    cp = cos(psi);
    sp = sin(psi);
    theta = roll_start;
    ct = cos(theta);
    st = sin(theta);
    gamma = yaw_start;
    cg = cos(gamma);
    sg = sin(gamma);

    R_to_i_from_w = \
            array([[cg*cp-sg*st*sp, -sg*ct, cg*sp+sg*st*cp],
                  [sg*cp+cg*st*sp, cg*ct, sg*sp-cg*st*cp],
                  [-ct*sp, st, ct*cp]]).transpose()


    pts = WGS84toENU(start_frame[1:4], GPSData[:, 1:4])
    world_coordinates = pts;
    pos_wrt_imu = dot(R_to_i_from_w, world_coordinates);
    R_to_c_from_i = Camera['R_to_c_from_i']
    R_camera_pitch = euler_matrix(Camera['rot_x'], Camera['rot_y'],\
            Camera['rot_z'], 'sxyz')[0:3,0:3]
    R_to_c_from_i = dot(R_camera_pitch, R_to_c_from_i) 

    pos_wrt_camera = dot(R_to_c_from_i, pos_wrt_imu);

    pos_wrt_camera[0,:] += Camera['t_x'] #move to left/right
    pos_wrt_camera[1,:] += Camera['t_y'] #move up/down image
    pos_wrt_camera[2,:] += Camera['t_z'] #move away from cam
    return pos_wrt_camera



def GPSPosShifted(GPSData, Camera, start_frame):
    roll_start = -deg2rad(start_frame[7]);
    pitch_start = deg2rad(start_frame[8]);
    yaw_start = -deg2rad(start_frame[9]+90);

    psi = pitch_start; 
    cp = cos(psi);
    sp = sin(psi);
    theta = roll_start;
    ct = cos(theta);
    st = sin(theta);
    gamma = yaw_start;
    cg = cos(gamma);
    sg = sin(gamma);

    R_to_i_from_w = \
            array([[cg*cp-sg*st*sp, -sg*ct, cg*sp+sg*st*cp],
                  [sg*cp+cg*st*sp, cg*ct, sg*sp-cg*st*cp],
                  [-ct*sp, st, ct*cp]]).transpose()


    pts = WGS84toENU(start_frame[1:4], GPSData[:, 1:4])
    vel = GPSData[:,4:7]
    vel[:,[0, 1]] = vel[:,[1, 0]]
    sideways = np.cross(vel, np.array([0,0,1]), axisa=1)
    sideways/= np.sqrt((sideways ** 2).sum(-1))[..., np.newaxis]
    #pts[0,:] = pts[0,:]+np.transpose(GPSData[:,4])/40
    #pts[1,:] = pts[1,:]-np.transpose(GPSData[:,5])/40
    pts = pts+sideways.transpose()
    world_coordinates = pts;
    pos_wrt_imu = dot(R_to_i_from_w, world_coordinates);
    R_to_c_from_i = Camera['R_to_c_from_i']
    R_camera_pitch = euler_matrix(Camera['rot_x'], Camera['rot_y'],\
            Camera['rot_z'], 'sxyz')[0:3,0:3]
    R_to_c_from_i = dot(R_camera_pitch, R_to_c_from_i) 

    pos_wrt_camera = dot(R_to_c_from_i, pos_wrt_imu);

    pos_wrt_camera[0,:] += Camera['t_x'] #move to left/right
    pos_wrt_camera[1,:] += Camera['t_y'] #move up/down image
    pos_wrt_camera[2,:] += Camera['t_z'] #move away from cam
    return pos_wrt_camera



def ENU2IMU(world_coordinates, start_frame):
    roll_start = -deg2rad(start_frame[7]);
    pitch_start = deg2rad(start_frame[8]);
    yaw_start = -deg2rad(start_frame[9]+90);

    psi = pitch_start; 
    cp = cos(psi);
    sp = sin(psi);
    theta = roll_start;
    ct = cos(theta);
    st = sin(theta);
    gamma = yaw_start;
    cg = cos(gamma);
    sg = sin(gamma);

    R_to_i_from_w = \
            array([[cg*cp-sg*st*sp, -sg*ct, cg*sp+sg*st*cp],
                  [sg*cp+cg*st*sp, cg*ct, sg*sp-cg*st*cp],
                  [-ct*sp, st, ct*cp]]).transpose()
    pos_wrt_imu = dot(R_to_i_from_w, world_coordinates);
    return pos_wrt_imu


def GPSPosIMU(GPSData, Camera, start_frame):
    roll_start = -deg2rad(start_frame[7]);
    pitch_start = deg2rad(start_frame[8]);
    yaw_start = -deg2rad(start_frame[9]+90);

    psi = pitch_start; 
    cp = cos(psi);
    sp = sin(psi);
    theta = roll_start;
    ct = cos(theta);
    st = sin(theta);
    gamma = yaw_start;
    cg = cos(gamma);
    sg = sin(gamma);

    R_to_i_from_w = \
            array([[cg*cp-sg*st*sp, -sg*ct, cg*sp+sg*st*cp],
                  [sg*cp+cg*st*sp, cg*ct, sg*sp-cg*st*cp],
                  [-ct*sp, st, ct*cp]]).transpose()
    pts = WGS84toENU(start_frame[1:4], GPSData[:, 1:4])
    world_coordinates = pts;
    pos_wrt_imu = dot(R_to_i_from_w, world_coordinates);
    return pos_wrt_imu

def GPSPosCamera(pos_wrt_imu, Camera):
    R_to_c_from_i = Camera['R_to_c_from_i']
    R_camera_pitch = euler_matrix(Camera['rot_x'], Camera['rot_y'],Camera['rot_z'], 'sxyz')[0:3,0:3]
    R_to_c_from_i = dot(R_camera_pitch, R_to_c_from_i) 

    pos_wrt_camera = dot(R_to_c_from_i, pos_wrt_imu);

    pos_wrt_camera[0,:] += Camera['t_x'] #move to left/right
    pos_wrt_camera[1,:] += Camera['t_y'] #move up/down image
    pos_wrt_camera[2,:] += Camera['t_z'] #move away from cam
    return pos_wrt_camera




def GPSColumns(GPSData, Camera, start_frame):
    pos_wrt_camera = GPSPos(GPSData, Camera, start_frame)
    return PointsMask(pos_wrt_camera[:,1:], Camera)

def GPSShiftedColumns(GPSData, Camera, start_frame):
    pos_wrt_camera = GPSPosShifted(GPSData, Camera, start_frame)
    return PointsMask(pos_wrt_camera, Camera)

def PointsMask(pos_wrt_camera, Camera):
    if False:#'distort' in Camera:
      (vpix, J) = projectPoints(pts_wrt_camera.transpose(), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), Camera['KK'], Camera['distort'])
    else:
      vpix = dot(Camera['KK'], divide(pos_wrt_camera, pos_wrt_camera[2,:]))
    vpix = around(vpix).astype(np.int32)
    return vpix
   

def GPSMask(GPSData, Camera, width=2): 
    I = 255*ones((960,1280,3), np.uint8)
    vpix = GPSColumns(GPSData, Camera, GPSData[0, :])
    vpix = vpix[:,vpix[0,:] > 0 + width/2]
    if vpix.size > 0:
      vpix = vpix[:,vpix[1,:] > 0 + width/2]
    if vpix.size > 0:
      vpix = vpix[:,vpix[0,:] < 1279 - width/2]
    if vpix.size > 0:
      vpix = vpix[:,vpix[1,:] < 959 - width/2]
    
      for p in range(-width/2,width/2):
          I[vpix[1,:]+p, vpix[0,:], :] = 0
          I[vpix[1,:], vpix[0,:]+p, :] = 0
          I[vpix[1,:]-p, vpix[0,:], :] = 0
          I[vpix[1,:], vpix[0,:]-p, :] = 0

    """
    for idx in range(1,pts.shape[1]):
      pix = vpix[:,idx]
      if (pix[0] > 0 and pix[0] < 1280 and pix[1] > 0 and pix[1] < 960):
        I[pix[1]-width+1:pix[1]+width, pix[0]-width+1:pix[0]+width, 0] = 0;
        I[pix[1]-width+1:pix[1]+width, pix[0]-width+1:pix[0]+width, 1] = 0;
        I[pix[1]-width+1:pix[1]+width, pix[0]-width+1:pix[0]+width, 2] = 0;
    """
    
    return I

