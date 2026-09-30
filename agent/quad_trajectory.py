import numpy as np
from numpy import pi
from numpy.linalg import norm

from environment.obstacles import Obstacles

deg2rad = np.pi / 180.0

class QuadTrajectory:

    def __init__(self, quad, ctrlType, trajSelect, wayPoints=None):

        self.ctrlType = ctrlType
        self.xyzType = trajSelect[0]
        self.yawType = trajSelect[1]
        self.averVel = trajSelect[2]

        if wayPoints is None or len(wayPoints) != 4:
            self.t_wps, self.wps, self.y_wps, self.v_wp = [], [], [], []
        else:
            self.t_wps, self.wps, self.y_wps, self.v_wp = wayPoints


        self.external_desired_pos = None

        self.end_reached = 0

        if (self.ctrlType == "xyz_pos"):
            self.T_segment = np.diff(self.t_wps)

        # Get initial heading
        self.current_heading = quad.psi
        
        # Initialize trajectory setpoint
        self.desPos = np.zeros(3)    # Desired position (x, y, z)
        self.desVel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)    # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros(3)    # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros(3)    # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = 0.         # Desired yaw speed
        self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)).astype(float)

    def set_external_desired_pos(self, pos):
        self.external_desired_pos = np.array(pos)

    def desiredState(self, t, Ts, quad):

        if self.xyzType in [99] and self.ctrlType == "xyz_pos":
            if self.external_desired_pos is None:
                print("Warning: external desired position belum di-set di QuadTrajectory")
                desPos = np.zeros(3)  
            else:
                desPos = self.external_desired_pos

        else:
            desPos = np.zeros(3)    # Default desired position
        
        self.desPos = np.array(desPos)    # Desired position (x, y, z)
        self.desVel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)    # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros(3)    # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros(3)    # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = 0.         # Desired yaw speed

        # DEFINISI TRAJEKTORI
        def pos_waypoint_timed():
            if not (len(self.t_wps) == self.wps.shape[0]):
                raise Exception("Time array and waypoint array not the same size.")
            elif (np.diff(self.t_wps) <= 0).any():
                raise Exception("Time array isn't properly ordered.")  
            if (t == 0):
                self.t_idx = 0
            elif (t >= self.t_wps[-1]):
                self.t_idx = -1
            else:
                self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
            self.desPos = self.wps[self.t_idx,:]

        def pos_waypoint_interp():
            if not (len(self.t_wps) == self.wps.shape[0]):
                raise Exception("Time array and waypoint array not the same size.")
            elif (np.diff(self.t_wps) <= 0).any():
                raise Exception("Time array isn't properly ordered.") 
            if (t == 0):
                self.t_idx = 0
                self.desPos = self.wps[0,:]
            elif (t >= self.t_wps[-1]):
                self.t_idx = -1
                self.desPos = self.wps[-1,:]
            else:
                self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
                scale = (t - self.t_wps[self.t_idx])/self.T_segment[self.t_idx]
                self.desPos = (1 - scale) * self.wps[self.t_idx,:] + scale * self.wps[self.t_idx + 1,:]






        # DEFINISI YAW
        def yaw_waypoint_timed():
            if not (len(self.t_wps) == len(self.y_wps)):
                raise Exception("Time array and waypoint array not the same size.")
            self.desEul[2] = self.y_wps[self.t_idx]

        def yaw_waypoint_interp():
            if not (len(self.t_wps) == len(self.y_wps)):
                raise Exception("Time array and waypoint array not the same size.")
            if (t == 0) or (t >= self.t_wps[-1]):
                self.desEul[2] = self.y_wps[self.t_idx]
            else:
                scale = (t - self.t_wps[self.t_idx])/self.T_segment[self.t_idx]
                self.desEul[2] = (1 - scale)*self.y_wps[self.t_idx] + scale*self.y_wps[self.t_idx + 1]
                # Angle between current vector with the next heading vector
                delta_psi = self.desEul[2] - self.current_heading
                # Set Yaw rate
                self.desYawRate = delta_psi / Ts 
                # Prepare next iteration
                self.current_heading = self.desEul[2]

        def yaw_follow():
            if (self.xyzType == 0 or self.xyzType == 1 or self.xyzType == 2 or self.xyzType == 99):
                if (t == 0):
                    self.desEul[2] = 0
                else:
                    # Calculate desired Yaw
                    self.desEul[2] = np.arctan2(self.desPos[1]-quad.pos[1], self.desPos[0]-quad.pos[0])
            else:
                if (t == 0) or (t >= self.t_wps[-1]):
                    self.desEul[2] = self.y_wps[self.t_idx]
                else:
                    # Calculate desired Yaw
                    self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])
            # Dirty hack, detect when desEul[2] switches from -pi to pi (or vice-versa) and switch manualy current_heading 
            if (np.sign(self.desEul[2]) - np.sign(self.current_heading) and abs(self.desEul[2]-self.current_heading) >= 2*pi-0.1):
                self.current_heading = self.current_heading + np.sign(self.desEul[2])*2*pi
            # Angle between current vector with the next heading vector
            delta_psi = self.desEul[2] - self.current_heading
            # Set Yaw rate
            self.desYawRate = delta_psi / Ts 
            # Prepare next iteration
            self.current_heading = self.desEul[2]




        # Penentuan Kontrol
        if (self.ctrlType == "xyz_vel"):
            if (self.xyzType == 0):
                self.sDes = testVelControl(t)

        elif (self.ctrlType == "xy_vel_z_pos"):
            if (self.xyzType == 0):
                self.sDes = testVelControl(t)
        
        elif (self.ctrlType == "xyz_pos"):
            # Penentuan Metode Trajektori
            if (self.xyzType == 0):
                # Hover di waypoint
                self.desPos = self.wps[0]
            elif (self.xyzType == 1):
                # Linear interpolasi antar waypoint
                pos_waypoint_timed()
            elif (self.xyzType == 2):
                # Cubic interpolasi antar waypoint
                pos_waypoint_interp()




            # Penentuan Metode Yaw
            if (self.yawType == 0):
                # Yaw tetap pada heading awal
                self.desEul[2] = self.current_heading

            elif (self.yawType == 1):
                # Yaw mengikuti waypoint
                yaw_waypoint_timed()

            elif (self.yawType == 2):
                # Yaw mengikuti waypoint dengan interpolasi
                yaw_waypoint_interp()

            elif (self.yawType == 3):
                # Yaw mengikut arah kecepatan
                yaw_follow()

            elif (self.yawType == 4):
                # Yaw akan terus 0 derajat
                self.desEul[2] = 0 * deg2rad

            # Bangun vektor sDes (output final)
            self.sDes = np.hstack(( 
                                    self.desPos,        # [0:3]
                                    self.desVel,        # [3:6]
                                    self.desAcc,        # [6:9]
                                    self.desThr,        # [9:12]
                                    self.desEul,        # [12:15]
                                    self.desPQR,        # [15:18]
                                    self.desYawRate     # [18]
                                )) .astype(float)

        return self.sDes

## Testing scripts

def testVelControl(t):
    desPos = np.array([0., 0., 0.])
    desVel = np.array([0., 0., 0.])
    desAcc = np.array([0., 0., 0.])
    desThr = np.array([0., 0., 0.])
    desEul = np.array([0., 0., 0.])
    desPQR = np.array([0., 0., 0.])
    desYawRate = 0.

    if t >= 1 and t < 4:
        desVel = np.array([3, 2, 0])
    elif t >= 4 and t < 7:
        desVel = np.array([3, -1, 0])
    elif t >= 7:
        desVel = np.array([-3, -1, 0])

    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)
    
    return sDes
