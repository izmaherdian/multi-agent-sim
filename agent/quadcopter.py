import numpy as np

from agent.quad_drawer import QuadDrawer
from agent.quad_dynamics import QuadDynamics
from agent.quad_control import QuadControl
from agent.quad_trajectory import QuadTrajectory

from agent.utils.rotation_conversion import quatToYPR_ZYX

rad2deg = 180.0 / np.pi
rads2rpm = 60.0 / (2.0 * np.pi)

class Quadcopter:
    def __init__(self, id, ax, init_pos=None, front_color='red', ctrlType="xyz_pos", trajSelect=(0,0,0), wayPoints=None):
        self.id = id
        self.dynamics   = QuadDynamics(Ti=0, custom_pos=init_pos, id=id)
        self.controller = QuadControl(self.dynamics, trajSelect[1])
        self.trajectory = QuadTrajectory(self.dynamics, ctrlType=ctrlType, trajSelect=trajSelect, wayPoints=wayPoints if wayPoints is not None else [])
        self.drawer     = QuadDrawer(ax, front_color=front_color)
        self.init_pos   = init_pos if init_pos is not None else np.zeros(3)

        self.position   = self.dynamics.pos
        self.x          = self.position[0]
        self.y          = self.position[1]
        self.z          = self.position[2]
        self.quat       = self.dynamics.quat
        self.q0         = self.quat[0]
        self.q1         = self.quat[1]
        self.q2         = self.quat[2]
        self.q3         = self.quat[3]
        self.euler      = self.dynamics.euler
        self.phi        = self.euler[0] * rad2deg
        self.theta      = self.euler[1] * rad2deg
        self.psi        = self.euler[2] * rad2deg
        self.velocity   = self.dynamics.vel
        self.vx         = self.velocity[0]
        self.vy         = self.velocity[1]
        self.vz         = self.velocity[2]
        self.omega      = self.dynamics.omega
        self.p          = self.omega[0] * rad2deg
        self.q          = self.omega[1] * rad2deg
        self.r          = self.omega[2] * rad2deg
        self.wMotor     = self.dynamics.wMotor
        self.wM1        = self.wMotor[0] * rads2rpm
        self.wM2        = self.wMotor[1] * rads2rpm
        self.wM3        = self.wMotor[2] * rads2rpm
        self.wM4        = self.wMotor[3] * rads2rpm

        self.sDes       = self.controller.sDesCalc
        self.xDes       = self.sDes[0]
        self.yDes       = self.sDes[1]
        self.zDes       = self.sDes[2]
        self.q0Des      = self.sDes[9]
        self.q1Des      = self.sDes[10]
        self.q2Des      = self.sDes[11]
        self.q3Des      = self.sDes[12]
        quatDes = self.sDes[9:13]              # [q0Des, q1Des, q2Des, q3Des]
        yaw0, pitch0, roll0 = quatToYPR_ZYX(quatDes)
        self.psiDes   = yaw0   * rad2deg
        self.thetaDes = pitch0 * rad2deg
        self.phiDes   = roll0  * rad2deg
        self.vxDes      = self.sDes[3]
        self.vyDes      = self.sDes[4]
        self.vzDes      = self.sDes[5]
        self.pDes       = self.trajectory.desPQR[0] * rad2deg
        self.qDes       = self.trajectory.desPQR[1] * rad2deg
        self.rDes       = self.trajectory.desPQR[2] * rad2deg      
        self.w_cmd      = self.controller.w_cmd
        self.wM1_cmd    = self.w_cmd[0] * rads2rpm
        self.wM2_cmd    = self.w_cmd[1] * rads2rpm
        self.wM3_cmd    = self.w_cmd[2] * rads2rpm
        self.wM4_cmd    = self.w_cmd[3] * rads2rpm

        self.yawRateDes = self.trajectory.desYawRate
        self.xthrDes    = self.sDes[6]
        self.ythrDes    = self.sDes[7]
        self.zthrDes    = self.sDes[8]

    def initialize_draw(self):
        self.drawer.draw(self.position, self.quat, label=f'Q{self.id}')

    def update(self, t, dt, wind):
        # 1. Update desired trajectory setpoint
        sDes = self.trajectory.desiredState(t, dt, self.dynamics)
        # print(f"Quadcopter {self.id} desired position: {sDes[:3]}")

        # 2. Compute control commands from current state and setpoint
        self.controller.controller(self.trajectory, self.dynamics, sDes, dt)
        # print(f"Quadcopter {self.id} control commands: {self.controller.w_cmd}")

        # 3. Update dynamics state with control commands and wind
        self.dynamics.update(t, dt, self.controller.w_cmd, wind)

        # 4. Update position and orientation
        self.position       = self.dynamics.pos
        self.x              = self.position[0]
        self.y              = self.position[1]
        self.z              = self.position[2]        
        self.quat           = self.dynamics.quat
        self.q0             = self.quat[0]
        self.q1             = self.quat[1]
        self.q2             = self.quat[2]
        self.q3             = self.quat[3]
        self.euler          = self.dynamics.euler
        self.phi            = self.euler[0] 
        self.theta          = self.euler[1] 
        self.psi            = self.euler[2] 
        self.velocity       = self.dynamics.vel
        self.vx             = self.velocity[0]
        self.vy             = self.velocity[1]
        self.vz             = self.velocity[2]
        self.omega          = self.dynamics.omega
        self.p              = self.omega[0] 
        self.q              = self.omega[1] 
        self.r              = self.omega[2] 
        self.wMotor         = self.dynamics.wMotor
        self.wM1            = self.wMotor[0] * rads2rpm
        self.wM2            = self.wMotor[1] * rads2rpm
        self.wM3            = self.wMotor[2] * rads2rpm
        self.wM4            = self.wMotor[3] * rads2rpm

        self.sDes           = self.controller.sDesCalc
        self.xDes           = self.sDes[0]
        self.yDes           = self.sDes[1]
        self.zDes           = self.sDes[2]
        self.q0Des          = self.sDes[9]
        self.q1Des          = self.sDes[10]
        self.q2Des          = self.sDes[11]
        self.q3Des          = self.sDes[12]
        quatDes             = self.sDes[9:13]              # [q0Des, q1Des, q2Des, q3Des]
        yaw0, pitch0, roll0 = quatToYPR_ZYX(quatDes)
        self.psiDes         = yaw0 
        self.thetaDes       = pitch0 
        self.phiDes         = roll0 
        self.vxDes          = self.sDes[3]
        self.vyDes          = self.sDes[4]
        self.vzDes          = self.sDes[5]
        self.pDes           = self.trajectory.desPQR[0] 
        self.qDes           = self.trajectory.desPQR[1] 
        self.rDes           = self.trajectory.desPQR[2] 
        self.w_cmd          = self.controller.w_cmd
        self.wM1_cmd        = self.w_cmd[0] * rads2rpm
        self.wM2_cmd        = self.w_cmd[1] * rads2rpm
        self.wM3_cmd        = self.w_cmd[2] * rads2rpm
        self.wM4_cmd        = self.w_cmd[3] * rads2rpm

        self.yawRateDes     = self.trajectory.desYawRate
        self.xthrDes        = self.sDes[6]
        self.ythrDes        = self.sDes[7]
        self.zthrDes        = self.sDes[8]

        # 5. Update visualization
        self.drawer.update_draw(self.position, self.quat, label=f'Q{self.id}')