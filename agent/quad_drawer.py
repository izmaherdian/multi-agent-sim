import numpy as np

from agent.quad_dynamics import sys_params 
from agent.utils.rotation_conversion import quat2Dcm

sys_params = sys_params()
dxm = sys_params['dxm']
dym = sys_params['dym']
dzm = sys_params['dzm']

orient = "NED"

class QuadDrawer:
    def __init__(self, ax, front_color='red', lw=3):
        self.ax = ax
        self.front_color = front_color
        self.lw = lw
        self.v_front = None
        self.v_back = None
        self.label = None
        self.trajectory_points = []
        self.traj_line = None

    def _compute_pts(self, position, quat):
        # posisi dasar
        x, y, z = position

        if orient == "NED":
            z = -z
            quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])

        # Konversi quaternion ke matriks rotasi
        R = quat2Dcm(quat)
        
        motorPoints = np.array([
                                [dxm, -dym, dzm], 
                                [0, 0, 0], 
                                [dxm, dym, dzm], 
                                [-dxm, dym, dzm], 
                                [0, 0, 0], 
                                [-dxm, -dym, dzm]
                                ])

        # rotasi dan translasi titik motor ke posisi dunia
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0, :] += x
        motorPoints[1, :] += y
        motorPoints[2, :] += z

        # bagi jadi dua garis (depan dan belakang)
        v_front = motorPoints[:, 0:3].T  
        v_back = motorPoints[:, 3:6].T

        return v_front, v_back


    def draw(self, position, quat=None, label=None):
        if quat is None:
            quat = [1, 0, 0, 0]  # default identitas
        v1, v2 = self._compute_pts(position, quat)

        self.v_front, = self.ax.plot(v1[:,0], v1[:,1], v1[:,2], color=self.front_color, lw=self.lw)
        self.v_back,  = self.ax.plot(v2[:,0], v2[:,1], v2[:,2], color='black', lw=self.lw)

        if label:
            x, y, z = position
            if orient == "NED":
                z = -z

            self.label = self.ax.text(x, y, z + 0.5, label, color=self.front_color, fontsize=8)

        self.trajectory_points.append(position.copy())
        if self.traj_line is not None:
            self.traj_line.remove()
        pts = np.array(self.trajectory_points)
        if orient == "NED":
            pts[:,2] = -pts[:,2]
        self.traj_line, = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=self.front_color, lw=1, linestyle='--', alpha=0.7)

    def update_draw(self, position, quat=None, label=None):
        if self.v_front is None or self.v_back is None:
            raise RuntimeError("Call draw() before update_draw()")

        if quat is None:
            quat = [1, 0, 0, 0]

        v1, v2 = self._compute_pts(position, quat)

        self.v_front.set_data(v1[:,0], v1[:,1])
        self.v_front.set_3d_properties(v1[:,2])
        
        self.v_back.set_data(v2[:,0], v2[:,1])
        self.v_back.set_3d_properties(v2[:,2])

        if label and self.label:
            x, y, z = position
            if orient == "NED":
                z = -z
            self.label.set_position((x, y))
            self.label.set_3d_properties(z + 0.1)
        elif label and not self.label:
            x, y, z = position
            if orient == "NED":
                z = -z
            self.label = self.ax.text(x, y, z + 0.5, label, color=self.front_color, fontsize=8)

        self.trajectory_points.append(position.copy())
        if self.traj_line is not None:
            self.traj_line.remove()
        pts = np.array(self.trajectory_points)
        if orient == "NED":
            pts[:,2] = -pts[:,2]
        self.traj_line, = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=self.front_color, lw=1, linestyle='--', alpha=0.7)