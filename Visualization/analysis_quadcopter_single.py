import pickle
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent.MultiAgentConfig import MultiAgentConfig
from Environment.Obstacles import Obstacles

with open(f'single_quad_data.pkl', 'rb') as file:
    loaded = pickle.load(file)
print(loaded.keys())
data = loaded['data_quad_single']

def plot_quad_data(arr_quad):
    """
    Menerima arr_quad (shape: [T,45]) dan membuat serangkaian plot:
    1) Posisi vs Setpoint
    2) Kecepatan vs Setpoint
    3) Thrust Setpoint (x_thr_sp, y_thr_sp, z_thr_sp)
    4) Sudut Euler vs Setpoint (phi, theta, psi)
    5) Kecepatan Sudut vs Setpoint (p, q, r)
    6) Kecepatan Motor vs Komando Motor
    7) Error Posisi (x_err, y_err, z_err)

    Parameter:
        - arr_quad: np.ndarray shape (T,45)
        - quad_id: jika diberikan, akan ditampilkan di judul plot
    """
    # Kolom‐kolom arr_quad sesuai urutan di main_multi_agent:
    # 0: t
    # 1: x      2: y      3: z
    # 4: q0     5: q1     6: q2     7: q3
    # 8: phi    9: theta 10: psi
    # 11: v_x   12: v_y  13: v_z
    # 14: p     15: q    16: r
    # 17: wM1   18: wM2  19: wM3   20: wM4
    # 21: x_sp  22: y_sp 23: z_sp
    # 24: q0_des 25: q1_des 26: q2_des 27: q3_des
    # 28: phi_sp 29: theta_sp 30: psi_sp
    # 31: v_x_sp 32: v_y_sp 33: v_z_sp
    # 34: p_des 35: q_des 36: r_des
    # 37: yaw_rate_sp
    # 38: x_thr_sp 39: y_thr_sp 40: z_thr_sp
    # 41: wM1_sp 42: wM2_sp 43: wM3_sp 44: wM4_sp

    t       = arr_quad[:, 0]

    # Posisi dan setpoint posisi
    x       = arr_quad[:, 1]
    y       = arr_quad[:, 2]
    z       = arr_quad[:, 3]
    x_sp    = arr_quad[:, 21]
    y_sp    = arr_quad[:, 22]
    z_sp    = arr_quad[:, 23]

    # Kecepatan dan setpoint kecepatan
    vx      = arr_quad[:, 11]
    vy      = arr_quad[:, 12]
    vz      = arr_quad[:, 13]
    vx_sp   = arr_quad[:, 31]
    vy_sp   = arr_quad[:, 32]
    vz_sp   = arr_quad[:, 33]

    # Thrust setpoint (karena tidak ada thrust aktual, hanya setpoint)
    x_thr_sp = arr_quad[:, 38]
    y_thr_sp = arr_quad[:, 39]
    z_thr_sp = arr_quad[:, 40]

    # Sudut Euler dan setpoint Euler
    phi     = arr_quad[:, 8]
    theta   = arr_quad[:, 9]
    psi     = arr_quad[:, 10]
    phi_sp  = arr_quad[:, 28]
    theta_sp= arr_quad[:, 29]
    psi_sp  = arr_quad[:, 30]

    # Kecepatan sudut dan setpoint kecepatan sudut
    p       = arr_quad[:, 14]
    q       = arr_quad[:, 15]
    r       = arr_quad[:, 16]
    p_sp    = arr_quad[:, 34]
    q_sp    = arr_quad[:, 35]
    r_sp    = arr_quad[:, 36]

    # Kecepatan motor aktual dan komando motor (RPM)
    wM1     = arr_quad[:, 17]
    wM2     = arr_quad[:, 18]
    wM3     = arr_quad[:, 19]
    wM4     = arr_quad[:, 20]
    wM1_sp  = arr_quad[:, 41]
    wM2_sp  = arr_quad[:, 42]
    wM3_sp  = arr_quad[:, 43]
    wM4_sp  = arr_quad[:, 44]

    # Error posisi
    x_err = x_sp - x
    y_err = y_sp - y
    z_err = z_sp - z

    # Supaya nanti judul plot tahu quad ID-nya
    judul_prefix = f"Quadcopter Single" 

    # ----------------------------------------------------------
    # 1) Plot Posisi Aktual vs Setpoint
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(t, x,  label='x')
    plt.plot(t, y,  label='y')
    plt.plot(t, z,  label='z')
    plt.plot(t, x_sp, '--', label='x_sp')
    plt.plot(t, y_sp, '--', label='y_sp')
    plt.plot(t, z_sp, '--', label='z_sp')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title(f'{judul_prefix}Posisi Aktual vs Setpoint')
    plt.tight_layout()

    # ----------------------------------------------------------
    # 2) Plot Kecepatan Linear Aktual vs Setpoint
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(t, vx,   label='Vx')
    plt.plot(t, vy,   label='Vy')
    plt.plot(t, vz,   label='Vz')
    plt.plot(t, vx_sp, '--', label='Vx_sp')
    plt.plot(t, vy_sp, '--', label='Vy_sp')
    plt.plot(t, vz_sp, '--', label='Vz_sp')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'{judul_prefix}Kecepatan Linear Aktual vs Setpoint')
    plt.tight_layout()

    # ----------------------------------------------------------
    # 3) Plot Thrust Setpoint (x_thr_sp, y_thr_sp, z_thr_sp)
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(t, x_thr_sp, label='x_thr_sp')
    plt.plot(t, y_thr_sp, label='y_thr_sp')
    plt.plot(t, z_thr_sp, label='z_thr_sp')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust Setpoint (N)')
    plt.title(f'{judul_prefix}Thrust Setpoint per Sumbu')
    plt.tight_layout()

    # ----------------------------------------------------------
    # 4) Plot Sudut Euler (phi, theta, psi) vs Setpoint
    # ----------------------------------------------------------
    rad2deg = 180.0 / np.pi
    phi_deg      = phi * rad2deg
    theta_deg    = theta * rad2deg
    psi_deg      = psi * rad2deg
    phi_sp_deg   = phi_sp * rad2deg
    theta_sp_deg = theta_sp * rad2deg
    psi_sp_deg   = psi_sp * rad2deg

    plt.figure(figsize=(8, 5))
    plt.plot(t, phi_deg,    label='roll')
    plt.plot(t, theta_deg,  label='pitch')
    plt.plot(t, psi_deg,    label='yaw')
    plt.plot(t, phi_sp_deg,    '--', label='roll_sp')
    plt.plot(t, theta_sp_deg,  '--', label='pitch_sp')
    plt.plot(t, psi_sp_deg,    '--', label='yaw_sp')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Euler Angle (°)')
    plt.title(f'{judul_prefix}Sudut Euler Aktual vs Setpoint')
    plt.tight_layout()

    # ----------------------------------------------------------
    # 5) Plot Kecepatan Sudut (p, q, r) vs Setpoint
    # ----------------------------------------------------------
    p_deg     = p * rad2deg
    q_deg     = q * rad2deg
    r_deg     = r * rad2deg
    p_sp_deg  = p_sp * rad2deg
    q_sp_deg  = q_sp * rad2deg
    r_sp_deg  = r_sp * rad2deg

    plt.figure(figsize=(8, 5))
    plt.plot(t, p_deg,    label='p')
    plt.plot(t, q_deg,    label='q')
    plt.plot(t, r_deg,    label='r')
    plt.plot(t, p_sp_deg, '--', label='p_sp')
    plt.plot(t, q_sp_deg, '--', label='q_sp')
    plt.plot(t, r_sp_deg, '--', label='r_sp')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (°/s)')
    plt.title(f'{judul_prefix}Kecepatan Sudut Aktual vs Setpoint')
    plt.tight_layout()

    # ----------------------------------------------------------
    # 6) Plot Kecepatan Motor Aktual vs Komando Motor
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(t, wM1,  label='wM1 (aktual)')
    plt.plot(t, wM2,  label='wM2 (aktual)')
    plt.plot(t, wM3,  label='wM3 (aktual)')
    plt.plot(t, wM4,  label='wM4 (aktual)')
    plt.plot(t, wM1_sp, '--', label='wM1_sp (komando)')
    plt.plot(t, wM2_sp, '--', label='wM2_sp (komando)')
    plt.plot(t, wM3_sp, '--', label='wM3_sp (komando)')
    plt.plot(t, wM4_sp, '--', label='wM4_sp (komando)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Motor RPM')
    plt.title(f'{judul_prefix}RPM Motor Aktual vs Komando')
    plt.tight_layout()

    # ----------------------------------------------------------
    # 7) Plot Error Posisi (x_err, y_err, z_err)
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(t, x_err, label='Pos x error')
    plt.plot(t, y_err, label='Pos y error')
    plt.plot(t, z_err, label='Pos z error')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title(f'{judul_prefix}Error Posisi')
    plt.tight_layout()

    # ----------------------------------------------------------
    # Tampilkan semua figure untuk quad ini
    # ----------------------------------------------------------
    plt.show()

# ----------------------------------------------------------
# Loop melalui seluruh elemen data_quad
# ----------------------------------------------------------
if len(data) == 0:
    print("Tidak ada data_quad untuk diplot.")
else:
    for entry in data:
        quad_id = entry.get('id', None)
        arr_quad = entry['path']
        print(f"Plotting data for quadcopter {quad_id}")
        plot_quad_data(arr_quad)