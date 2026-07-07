import pickle
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent.MultiAgentConfig import MultiAgentConfig
from Environment.Obstacles import Obstacles
from Agent.QuadMultiAgent import QuadMultiAgent

multi_agent_config = MultiAgentConfig()
# obstacle_scheme = multi_agent_config.OBSTACLE_SCHEME
obstacle_scheme = 'NONE'
# wind_type       = multi_agent_config.WIND_TYPE
wind_type       = 'NONE'
# controller_type = multi_agent_config.CONTROLLER
controller_type = 'erc'  
# formation_type  = multi_agent_config.FORMATION_TYPE
formation_type  = 1
orient          = multi_agent_config.ORIENT

with open(f'multi_quad_data_{obstacle_scheme}_{wind_type}_{controller_type}_formation{formation_type}_{orient}.pkl', 'rb') as file:
    loaded = pickle.load(file)
print(loaded.keys())
data_quad = loaded['data_quad']

with open(f'multi_agent_data_{obstacle_scheme}_{wind_type}_{controller_type}_formation{formation_type}_{orient}.pkl', 'rb') as file:
    loaded = pickle.load(file)
print(loaded.keys())
data_agent = loaded['data_agent']

# Mencari waktu terakhir dari data_quad
t = data_quad[0]['path'][:, 0]
print(f"Waktu terakhir dari data_quad: {t[-1]} detik")

def calculate_formation_error_history(data_quad, topology):
    """
    Menghitung riwayat error formasi berdasarkan data_quad_quad simulasi.

    Args:
        data_quad: data_quad simulasi yang dimuat dari file .pkl.
              Struktur: [{'path': array([...])}, {'path': array([...])}, ...]
        topology: Matriks topologi ideal dari MultiAgentConfig.

    Returns:
        tuple: (time_vector, error_vector)
    """
    num_agents = len(data_quad)
    # Asumsikan semua agen memiliki jumlah langkah waktu yang sama
    num_time_steps = len(data_quad[0]['path'])
    
    # Ambil vektor waktu dari agen pertama
    time_vector = data_quad[0]['path'][:, 0]
    error_history = []

    # Loop untuk setiap langkah waktu dalam simulasi
    for t_idx in range(num_time_steps):
        current_total_error = 0.0
        
        # Loop untuk setiap pasangan agen unik (i, j)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Ambil posisi aktual agen i dan j pada waktu t
                pos_i_actual = data_quad[i]['path'][t_idx, 1:4] # Kolom 1, 2, 3 adalah x, y, z
                pos_j_actual = data_quad[j]['path'][t_idx, 1:4]
                
                # Hitung jarak aktual antara agen i dan j
                actual_distance = np.linalg.norm(pos_i_actual - pos_j_actual)
                
                # Ambil posisi ideal agen i dan j dari topologi
                pos_i_ideal = topology[i, :]
                pos_j_ideal = topology[j, :]
                
                # Hitung jarak ideal antara agen i dan j
                ideal_distance = np.linalg.norm(pos_i_ideal - pos_j_ideal)
                
                # Hitung selisih kuadrat dari jarak dan tambahkan ke total error
                squared_error = (actual_distance - ideal_distance)**2
                current_total_error += squared_error
        
        error_history.append(current_total_error)
        
    return time_vector, np.array(error_history)


def plot_formation_error(data_quad):
    """Fungsi baru untuk mem-plot error formasi."""
    
    num_robot = 5
    multi_agent_config.set_num_robot(num_robot)
    topology = multi_agent_config.TOPOLOGY

    # Hitung riwayat error
    time_steps, formation_errors = calculate_formation_error_history(data_quad, topology)
    
    # Mulai plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, formation_errors, 'b-', label='Indeks Performansi Formasi ($E_{formasi}$)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Formation Error (squared distance)')
    plt.title(f'Formation Performance Index ($E_{{formation}}$)\nObs: {obstacle_scheme} - Wind: {wind_type} - Contr: {controller_type}\n Formation: {formation_type} - Orient: {orient}')
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.show()

def plot_order(data_quad):
    plt.figure(figsize=(10, 6))
    path = data_quad[0]['path']
    headings = []
    
    for i in range(1,len(path)):
        heading = np.zeros(2) 
        for j in range(len(data_quad)):
            heading += data_quad[j]['path'][i,11:13]/np.linalg.norm(data_quad[j]['path'][i,11:13])
        headings.append(np.linalg.norm(heading)/len(data_quad))
    plt.plot(path[1:,0], headings, 'b-', label='Ordo Sistem')

    plt.xlabel("Time (s)")
    plt.ylabel("Order")
    plt.title(f'System Order\nObs: {obstacle_scheme} - Wind: {wind_type} - Contr: {controller_type}\n Formation: {formation_type} - Orient: {orient}')
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.show()

def get_circle(x, y, r):
    theta = np.linspace(0, 2 * np.pi, 20)
    a = x + r * np.cos(theta)
    b = y + r * np.sin(theta)
    return a, b

def plot_paths(data_quad):
    import matplotlib.pyplot as plt
    import numpy as np

    # 174 mm = 6.85 inches (at 72 dpi) 2.85
    plt.figure(figsize=(6.85, 2.85)) #2.85, 4.85
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    obstacles = Obstacles(scheme=obstacle_scheme)
    obstacles_2d = obstacles.obstacles_2d

    # Warna lintasan agen
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    # Gambar obstacle
    for i in range(len(obstacles_2d)):
        obstacle = obstacles_2d[i]
        if i == 0:
            plt.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.3, color='grey', label='Obstacle')
        else:
            plt.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.3, color='grey')

    # Waktu penanda (misal tiap 10 detik)
    all_times = data_quad[0]['path'][:, 0]
    t_max = all_times[-1]
    time_marks = np.arange(0, t_max + 1, 10)

    # Gambar lintasan + lingkaran waktu dengan warna masing-masing
    for i, data in enumerate(data_quad):
        path = data['path']
        t = path[:, 0]
        x = path[:, 1]
        y = path[:, 2]
        color = colors[i % len(colors)]

        plt.plot(x, y, label=f'Quad {i+1}', linewidth=1.5, color=color)

        # Marker waktu dengan warna sama
        for tm in time_marks:
            idx = (np.abs(t - tm)).argmin()
            cx, cy = get_circle(x[idx], y[idx], 0.15)
            plt.plot(cx, cy, color=color, linewidth=1.2, alpha=1.0)
            # plt.text(x[idx], y[idx], f'{int(tm)}s', fontsize=12, ha='center', va='center', color='black')

    # Garis koneksi antar agen pada waktu tertentu saja
    for tm in time_marks:
        xs, ys = [], []
        for data in data_quad:
            path = data['path']
            t = path[:, 0]
            x = path[:, 1]
            y = path[:, 2]
            idx = (np.abs(t - tm)).argmin()
            xs.append(x[idx])
            ys.append(y[idx])
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                plt.plot([xs[i], xs[j]], [ys[i], ys[j]], 'k--', linewidth=1.0, alpha=1.0)

    # Plot akhir
    plt.axis('equal') 
    plt.xlim((-7, 25))
    plt.ylim((-3, 7))   
    # plt.xlim((-10, 10))
    # plt.ylim((-16, 10))
    # plt.xlim((-7, 15))
    # plt.ylim((-6, 6))
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    # plt.title(f'Swarm Trajectory\nObs: {obstacle_scheme} - Wind: {wind_type} - Contr: {controller_type}\nFormation: {formation_type} - Orient: {orient}', fontsize=12)
    plt.grid()
    plt.legend(fontsize=10, ncols=2, loc='lower right')
    ax = plt.gca()
    textstr = f'Total Time: {t_max:.3f}s'
    ax.text(0.98, 0.90, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgray'))
    plt.tight_layout()
    plt.show()

def plot_speed_per_mode(data_quad, data_agent, config, **plot_info):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    num_agents = len(data_quad)
    time = data_quad[0]['path'][:, 0]
    mode_history = data_agent[0]['path'][:, 7]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    fig, ax = plt.subplots(figsize=(6.85, 2.85))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10

    # --- Plot garis kecepatan untuk setiap agen ---
    quad_handles = []
    for i in range(num_agents):
        path = data_quad[i]['path']
        speed = np.linalg.norm(path[:, 11:14], axis=1)
        line, = ax.plot(time, speed, label=f'Quad {i+1}', color=colors[i % len(colors)], linewidth=1.8)
        quad_handles.append(line)
    
    # Mean speed seluruh swarm
    mean_speed = np.mean([np.linalg.norm(data_quad[i]['path'][:, 11:14], axis=1) for i in range(num_agents)], axis=0)

    # --- Tambahkan latar belakang warna per mode ---
    mode_colors = {
        -1: {'color': "#7aa1ba", 'label': 'Takeoff'},
         0: {'color': "#f9d7b9", 'label': 'Formation'},
         1: {'color': "#a4c1a4", 'label': 'Tailgating'}
    }

    mode_handles = []
    current_mode = mode_history[0]
    start_time = time[0]
    added_modes = set()

    for i in range(1, len(time)):
        if mode_history[i] != current_mode:
            end_time = time[i]
            mode_info = mode_colors[current_mode]
            ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
            if mode_info['label'] not in added_modes:
                patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=mode_info['label'])
                mode_handles.append(patch)
                added_modes.add(mode_info['label'])
            current_mode = mode_history[i]
            start_time = end_time

    # Tambahkan blok terakhir
    end_time = time[-1]
    mode_info = mode_colors[current_mode]
    ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
    if mode_info['label'] not in added_modes:
        patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=mode_info['label'])
        mode_handles.append(patch)

    # --- Plot settings ---
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    ax.set_title(f'Swarm Speed per Mode\n'
                 f'Mean Speed: {np.mean(mean_speed):.3f} m/s\n'
                 f'Obs: {plot_info.get("obstacle_scheme", "N/A")}, '
                 f'Wind: {plot_info.get("wind_type", "N/A")}, '
                 f'Contr: {plot_info.get("controller_type", "N/A")}', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(bottom=0)

    # --- Gabungkan legend mode + quad dalam satu blok ---
    combined_handles = mode_handles + quad_handles
    combined_labels = [h.get_label() for h in combined_handles]
    ax.legend(combined_handles, combined_labels, loc='upper right', ncol=4, fontsize=12)

    plt.tight_layout()
    plt.show()



def plot_quad_data_quad(arr_quad, quad_id=None):
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
    judul_prefix = f"Quadcopter {quad_id} – " if quad_id is not None else ""

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


def calculate_takeoff_rmse_history(data_quad, target_altitude):
    """
    Menghitung riwayat RMSE ketinggian selama fase lepas landas.
    """
    num_agents = len(data_quad)
    num_time_steps = len(data_quad[0]['path'])
    time_vector = data_quad[0]['path'][:, 0]
    rmse_history = []

    for t_idx in range(num_time_steps):
        squared_errors = []
        for i in range(num_agents):
            # Ambil posisi z (ketinggian) aktual
            z_actual = data_quad[i]['path'][t_idx, 3] # Asumsi kolom ke-3 adalah z
            error = z_actual - target_altitude
            squared_errors.append(error**2)
        
        # Hitung Mean Squared Error dari eror ketinggian
        mean_squared_error = np.mean(squared_errors)
        
        # Hitung Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error)
        rmse_history.append(rmse)
        
    return time_vector, np.array(rmse_history)

def calculate_formation_rmse_history(data_quad, topology):
    """
    Menghitung riwayat RMSE dari bentuk formasi.
    """
    num_agents = len(data_quad)
    if num_agents < 2:
        return np.array([]), np.array([])

    num_unique_pairs = num_agents * (num_agents - 1) / 2
    num_time_steps = len(data_quad[0]['path'])
    time_vector = data_quad[0]['path'][:, 0]
    rmse_history = []

    for t_idx in range(num_time_steps):
        sum_of_squared_errors = 0.0
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                pos_i_actual = data_quad[i]['path'][t_idx, 1:4]
                pos_j_actual = data_quad[j]['path'][t_idx, 1:4]
                actual_distance = np.linalg.norm(pos_i_actual - pos_j_actual)
                
                pos_i_ideal = topology[i, :]
                pos_j_ideal = topology[j, :]
                ideal_distance = np.linalg.norm(pos_i_ideal - pos_j_ideal)
                
                sum_of_squared_errors += (actual_distance - ideal_distance)**2
        
        mean_squared_error = sum_of_squared_errors / num_unique_pairs
        rmse = np.sqrt(mean_squared_error)
        rmse_history.append(rmse)
        
    return time_vector, np.array(rmse_history)

def calculate_tailgating_rmse_history(data_quad, ideal_following_distance):
    """
    Menghitung riwayat RMSE dari jarak pengekoran (tailgating).
    """
    num_agents = len(data_quad)
    num_time_steps = len(data_quad[0]['path'])
    time_vector = data_quad[0]['path'][:, 0]
    rmse_history = []

    for t_idx in range(num_time_steps):
        squared_errors = []
        
        # Ekstrak posisi dan kecepatan semua agen pada waktu t
        positions = np.array([agent['path'][t_idx, 1:4] for agent in data_quad])
        velocities = np.array([agent['path'][t_idx, 4:7] for agent in data_quad])
        
        # Tentukan arah gerak kawanan (rata-rata kecepatan)
        swarm_velocity = np.mean(velocities, axis=0)
        norm_swarm_velocity = np.linalg.norm(swarm_velocity)
        if norm_swarm_velocity < 1e-6:
            # Jika kawanan diam, gunakan arah default atau arah dari frame sebelumnya
            # Untuk simplifikasi, kita asumsikan tidak ada eror jika diam
            rmse_history.append(0)
            continue
        
        swarm_direction = swarm_velocity / norm_swarm_velocity

        # Loop untuk setiap agen untuk mencari pemimpinnya
        for i in range(num_agents):
            best_leader_idx = -1
            min_distance_in_front = float('inf')
            
            # Cari pemimpin di antara agen lain
            for j in range(num_agents):
                if i == j:
                    continue
                
                vec_to_j = positions[j] - positions[i]
                dist_in_front = np.dot(vec_to_j, swarm_direction)
                
                # Cek apakah agen j ada di depan dan lebih dekat dari pemimpin sebelumnya
                if 0 < dist_in_front < min_distance_in_front:
                    min_distance_in_front = dist_in_front
                    best_leader_idx = j
            
            # Jika agen i punya pemimpin (bukan yang paling depan)
            if best_leader_idx != -1:
                actual_distance = np.linalg.norm(positions[i] - positions[best_leader_idx])
                error = actual_distance - ideal_following_distance
                squared_errors.append(error**2)

        if not squared_errors: # Jika tidak ada follower (misal hanya 1 agen)
            rmse_history.append(0)
            continue
            
        # Hitung RMSE hanya dari agen-agen yang menjadi follower
        mean_squared_error = np.mean(squared_errors)
        rmse = np.sqrt(mean_squared_error)
        rmse_history.append(rmse)
        
    return time_vector, np.array(rmse_history)

def plot_RMSE_total(data_quad, data_agent, topology, config, **plot_info):
    import matplotlib.patches as mpatches

    time, takeoff_rmse = calculate_takeoff_rmse_history(data_quad, config.TARGET_ALTITUDE)
    _, formation_rmse = calculate_formation_rmse_history(data_quad, topology)
    _, tailgating_rmse = calculate_tailgating_rmse_history(data_quad, config.DREF)

    mode_history = data_agent[0]['path'][:, 7]
    total_rmse = np.zeros_like(time)

    for i in range(len(time)):
        current_mode = mode_history[i]
        if current_mode == -1:
            total_rmse[i] = takeoff_rmse[i]
        elif current_mode == 0:
            total_rmse[i] = formation_rmse[i]
        elif current_mode == 1:
            total_rmse[i] = tailgating_rmse[i]

    scalar_rmse = np.mean(total_rmse)

    fig, ax = plt.subplots(figsize=(6.85, 2.85))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    line_rmse, = ax.plot(time, total_rmse, 'k-', label='Operational RMSE (m)', linewidth=2.0)

    # Background warna
    mode_colors = {
        -1: {'color': "#7aa1ba", 'label': 'Takeoff'},
         0: {'color': "#f9d7b9", 'label': 'Formation'},
         1: {'color': "#a4c1a4", 'label': 'Tailgating'}
    }

    current_mode = mode_history[0]
    start_time = time[0]
    drawn_labels = set()
    mode_patches = []

    for i in range(1, len(time)):
        if mode_history[i] != current_mode:
            end_time = time[i]
            mode_info = mode_colors[current_mode]
            label = mode_info['label']
            ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
            if label not in drawn_labels:
                patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=label)
                mode_patches.append(patch)
                drawn_labels.add(label)
            current_mode = mode_history[i]
            start_time = end_time

    # Tambahkan blok terakhir
    end_time = time[-1]
    mode_info = mode_colors[current_mode]
    label = mode_info['label']
    ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
    if label not in drawn_labels:
        patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=label)
        mode_patches.append(patch)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('RMSE (meters)', fontsize=12)
    ax.set_title(
        f"Operational Accuracy RMSE by Mode\n"
        f"$\\bf{{\\overline{{RMSE}}_{{total}} = {scalar_rmse:.3f}}}$ meters\n"
        f"Obs: {plot_info.get('obstacle_scheme', 'N/A')}, Wind: {plot_info.get('wind_type', 'N/A')}, Contr: {plot_info.get('controller_type', 'N/A')}",
        fontsize=12
    )
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(bottom=-0.1)

    # Gabungkan legend
    handles = mode_patches + [line_rmse]
    ax.legend(handles=handles, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_order_total(data_quad, data_agent, config, **plot_info):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    time = data_quad[0]['path'][:, 0]
    mode_history = data_agent[0]['path'][:, 7]

    # Hitung indeks keteraturan (Phi)
    headings = []
    for i in range(1, len(time)):
        heading_sum = np.zeros(2)
        for j in range(len(data_quad)):
            v_xy = data_quad[j]['path'][i, 11:13]  # v_x, v_y
            norm = np.linalg.norm(v_xy)
            if norm > 1e-6:
                heading_sum += v_xy / norm
        order_phi = np.linalg.norm(heading_sum) / len(data_quad)
        headings.append(order_phi)

    time_order = time[1:]  # karena mulai dari i=1
    phi_array = np.array(headings)
    avg_phi = np.mean(phi_array)

    # Setup plot
    fig, ax = plt.subplots(figsize=(6.85, 2.85))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    ax.plot(time_order, phi_array, 'k-', linewidth=2.0, label='Order Index ($\\Phi$)')

    # --- Background warna berdasarkan mode ---
    mode_colors = {
        -1: {'color': "#7aa1ba", 'label': 'Takeoff'},
         0: {'color': "#f9d7b9", 'label': 'Formation'},
         1: {'color': "#a4c1a4", 'label': 'Tailgating'}
    }

    current_mode = mode_history[1]  # karena time_order dimulai dari indeks 1
    start_time = time_order[0]
    drawn_labels = set()
    mode_handles = []

    for i in range(1, len(time_order)):
        if mode_history[i + 1] != current_mode:
            end_time = time_order[i]
            mode_info = mode_colors.get(current_mode)
            label = mode_info['label'] if mode_info['label'] not in drawn_labels else ""
            ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
            if label:
                patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=label)
                mode_handles.append(patch)
                drawn_labels.add(label)
            current_mode = mode_history[i + 1]
            start_time = end_time

    # Tambahkan blok terakhir
    end_time = time_order[-1]
    mode_info = mode_colors.get(current_mode)
    label = mode_info['label'] if mode_info['label'] not in drawn_labels else ""
    ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
    if label:
        patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=label)
        mode_handles.append(patch)

    # Finalize
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Order Index ($\\Phi$)", fontsize=12)
    ax.set_title(
        f"System Order Index by Mode\n"
        f"$\\bf{{\\overline{{\\Phi}} = {avg_phi:.3f}}}$\n"
        f"Obs: {plot_info.get('obstacle_scheme', 'N/A')}, "
        f"Wind: {plot_info.get('wind_type', 'N/A')}, "
        f"Contr: {plot_info.get('controller_type', 'N/A')}",
        fontsize=12
    )
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(time_order[0], time_order[-1])
    ax.set_ylim(0, 1.05)

    # Legend: mode (blocks) + Phi line
    phi_line = plt.Line2D([], [], color='k', linewidth=2.0, label='Order Index ($\\Phi$)')
    handles = mode_handles + [phi_line]
    ax.legend(handles=handles, loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.show()


# Asumsikan Anda sudah punya:
# data_quad, config, topology
num_robot = 5
multi_agent_config.set_num_robot(num_robot)
topology = multi_agent_config.TOPOLOGY

# --- Contoh Cara Pemanggilan Fungsi ---
plot_RMSE_total(
    data_quad, 
    data_agent,
    topology, 
    multi_agent_config,
    obstacle_scheme=obstacle_scheme, 
    wind_type=wind_type,
    controller_type=controller_type,
)
plot_order_total(
    data_quad, 
    data_agent, 
    multi_agent_config,
    obstacle_scheme=obstacle_scheme, 
    wind_type=wind_type, 
    controller_type=controller_type
)


# plot_paths(data_quad)
# plot_formation_error(data_quad)
# plot_speed_per_mode(
#     data_quad, 
#     data_agent, 
#     multi_agent_config,
#     obstacle_scheme=obstacle_scheme, 
#     wind_type=wind_type, 
#     controller_type=controller_type
# )
# plot_order(data_quad)

# ----------------------------------------------------------
# Loop melalui seluruh elemen data_quad_quad
# ----------------------------------------------------------
# if len(data_quad) == 0:
#     print("Tidak ada data_quad_quad untuk diplot.")
# else:
#     for entry in data_quad:
#         quad_id = entry.get('id', None)
#         arr_quad = entry['path']
#         print(f"Plotting data_quad for quadcopter {quad_id}")
#         plot_quad_data_quad(arr_quad, quad_id)