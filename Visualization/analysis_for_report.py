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
obstacle_scheme = 'scheme2'
# wind_type       = multi_agent_config.WIND_TYPE
wind_type       = 'NONE'
# controller_type = multi_agent_config.CONTROLLER
controller_type = 'erc'  
# formation_type  = multi_agent_config.FORMATION_TYPE
formation_type  = 1
orient          = multi_agent_config.ORIENT

with open(f'multi_quad_data_{obstacle_scheme}_NONE_erc_formation{formation_type}_{orient}.pkl', 'rb') as file:
    loaded = pickle.load(file)
print(loaded.keys())
data_erc_none = loaded['data_quad']

with open(f'multi_quad_data_{obstacle_scheme}_GUST_erc_formation{formation_type}_{orient}.pkl', 'rb') as file:
    loaded = pickle.load(file)
print(loaded.keys())
data_erc_gust = loaded['data_quad']

with open(f'multi_agent_data_{obstacle_scheme}_NONE_erc_formation{formation_type}_{orient}.pkl', 'rb') as file:
    loaded = pickle.load(file)
print(loaded.keys())
data_agent = loaded['data_agent']

def calculate_formation_error_history(data, topology):
    """
    Menghitung riwayat error formasi berdasarkan data simulasi.

    Args:
        data: Data simulasi yang dimuat dari file .pkl.
              Struktur: [{'path': array([...])}, {'path': array([...])}, ...]
        topology: Matriks topologi ideal dari MultiAgentConfig.

    Returns:
        tuple: (time_vector, error_vector)
    """
    num_agents = len(data)
    # Asumsikan semua agen memiliki jumlah langkah waktu yang sama
    num_time_steps = len(data[0]['path'])
    
    # Ambil vektor waktu dari agen pertama
    time_vector = data[0]['path'][:, 0]
    error_history = []

    # Loop untuk setiap langkah waktu dalam simulasi
    for t_idx in range(num_time_steps):
        current_total_error = 0.0
        
        # Loop untuk setiap pasangan agen unik (i, j)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Ambil posisi aktual agen i dan j pada waktu t
                pos_i_actual = data[i]['path'][t_idx, 1:4] # Kolom 1, 2, 3 adalah x, y, z
                pos_j_actual = data[j]['path'][t_idx, 1:4]
                
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

def plot_formation_error_comparison(data1, data2, label1='IAPF', label2='ERC'):
    """Plot perbandingan error formasi antara dua metode."""
    num_robot = 5
    multi_agent_config.set_num_robot(num_robot)
    topology = multi_agent_config.TOPOLOGY

    time1, error1 = calculate_formation_error_history(data1, topology)
    time2, error2 = calculate_formation_error_history(data2, topology)

    plt.figure(figsize=(10, 6))
    plt.plot(time1, error1, label=f'Error Formasi - {label1}', color='blue')
    plt.plot(time2, error2, label=f'Error Formasi - {label2}', color='orange')

    plt.xlabel('Waktu (s)')
    plt.ylabel('Error Formasi (kuadrat jarak)')
    plt.title('Perbandingan Error Formasi ($E_{formasi}$) Skenario 2 antara ERC dengan dan tanpa Angin')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_order_comparison(data1, data2, label1='IAPF', label2='ERC'):
    """Plot perbandingan order antar agen untuk dua metode."""
    plt.figure(figsize=(10, 6))

    def compute_order(data):
        path = data[0]['path']
        headings = []
        for i in range(1, len(path)):
            heading = 0
            for j in range(len(data)):
                v = data[j]['path'][i, 4:6]
                norm = np.linalg.norm(v)
                if norm != 0:
                    heading += v / norm
            headings.append(np.linalg.norm(heading) / len(data))
        return path[1:, 0], headings

    time1, order1 = compute_order(data1)
    time2, order2 = compute_order(data2)

    plt.plot(time1, order1, label=f'Order - {label1}', color='blue')
    plt.plot(time2, order2, label=f'Order - {label2}', color='orange')

    plt.xlabel("Waktu (s)")
    plt.ylabel("Order")
    plt.xlim([0, max(time1[-1], time2[-1])])
    plt.ylim(0, 1.1)
    plt.title("Perbandingan Order Agen Skenario 2 ERC dengan dan tanpa Angin")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def _calculate_average_effort(data):
    """
    Fungsi helper untuk menghitung rata-rata usaha kontrol dari satu set data.
    Mengembalikan vektor waktu dan vektor rata-rata usaha.
    """
    num_agents = len(data)
    if num_agents == 0:
        return np.array([]), np.array([])
        
    # Asumsikan semua agen punya jumlah waktu yang sama
    time_vector = data[0]['path'][:, 0]
    num_time_steps = len(time_vector)
    
    # Inisialisasi array untuk menyimpan total usaha pada setiap langkah waktu
    total_effort_per_timestep = np.zeros(num_time_steps)
    
    # Loop untuk setiap agen
    for j in range(num_agents):
        path = data[j]['path']
        
        commanded_motor_speeds = path[:, 17:20]
        
        # Hitung kuadrat dari setiap perintah kecepatan motor
        squared_speeds = commanded_motor_speeds**2
        
        # Jumlahkan usaha dari keempat motor untuk mendapatkan total usaha per agen
        effort_signal_per_agent = np.sum(squared_speeds, axis=1)
        
        # Tambahkan usaha agen ini ke total usaha kawanan
        total_effort_per_timestep += effort_signal_per_agent
        
    # Hitung rata-rata usaha di seluruh kawanan
    average_effort = total_effort_per_timestep / num_agents
    
    return time_vector, average_effort

import matplotlib.patches as mpatches

def plot_control_effort_comparison_with_mode(data_erc_none, data_erc_gust, data_agent, plot_info=None):
    """
    Mem-plot perbandingan usaha kontrol rata-rata antara dua skenario,
    dengan blok warna berdasarkan mode agent (takeoff, formation, tailgating).
    """
    if plot_info is None:
        plot_info = {}

    # Set font to Arial, size 12
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # Calculate data for both scenarios
    time_no_wind, effort_no_wind = _calculate_average_effort(data_erc_none)
    time_wind, effort_wind = _calculate_average_effort(data_erc_gust)

    # Ambil mode history dari salah satu agen
    mode_history = data_agent[0]['path'][:, 7]

    # Warna dan label untuk setiap mode
    mode_colors = {
        -1: {'color': '#7aa1ba', 'label': 'Takeoff'},
         0: {'color': '#f9d7b9', 'label': 'Formation'},
         1: {'color': '#a4c1a4', 'label': 'Tailgating'}
    }

    # Buat plot
    fig, ax = plt.subplots(figsize=(6.85, 2.85))

    # Tambahkan blok warna mode
    current_mode = mode_history[0]
    start_time = time_no_wind[0]
    drawn_labels = set()
    mode_patches = []

    for i in range(1, len(time_no_wind)):
        if mode_history[i] != current_mode:
            end_time = time_no_wind[i]
            info = mode_colors[current_mode]
            ax.axvspan(start_time, end_time, color=info['color'], alpha=0.2)
            if info['label'] not in drawn_labels:
                patch = mpatches.Patch(color=info['color'], alpha=0.2, label=info['label'])
                mode_patches.append(patch)
                drawn_labels.add(info['label'])
            current_mode = mode_history[i]
            start_time = end_time

    # Tambahkan blok terakhir
    end_time = time_no_wind[-1]
    info = mode_colors[current_mode]
    ax.axvspan(start_time, end_time, color=info['color'], alpha=0.2)
    if info['label'] not in drawn_labels:
        patch = mpatches.Patch(color=info['color'], alpha=0.2, label=info['label'])
        mode_patches.append(patch)

    # Plot control effort curves
    line1, = ax.plot(time_no_wind, effort_no_wind, label='Without Wind Disturbance', color='b')
    line2, = ax.plot(time_wind, effort_wind, label='With Wind Disturbance (GUST)', color='r')

    # Labels and decoration
    ax.set_xlabel("Time (s)", fontsize=12, fontname='Arial')
    ax.set_ylabel("Average Control Effort (Σ wM²)", fontsize=12, fontname='Arial')
    # ax.set_title(f'Control Effort Comparison\nObs: {plot_info.get("obstacle_scheme", "N/A")} - Wind: {plot_info.get("wind_type", "N/A")} - Contr: {plot_info.get("controller_type", "N/A")}\nFormation: {plot_info.get("formation_type", "N/A")} - Orient: {plot_info.get("orient", "N/A")}', fontsize=12, fontname='Arial')
    ax.grid(True)
    ax.set_xlim(time_no_wind[0], time_no_wind[-1])

    # Gabungkan legenda
    handles = mode_patches + [line1, line2]
    ax.legend(handles=handles, loc='upper right', fontsize=12, ncol=2)

    plt.tight_layout()
    plt.show()

# plot_formation_error_comparison(data_erc_none, data_erc_gust, label1='Tanpa Angin', label2='Dengan Angin')
# plot_order_comparison(data_erc_none, data_erc_gust, label1='Tanpa Angin', label2='Dengan Angin')

plot_control_effort_comparison_with_mode(
    data_erc_none, 
    data_erc_gust, 
    data_agent=loaded['data_agent'],
    plot_info={
        'obstacle_scheme': obstacle_scheme,
        'wind_type': wind_type,
        'controller_type': controller_type,
        'formation_type': formation_type,
        'orient': orient
    }
)