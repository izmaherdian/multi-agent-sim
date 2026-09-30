import matplotlib
matplotlib.use('Agg')

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path to import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.config import MultiAgentConfig
from environment.obstacles import Obstacles

def get_circle(x, y, r):
    theta = np.linspace(0, 2*np.pi, 100)
    return x + r * np.cos(theta), y + r * np.sin(theta)

def plot_paths_to_file(data_quad, obstacle_scheme, wind_type, controller_type, formation_type, orient, output_path):
    plt.figure(figsize=(6.85, 2.85))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    
    obstacles = Obstacles(scheme=obstacle_scheme)
    obstacles_2d = obstacles.obstacles_2d

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    for i in range(len(obstacles_2d)):
        obstacle = obstacles_2d[i]
        if i == 0:
            plt.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.3, color='grey', label='Obstacle')
        else:
            plt.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.3, color='grey')

    all_times = data_quad[0]['path'][:, 0]
    t_max = all_times[-1]
    time_marks = np.arange(0, t_max + 1, 10)

    for i, data in enumerate(data_quad):
        path = data['path']
        t = path[:, 0]
        x = path[:, 1]
        y = path[:, 2]
        color = colors[i % len(colors)]

        plt.plot(x, y, label=f'Quad {i+1}', linewidth=1.5, color=color)

        for tm in time_marks:
            idx = (np.abs(t - tm)).argmin()
            cx, cy = get_circle(x[idx], y[idx], 0.15)
            plt.plot(cx, cy, color=color, linewidth=1.2, alpha=1.0)

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
                plt.plot([xs[i], xs[j]], [ys[i], ys[j]], 'k--', linewidth=1.0, alpha=0.5)

    plt.axis('equal') 
    plt.xlim((-7, 25))
    plt.ylim((-3, 7))   
    plt.xlabel('X Position (m)', fontsize=11)
    plt.ylabel('Y Position (m)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, ncols=2, loc='lower right')
    
    ax = plt.gca()
    textstr = f'Total Time: {t_max:.3f}s'
    ax.text(0.98, 0.90, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def calculate_takeoff_rmse_history(data_quad, target_altitude):
    num_agents = len(data_quad)
    num_time_steps = len(data_quad[0]['path'])
    time_vector = data_quad[0]['path'][:, 0]
    rmse_history = []

    for t_idx in range(num_time_steps):
        squared_errors = []
        for i in range(num_agents):
            z_actual = data_quad[i]['path'][t_idx, 3]
            error = z_actual - target_altitude
            squared_errors.append(error**2)
        mean_squared_error = np.mean(squared_errors)
        rmse = np.sqrt(mean_squared_error)
        rmse_history.append(rmse)
    return time_vector, np.array(rmse_history)

def calculate_formation_rmse_history(data_quad, topology):
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
    num_agents = len(data_quad)
    num_time_steps = len(data_quad[0]['path'])
    time_vector = data_quad[0]['path'][:, 0]
    rmse_history = []

    for t_idx in range(num_time_steps):
        squared_errors = []
        positions = np.array([agent['path'][t_idx, 1:4] for agent in data_quad])
        velocities = np.array([agent['path'][t_idx, 4:7] for agent in data_quad])
        
        swarm_velocity = np.mean(velocities, axis=0)
        norm_swarm_velocity = np.linalg.norm(swarm_velocity)
        if norm_swarm_velocity < 1e-6:
            rmse_history.append(0)
            continue
        
        swarm_direction = swarm_velocity / norm_swarm_velocity

        for i in range(num_agents):
            best_leader_idx = -1
            min_distance_in_front = float('inf')
            
            for j in range(num_agents):
                if i == j:
                    continue
                vec_to_j = positions[j] - positions[i]
                dist_in_front = np.dot(vec_to_j, swarm_direction)
                if 0 < dist_in_front < min_distance_in_front:
                    min_distance_in_front = dist_in_front
                    best_leader_idx = j
            
            if best_leader_idx != -1:
                actual_distance = np.linalg.norm(positions[i] - positions[best_leader_idx])
                error = actual_distance - ideal_following_distance
                squared_errors.append(error**2)

        if not squared_errors:
            rmse_history.append(0)
            continue
            
        mean_squared_error = np.mean(squared_errors)
        rmse = np.sqrt(mean_squared_error)
        rmse_history.append(rmse)
    return time_vector, np.array(rmse_history)

def plot_RMSE_total_to_file(data_quad, data_agent, topology, config, output_path, **plot_info):
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
            mode_info = mode_colors.get(current_mode, {'color': 'white', 'label': 'Unknown'})
            label = mode_info['label']
            ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
            if label not in drawn_labels and label != 'Unknown':
                patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=label)
                mode_patches.append(patch)
                drawn_labels.add(label)
            current_mode = mode_history[i]
            start_time = end_time

    end_time = time[-1]
    mode_info = mode_colors.get(current_mode, {'color': 'white', 'label': 'Unknown'})
    label = mode_info['label']
    ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
    if label not in drawn_labels and label != 'Unknown':
        patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=label)
        mode_patches.append(patch)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('RMSE (meters)', fontsize=11)
    ax.set_title(
        f"Operational Accuracy RMSE by Mode\n"
        f"Average RMSE = {scalar_rmse:.3f} m | Controller: {plot_info.get('controller_type', 'N/A').upper()}",
        fontsize=11
    )
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(bottom=-0.1)

    handles = mode_patches + [line_rmse]
    ax.legend(handles=handles, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_order_total_to_file(data_quad, data_agent, config, output_path, **plot_info):
    time = data_quad[0]['path'][:, 0]
    mode_history = data_agent[0]['path'][:, 7]

    headings = []
    for i in range(1, len(time)):
        heading_sum = np.zeros(2)
        for j in range(len(data_quad)):
            v_xy = data_quad[j]['path'][i, 11:13]
            norm = np.linalg.norm(v_xy)
            if norm > 1e-6:
                heading_sum += v_xy / norm
        order_phi = np.linalg.norm(heading_sum) / len(data_quad)
        headings.append(order_phi)

    time_order = time[1:]
    phi_array = np.array(headings)
    avg_phi = np.mean(phi_array)

    fig, ax = plt.subplots(figsize=(6.85, 2.85))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    
    ax.plot(time_order, phi_array, 'k-', linewidth=2.0, label='Order Index ($\\Phi$)')

    mode_colors = {
        -1: {'color': "#7aa1ba", 'label': 'Takeoff'},
         0: {'color': "#f9d7b9", 'label': 'Formation'},
         1: {'color': "#a4c1a4", 'label': 'Tailgating'}
    }

    current_mode = mode_history[1]
    start_time = time_order[0]
    drawn_labels = set()
    mode_handles = []

    for i in range(1, len(time_order)):
        if mode_history[i + 1] != current_mode:
            end_time = time_order[i]
            mode_info = mode_colors.get(current_mode, {'color': 'white', 'label': 'Unknown'})
            label = mode_info['label'] if mode_info['label'] not in drawn_labels else ""
            ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
            if label and label != 'Unknown':
                patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=label)
                mode_handles.append(patch)
                drawn_labels.add(label)
            current_mode = mode_history[i + 1]
            start_time = end_time

    end_time = time_order[-1]
    mode_info = mode_colors.get(current_mode, {'color': 'white', 'label': 'Unknown'})
    label = mode_info['label'] if mode_info['label'] not in drawn_labels else ""
    ax.axvspan(start_time, end_time, color=mode_info['color'], alpha=0.2)
    if label and label != 'Unknown':
        patch = mpatches.Patch(color=mode_info['color'], alpha=0.2, label=label)
        mode_handles.append(patch)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Order Index ($\\Phi$)", fontsize=11)
    ax.set_title(
        f"System Order Index by Mode\n"
        f"Average Order Index = {avg_phi:.3f} | Controller: {plot_info.get('controller_type', 'N/A').upper()}",
        fontsize=11
    )
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(time_order[0], time_order[-1])
    ax.set_ylim(0, 1.05)

    phi_line = plt.Line2D([], [], color='k', linewidth=2.0, label='Order Index ($\\Phi$)')
    handles = mode_handles + [phi_line]
    ax.legend(handles=handles, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def _calculate_average_effort(data):
    num_agents = len(data)
    num_time_steps = len(data[0]['path'])
    time_vector = data[0]['path'][:, 0]
    total_effort = np.zeros(num_time_steps)

    for t_idx in range(num_time_steps):
        current_step_effort_sum = 0.0
        for i in range(num_agents):
            wM1 = data[i]['path'][t_idx, 17]
            wM2 = data[i]['path'][t_idx, 18]
            wM3 = data[i]['path'][t_idx, 19]
            wM4 = data[i]['path'][t_idx, 20]
            current_step_effort_sum += (wM1**2 + wM2**2 + wM3**2 + wM4**2)
        total_effort[t_idx] = current_step_effort_sum / num_agents

    return time_vector, total_effort

def plot_control_effort_comparison_to_file(data_erc_none, data_erc_gust, data_agent_none, output_path):
    time_no_wind, effort_no_wind = _calculate_average_effort(data_erc_none)
    time_wind, effort_wind = _calculate_average_effort(data_erc_gust)
    mode_history = data_agent_none[0]['path'][:, 7]

    mode_colors = {
        -1: {'color': '#7aa1ba', 'label': 'Takeoff'},
         0: {'color': '#f9d7b9', 'label': 'Formation'},
         1: {'color': '#a4c1a4', 'label': 'Tailgating'}
    }

    fig, ax = plt.subplots(figsize=(6.85, 2.85))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10

    current_mode = mode_history[0]
    start_time = time_no_wind[0]
    drawn_labels = set()
    mode_patches = []

    for i in range(1, len(time_no_wind)):
        if mode_history[i] != current_mode:
            end_time = time_no_wind[i]
            info = mode_colors.get(current_mode, {'color': 'white', 'label': 'Unknown'})
            ax.axvspan(start_time, end_time, color=info['color'], alpha=0.2)
            if info['label'] not in drawn_labels and info['label'] != 'Unknown':
                patch = mpatches.Patch(color=info['color'], alpha=0.2, label=info['label'])
                mode_patches.append(patch)
                drawn_labels.add(info['label'])
            current_mode = mode_history[i]
            start_time = end_time

    end_time = time_no_wind[-1]
    info = mode_colors.get(current_mode, {'color': 'white', 'label': 'Unknown'})
    ax.axvspan(start_time, end_time, color=info['color'], alpha=0.2)
    if info['label'] not in drawn_labels and info['label'] != 'Unknown':
        patch = mpatches.Patch(color=info['color'], alpha=0.2, label=info['label'])
        mode_patches.append(patch)

    line1, = ax.plot(time_no_wind, effort_no_wind, label='Without Wind Disturbance', color='b')
    line2, = ax.plot(time_wind, effort_wind, label='With Wind Disturbance (GUST)', color='r')

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Avg Control Effort ($\\Sigma \\omega_M^2$)", fontsize=11)
    ax.set_title("Average Control Effort Comparison (ERC, Scheme 2)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(time_no_wind[0], time_no_wind[-1])

    handles = mode_patches + [line1, line2]
    ax.legend(handles=handles, loc='upper right', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    config = MultiAgentConfig()
    config.set_num_robot(5)
    topology = config.TOPOLOGY
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("Loading ERC Scheme 2 None...")
    with open(os.path.join(workspace_dir, 'results', 'data', 'multi_quad_data_scheme2_NONE_erc_formation1_NED.pkl'), 'rb') as f:
        erc_none_quad = pickle.load(f)['data_quad']
    with open(os.path.join(workspace_dir, 'results', 'data', 'multi_agent_data_scheme2_NONE_erc_formation1_NED.pkl'), 'rb') as f:
        erc_none_agent = pickle.load(f)['data_agent']
        
    print("Loading IAPF Scheme 2 None...")
    with open(os.path.join(workspace_dir, 'results', 'data', 'multi_quad_data_scheme2_NONE_iapf_formation1_NED.pkl'), 'rb') as f:
        iapf_none_quad = pickle.load(f)['data_quad']
    with open(os.path.join(workspace_dir, 'results', 'data', 'multi_agent_data_scheme2_NONE_iapf_formation1_NED.pkl'), 'rb') as f:
        iapf_none_agent = pickle.load(f)['data_agent']
        
    print("Loading ERC Scheme 2 Gust...")
    with open(os.path.join(workspace_dir, 'results', 'data', 'multi_quad_data_scheme2_GUST_erc_formation1_NED.pkl'), 'rb') as f:
        erc_gust_quad = pickle.load(f)['data_quad']
    
    print("Generating Trajectory Plots...")
    plot_paths_to_file(erc_none_quad, 'scheme2', 'NONE', 'erc', 1, 'NED', 
                       os.path.join(image_dir, 'trajectory_erc_scheme2.png'))
    plot_paths_to_file(iapf_none_quad, 'scheme2', 'NONE', 'iapf', 1, 'NED', 
                       os.path.join(image_dir, 'trajectory_iapf_scheme2.png'))
                       
    print("Generating RMSE Plots...")
    plot_RMSE_total_to_file(erc_none_quad, erc_none_agent, topology, config, 
                            os.path.join(image_dir, 'rmse_erc_scheme2.png'), 
                            controller_type='erc', obstacle_scheme='scheme2', wind_type='NONE')
                            
    print("Generating Order Index Plots...")
    plot_order_total_to_file(erc_none_quad, erc_none_agent, config, 
                             os.path.join(image_dir, 'order_erc_scheme2.png'), 
                             controller_type='erc', obstacle_scheme='scheme2', wind_type='NONE')
                             
    print("Generating Control Effort Plots...")
    plot_control_effort_comparison_to_file(erc_none_quad, erc_gust_quad, erc_none_agent, 
                                           os.path.join(image_dir, 'control_effort_comparison.png'))
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
