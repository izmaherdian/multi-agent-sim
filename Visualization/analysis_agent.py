import pickle
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent.MultiAgentConfig import MultiAgentConfig
from Environment.Obstacles import Obstacles

multi_agent_config = MultiAgentConfig()
# obstacle_scheme = multi_agent_config.OBSTACLE_SCHEME
obstacle_scheme = 'scheme2'
# wind_type       = multi_agent_config.WIND_TYPE
wind_type       = 'GUST'
# controller_type = multi_agent_config.CONTROLLER
controller_type = 'erc'
# formation_type  = multi_agent_config.FORMATION_TYPE
formation_type  = 1
orient          = multi_agent_config.ORIENT

with open(f'multi_agent_data_{obstacle_scheme}_{wind_type}_{controller_type}_formation{formation_type}_{orient}.pkl', 'rb') as file:
    loaded = pickle.load(file)
print(loaded.keys())
data = loaded['data_agent']

obstacles = Obstacles(scheme=obstacle_scheme)
obstacles_2d = obstacles.obstacles_2d

def plot_paths(data):
    plt.figure(figsize=(10, 6))

    for i in range(len(obstacles_2d)):
        obstacle = obstacles_2d[i]
        plt.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.3, label=f'Obstacle {i+1}', color='grey')

    for i in range(len(data)):
        path = data[i]['path']
        x = path[:, 1]
        y = path[:, 2]
        plt.plot(x, y, label=f'Agent {i+1}', linewidth=1.5)
    
    plt.axis('scaled')
    plt.xlim((-5, 25))
    plt.ylim((-5, 10))   
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Agent Paths')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_speed(data):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    path = data[0]['path']
    size = 10
    x = np.arange(0, path.shape[0], size)

    num_formation = np.zeros_like(path[:, 0])
    num_tailgating = np.zeros_like(path[:, 0])
    num_takeoff = np.zeros_like(path[:, 0])

    for i in range(len(data)):
        num_formation += (data[i]['path'][:, 7] == 0)
        num_tailgating += (data[i]['path'][:, 7] == 1)
        num_takeoff += (data[i]['path'][:, 7] == -1)

        path = data[i]['path']
        t = path[:, 0]
        speed = (path[:, 4]**2 + path[:, 5]**2 + path[:, 6]**2)**0.5
        plt.plot(t, speed, label=f'Agent {i+1}', linewidth=1.5)
    
    num_formation = num_formation[x]
    num_tailgating = num_tailgating[x]
    num_takeoff = num_takeoff[x]
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')

    ax.set_title('Agent Speed Over Time')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_mode(data):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)
    size = 10
    path = data[0]['path']
    x = np.arange(0, path.shape[0], size)

    num_formation = np.zeros_like(path[:, 0])
    num_tailgating = np.zeros_like(path[:, 0])
    num_takeoff = np.zeros_like(path[:, 0])

    for i in range(len(data)):
        num_formation += (data[i]['path'][:, 7] == 0)
        num_tailgating += (data[i]['path'][:, 7] == 1)
        num_takeoff += (data[i]['path'][:, 7] == -1)

    num_formation = num_formation[x]
    num_tailgating = num_tailgating[x]
    num_takeoff = num_takeoff[x]

    scales = []
    for i in range(len(data)):
        scale = data[i]['path'][:, 8]
        scale[scale == -1] = 0
        scales.append(scale)
    scales = np.array(scales).T

    # Gunakan warna yang konsisten
    ax.bar(path[:, 0][x], num_takeoff, label="Takeoff", color='#cee5f5')
    ax.bar(path[:, 0][x], num_formation, label="Formation", color='#ffe5cf')
    ax.bar(path[:, 0][x], num_tailgating, label="Tailgating", color='#d5edd5')

    ax2.fill_between(path[:, 0][x], np.min(scales, axis=1)[x], np.max(scales, axis=1)[x], color="k", label="Max/Min", alpha=0.3)
    ax2.plot(path[:, 0][x], np.mean(scales, axis=1)[x], 'k-', label="Rata-rata")

    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Jumlah Agen")
    ax2.set_ylabel("Faktor Skala")

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(bottom=False, labelbottom=False)

    plt.xlim([0, path[-1, 0]])
    ax.set_ylim([0, len(data)])
    ax2.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Mode Agen dan Faktor Skala")
    plt.show()


def plot_order(data):
    plt.figure(figsize=(10, 6))
    path = data[0]['path']
    headings = []
    
    for i in range(1,len(path)):
        heading = 0
        for j in range(len(data)):
            heading += data[j]['path'][i,4:6]/np.linalg.norm(data[j]['path'][i,4:6])
        headings.append(np.linalg.norm(heading)/len(data))
    plt.plot(path[1:,0], headings, 'b-')
    plt.xlabel("Time (s)")
    plt.ylabel("Order")
    plt.xlim([0, path[-1,0]])
    plt.ylim(0,1.1)
    plt.tight_layout()
    plt.grid(True)
    plt.title("Order of Agents")
    plt.show()

# plot_paths(data)
plot_speed(data)
# plot_mode(data)
# plot_order(data)
