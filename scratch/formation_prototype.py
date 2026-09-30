import numpy as np
import math
from enum import Enum
# import pickle # Tidak diperlukan lagi

# --- Import dari file Anda yang lain ---
from environment.obstacles import Obstacles # Asumsikan Obstacles.py ada di direktori yang sama atau dapat diakses

# Fungsi bantuan untuk mendapatkan data agents_config yang mirip dengan main.py
def get_main_agents_config():
    agents_config = [
        {'id': 1, 'init_pos': [1.0, 1.0, 0.0], 'front_color': 'orange', 'ctrlType': "xyz_pos", 'trajSelect': (0, 3, 0)},
        {'id': 2, 'init_pos': [-3.0, 3.0, 0.0], 'front_color': 'blue', 'ctrlType': "xyz_pos", 'trajSelect': (0, 3, 0)},
        {'id': 3, 'init_pos': [-2.0, 2.0, 0.0], 'front_color': 'green', 'ctrlType': "xyz_pos", 'trajSelect': (0, 3, 0)},
        {'id': 4, 'init_pos': [-2.0, -2.0, 0.0], 'front_color': 'red', 'ctrlType': "xyz_pos", 'trajSelect': (0, 3, 0)},
        {'id': 5, 'init_pos': [-3.0, -3.0, 0.0], 'front_color': 'purple', 'ctrlType': "xyz_pos", 'trajSelect': (0, 3, 0)}
    ]
    return agents_config

# --- Dari utils.py ---
def perpendicular(x:np.array, a:np.array, b:np.array):
    d_ab = np.linalg.norm(a-b)
    d_ax = np.linalg.norm(a-x)
    d_bx = np.linalg.norm(b-x)
    if d_ab != 0:
        if np.dot(a-b,x-b)*np.dot(b-a,x-a) >= 0:
            px = b[0]-a[0]; py = b[1]-a[1]; dAB = px*px + py*py
            u = ((x[0] - a[0]) * px + (x[1] - a[1]) * py) / dAB
            p = [a[0] + u * px, a[1] + u * py]
        else:
            p = a if d_ax < d_bx else b
    else:
        p = a
    return p

def nearest_point_to_obstacle(pose, obstacle_geometry):
    nearest_point = []
    nearest_dis= float(np.inf)
    for i in range(len(obstacle_geometry)):
        per = perpendicular(pose, np.array(obstacle_geometry[i]), np.array(obstacle_geometry[(i+1) % len(obstacle_geometry)]))
        dis_per = np.linalg.norm(pose - per)
        if dis_per < nearest_dis:
            nearest_dis = dis_per
            nearest_point = per
    return nearest_point

class Config:
    def __init__(self):
        self.TIMESTEP = 0.05
        self.ROBOT_RADIUS = 0.2
        self.ALERT_RADIUS = 3 * self.ROBOT_RADIUS
        self.SENSING_RADIUS = 3.0
        self.ITER_MAX = 2000
        self.EPSILON = 0.1
        self.ALPHA = 8
        self.VREF = 0.5
        self.VMAX = 1.0
        self.UMAX = 2.0
        self.UREF = np.array([1,0,0])
        self.DREF = 1.0
        self.W_form = 1.0
        self.W_tail = 1.0
        self.W_obs = 8.0
        self.W_col = 8.0
        self.W_rand = 2e-2
        self.XLIM = [-20., 25.]
        self.YLIM = [  -5., 8.] 
        self.SIZE = [10, 3]

        agents_config_data = get_main_agents_config()
        self.INITS = np.array([config_item['init_pos'] for config_item in agents_config_data])
        self.NUM_ROBOT = self.INITS.shape[0] 
        self.XGOAL = 25.0
        self.YGOAL = 3.0
        self.ZGOAL = 1.0 

        self.FORMATION_TYPE = 2 
        if self.FORMATION_TYPE == 1:
            self.TOPOLOGY = np.array([[np.cos(2*np.pi/self.NUM_ROBOT*i), np.sin(2*np.pi/self.NUM_ROBOT*i), 0.0] for i in range(self.NUM_ROBOT)])
        elif self.FORMATION_TYPE == 2: 
            if self.NUM_ROBOT == 5: 
                self.TOPOLOGY = np.array([[ 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [ 0.0, 0.5, 0.0], [ 0.0,-0.5, 0.0], [-1.0,-1.0, 0.0]])
            else:
                print(f"Peringatan: Topologi V-shape untuk {self.NUM_ROBOT} robot tidak standar, menggunakan formasi garis.")
                self.TOPOLOGY = np.array([[float(-i) * 0.5, 0.0, 0.0] for i in range(self.NUM_ROBOT)])

        self.CONTROLLER = 'iapf'
        obstacle_provider = Obstacles(scheme='scheme1') 
        self.OBSTACLE_GEOMETRIES = [obs.tolist() for obs in obstacle_provider.obstacles_2d]
        self.OBSTACLE_HEIGHTS = list(obstacle_provider.obstacle_heights)
        
        self.ANIMATOR_TRAIL_LENGTH = 30 # Jumlah titik data terakhir yang ditampilkan sebagai jejak
        self.GIF_NAME = f"results/coba_gif_{self.CONTROLLER}_shape{self.FORMATION_TYPE}_trail{self.ANIMATOR_TRAIL_LENGTH}.gif"

class Mode(Enum):
    FORMATION = 0
    TAILGATING = 1

class RobotIAPF:
    def __init__(self, index, position, config, velocity=np.zeros(3)):
        self.index = index
        self.config = config 
        self.stamp = 0.0
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.control = np.zeros(3, dtype=float)
        self.mode = Mode.FORMATION
        self.path = [np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value]])]

    def update_state(self, control, dt):
        control_norm = np.linalg.norm(control)
        if control_norm > self.config.UMAX: control = control/control_norm*self.config.UMAX
        self.velocity = self.velocity + control*dt
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > self.config.VMAX: self.velocity = self.velocity/velocity_norm*self.config.VMAX
        self.position = self.position + self.velocity*dt
        self.stamp += dt
        self.control = control
        self.path.append(np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value]]))

    def compute_control(self, robots, environment, dt):
        v_mig = self.behavior_migration(robots)
        v_obs = self.behavior_obstacle(environment)
        v_col = self.behavior_collision(robots)
        v_form = self.behavior_formation(robots)
        v_rand = self.behavior_random()
        desired_velocity = v_mig + v_form + v_obs + v_col + v_rand
        desired_control = (desired_velocity - self.velocity)/dt
        self.update_state(desired_control, dt)
        print(f"Robot {self.index} - Posisi: {self.position}, Kecepatan: {self.velocity}, Kontrol: {self.control}, Mode: {self.mode.name}")

    def behavior_migration(self, robots):
        center = np.mean([robot.position for robot in robots], axis=0)
        goal = np.array([self.config.XGOAL, self.config.YGOAL, self.config.ZGOAL])
        error_vec = goal - center
        error_dist = np.linalg.norm(error_vec)
        speed = self.config.VREF
        if error_dist < 1.0 and error_dist > 0: speed *= (error_dist / 1.0)
        elif error_dist == 0: speed = 0
        if speed < 0.05: speed = 0
        direction = error_vec / error_dist if error_dist > 0 else np.zeros(3)
        return speed * direction
    
    def behavior_formation(self, robots):
        v_form = np.zeros(3) 
        for i in range(self.config.NUM_ROBOT):
            v_form += (robots[i].position - self.position) - (self.config.TOPOLOGY[i,:] - self.config.TOPOLOGY[self.index,:])
        return self.config.W_form * v_form
    
    def behavior_obstacle(self, environment):
        v_obs = np.zeros(3) 
        for obstacle in environment.obstacles: 
            obs_point = nearest_point_to_obstacle(self.position[:2], obstacle.geometry) 
            z_rel = 0
            if self.position[2] < obstacle.z_min: z_rel = obstacle.z_min - self.position[2] 
            elif self.position[2] > obstacle.z_max: z_rel = self.position[2] - obstacle.z_max 
            obs_rel_xy = self.position[:2] - obs_point
            horizontal_dist = np.linalg.norm(obs_rel_xy)
            obs_dis = np.sqrt(horizontal_dist**2 + z_rel**2) if horizontal_dist > 0 else abs(z_rel)
            if obs_dis == 0: obs_dis = 1e-6

            if obs_dis < self.config.ALERT_RADIUS:
                direction_xy = obs_rel_xy / horizontal_dist if horizontal_dist != 0 else np.zeros(2)
                magnitude = 0.5 * (1/obs_dis - 1/self.config.ALERT_RADIUS) / (obs_dis**2) 
                v_obs_component = magnitude * np.array([direction_xy[0], direction_xy[1], 0]) 
                if not np.any(np.isnan(v_obs_component)) and not np.any(np.isinf(v_obs_component)):
                    v_obs += v_obs_component
                elif horizontal_dist > 1e-6:
                     v_obs += 0.1 * np.array([direction_xy[0], direction_xy[1], 0]) / horizontal_dist 
        return self.config.W_obs * v_obs
    
    def behavior_collision(self, robots):
        v_col = np.zeros(3) 
        for i in range(self.config.NUM_ROBOT):
            if i == self.index: continue
            pos_rel = self.position - robots[i].position
            pos_dis = np.linalg.norm(pos_rel)
            if pos_dis < self.config.ALERT_RADIUS and pos_dis > 1e-6: 
                v_col_component = 2*(1/pos_dis - 1/self.config.ALERT_RADIUS)/(pos_dis**2)*pos_rel/pos_dis
                if not np.any(np.isnan(v_col_component)) and not np.any(np.isinf(v_col_component)):
                     v_col += v_col_component
        return self.config.W_col*v_col

    def behavior_random(self):
        return self.config.W_rand*np.concatenate([np.random.rand(2),[0]])

class Obstacle: 
    def __init__(self, geometry, height_range):
        self.geometry = geometry 
        self.z_min = height_range[0]
        self.z_max = height_range[1]

class Environment:
    def __init__(self, config):
        self.config = config
        self.robots = []
        self.obstacles = []
        self._initialize_obstacles()

    def _initialize_obstacles(self):
        for geo, height in zip(self.config.OBSTACLE_GEOMETRIES, self.config.OBSTACLE_HEIGHTS): 
            self.obstacles.append(Obstacle(geo, height)) 

    def add_robot(self, robot): self.robots.append(robot)

class Simulation:
    def __init__(self, config, environment):
        self.config = config
        self.environment = environment
        self.robots = []
        self._initialize_robots()

    def _initialize_robots(self):
        for i in range(self.config.NUM_ROBOT): 
            if self.config.CONTROLLER == 'iapf':
                robot = RobotIAPF(i, self.config.INITS[i,:], self.config) 
            else: raise ValueError(f"Tipe controller tidak diketahui: {self.config.CONTROLLER}")
            self.robots.append(robot)
            self.environment.add_robot(robot) 

    def check_collision(self):
        for i in range(self.config.NUM_ROBOT):
            robot_pos = self.robots[i].position
            for obstacle_obj in self.environment.obstacles: 
                if not (robot_pos[2] + self.config.ROBOT_RADIUS < obstacle_obj.z_min or \
                        robot_pos[2] - self.config.ROBOT_RADIUS > obstacle_obj.z_max):
                    obs_point_2d = nearest_point_to_obstacle(robot_pos[:2], obstacle_obj.geometry)
                    if np.linalg.norm(robot_pos[:2]-obs_point_2d) < self.config.ROBOT_RADIUS:
                        print(f"Tabrakan: Robot {i} dengan obstacle")
                        return True
        for i in range(self.config.NUM_ROBOT-1):
            for j in range(i+1, self.config.NUM_ROBOT):
                if np.linalg.norm(self.robots[i].position - self.robots[j].position) < 2*self.config.ROBOT_RADIUS:
                    print(f"Tabrakan: Robot {i} dengan Robot {j}")
                    return True
        return False

    def check_reach_goal(self, prev_center, stable_count):
        center = np.mean([robot.position for robot in self.robots], axis=0)
        avg_velocity = np.linalg.norm(np.sum([robot.velocity for robot in self.robots], axis=0)) / self.config.NUM_ROBOT
        goal = np.array([self.config.XGOAL, self.config.YGOAL, self.config.ZGOAL])
        dist_to_goal = np.linalg.norm(center - goal)
        new_stable_count = stable_count
        goal_reached = False
        if dist_to_goal < 0.1 and avg_velocity < 0.1:
            if prev_center is not None and np.linalg.norm(center - prev_center) < 0.1:
                new_stable_count += 1
            else: new_stable_count = 0
            if new_stable_count >= 10:
                goal_reached = True
        else: new_stable_count = 0
        return goal_reached, center.copy(), new_stable_count

    def run(self):
        prev_center = None; stable_count = 0; iter_count = 0
        while iter_count < self.config.ITER_MAX:
            iter_count += 1
            # Hapus print iterasi dari sini agar tidak terlalu banyak output jika tidak sinkron dengan plot
            # if iter_count % 100 == 0: print(f"Iterasi {iter_count}") 
            for robot in self.robots:
                robot.compute_control(self.robots, self.environment, self.config.TIMESTEP)
            if self.check_collision():
                print("Tabrakan terdeteksi. Simulasi berhenti."); break
            goal_reached, prev_center, stable_count = self.check_reach_goal(prev_center, stable_count)
            if goal_reached: break
        # Print status akhir simulasi bisa tetap di sini
        print(f"Simulasi selesai setelah {iter_count} iterasi.")
        return self.robots 

class Animator:
    def __init__(self, config, direct_data):
        self.config = config
        self.data = direct_data 
        if self.data is None: print("Peringatan: Animator menerima data kosong.")

    def _plot_prism(self, ax, polygon_2d, z_min=0, z_max=3, color='gray', alpha=0.3, edge_color='k'):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
        xs = [p[0] for p in polygon_2d]; ys = [p[1] for p in polygon_2d]
        verts_bottom = list(zip(xs, ys, [z_min]*len(xs)))
        verts_top = list(zip(xs, ys, [z_max]*len(xs)))
        ax.add_collection3d(Poly3DCollection([verts_bottom], facecolors=color, alpha=alpha))
        ax.add_collection3d(Poly3DCollection([verts_top], facecolors=color, alpha=alpha))
        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            side = [verts_bottom[i], verts_bottom[j], verts_top[j], verts_top[i]]
            ax.add_collection3d(Poly3DCollection([side], facecolors=color, alpha=alpha))
            ax.plot([verts_bottom[i][0], verts_bottom[j][0]], [verts_bottom[i][1], verts_bottom[j][1]], [verts_bottom[i][2], verts_bottom[j][2]], color=edge_color)
            ax.plot([verts_top[i][0], verts_top[j][0]], [verts_top[i][1], verts_top[j][1]], [verts_top[i][2], verts_top[j][2]], color=edge_color)
            ax.plot([verts_bottom[i][0], verts_top[i][0]], [verts_bottom[i][1], verts_top[i][1]], [verts_bottom[i][2], verts_top[i][2]], color=edge_color)

    def animate(self, export_gif=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d 

        if self.data is None or not self.data:
            print("Tidak ada data untuk dianimasikan."); return
        if 'path' not in self.data[0] or self.data[0]['path'].shape[0] == 0:
            print("Data path tidak valid atau kosong."); return
        
        length = self.data[0]['path'].shape[0]
        TRAIL_LENGTH = self.config.ANIMATOR_TRAIL_LENGTH 

        fig = plt.figure(figsize=(10,7))
        ax = plt.axes(projection='3d')
        center_trajectory = [] 
        color_list = ['blue', 'orange', 'green', 'red', 'purple', 'cyan']
        image_array = []
        
        if export_gif:
            try: import cv2; import imageio
            except ImportError: print("Peringatan: cv2/imageio tidak terinstal. GIF tidak diekspor."); export_gif = False

        for iter_idx in range(0, length, 2): 
            ax.cla()
            # Cetak informasi frame animasi
            # iter_idx adalah indeks data simulasi, iter_idx//2 bisa dianggap nomor frame animasi
            # print(f"Animator - Frame Animasi: {iter_idx//2}, Data Simulasi Indeks: {iter_idx}")


            for i, obstacle_geo in enumerate(self.config.OBSTACLE_GEOMETRIES): 
                z_min, z_max = self.config.OBSTACLE_HEIGHTS[i] 
                self._plot_prism(ax, obstacle_geo, z_min=z_min, z_max=z_max)

            current_center_pos = np.zeros(3)
            for i in range(self.config.NUM_ROBOT): 
                color = color_list[i % len(color_list)]
                full_path_data_robot_i = self.data[i]['path']
                
                trail_segment_start_index = max(0, iter_idx - TRAIL_LENGTH + 1)
                trail_to_plot = full_path_data_robot_i[trail_segment_start_index : iter_idx+1, :]
                
                plot_label_robot = None
                if iter_idx == 0 and trail_segment_start_index == 0 : 
                    plot_label_robot = f"Robot {i}"

                if trail_to_plot.shape[0] > 1:
                    ax.plot(trail_to_plot[:,1], trail_to_plot[:,2], trail_to_plot[:,3], color=color, label=plot_label_robot)
                elif trail_to_plot.shape[0] == 1: 
                     ax.plot(trail_to_plot[:,1], trail_to_plot[:,2], trail_to_plot[:,3], color=color, marker='o', markersize=2, label=plot_label_robot)
                
                pose = full_path_data_robot_i[iter_idx,:]
                ax.scatter(pose[1], pose[2], pose[3], s=50, c=color) 
                ax.quiver(pose[1], pose[2], pose[3], pose[4]*0.3, pose[5]*0.3, pose[6]*0.3, length=0.5, normalize=True, color='k')
                current_center_pos += pose[1:4]
            
            current_center_pos /= self.config.NUM_ROBOT
            # Cetak posisi pusat yang dihitung dan akan diplot oleh Animator
            # print(f"Animator - Posisi Pusat Diplot: {current_center_pos}")

            center_trajectory.append(current_center_pos) 
            center_arr = np.array(center_trajectory) 

            num_center_points_so_far = center_arr.shape[0]
            center_trail_start_index = max(0, num_center_points_so_far - TRAIL_LENGTH) 
            center_trail_to_plot = center_arr[center_trail_start_index : num_center_points_so_far, :]
            
            plot_label_center = None
            plot_label_center_path = None
            if iter_idx == 0: 
                plot_label_center = 'Center'
                if center_trail_start_index == 0:
                     plot_label_center_path = 'Center Path'

            ax.scatter(current_center_pos[0], current_center_pos[1], current_center_pos[2], c='r', s=50, label=plot_label_center)
            if center_trail_to_plot.shape[0] > 1:
                 ax.plot(center_trail_to_plot[:, 0], center_trail_to_plot[:, 1], center_trail_to_plot[:, 2], linestyle='--', color='cyan', label=plot_label_center_path)
            elif center_trail_to_plot.shape[0] == 1 and plot_label_center_path: 
                 ax.plot(center_trail_to_plot[:, 0], center_trail_to_plot[:, 1], center_trail_to_plot[:, 2], linestyle='--', color='cyan', marker='o', markersize=2, label=plot_label_center_path)

            ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]'); ax.grid(True)
            if iter_idx == 0 : ax.legend()

            ax.set_xlim(current_center_pos[0]-7, current_center_pos[0]+7)
            ax.set_ylim(current_center_pos[1]-5, current_center_pos[1]+5)
            ax.set_zlim(0, 5)  
            plt.tight_layout(); plt.pause(0.001)

            if export_gif:
                import os 
                if not os.path.exists("results"): os.makedirs("results")
                temp_file_name = "results/_temp_frame.png"
                plt.savefig(temp_file_name)
                img = cv2.imread(temp_file_name)
                if img is not None: image_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else: print(f"Peringatan: Gagal memuat frame {temp_file_name}")
        
        if export_gif and image_array:
            try: imageio.mimsave(self.config.GIF_NAME, image_array, fps=10); print(f"GIF disimpan ke {self.config.GIF_NAME}")
            except Exception as e: print(f"Error menyimpan GIF: {e}")
        plt.show()

def format_robots_data_for_animation(robots_list):
    return [{'path': np.array(robot.path)} for robot in robots_list]

if __name__ == "__main__":
    config = Config()
    environment = Environment(config)
    simulation = Simulation(config, environment)
    raw_robots_list_from_sim = simulation.run() 
    animation_ready_data = format_robots_data_for_animation(raw_robots_list_from_sim)
    animator = Animator(config, direct_data=animation_ready_data) 
    if animator.data: 
        animator.animate(export_gif=False)
    else: 
        print("Tidak dapat menganimasikan karena tidak ada data atau data tidak valid.")

