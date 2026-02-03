import math
import numpy as np

from enum import Enum
from Agent.MultiAgentUtils import nearest_point_to_obstacle

class Mode(Enum):
    TAKEOFF = -1
    FORMATION = 0
    TAILGATING = 1

class MultiAgentMethod():
    def __init__(self, index, position, config, num_robot, initial_positions, obstacles_2d, obstacle_heights, velocity=np.zeros(3)):
        self.index = index
        self.config = config 
        self.num_robot = num_robot
        self.initial_positions = initial_positions
        self.obstacles_2d = obstacles_2d
        self.obstacle_heights = obstacle_heights
        self.stamp = 0.0
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.control = np.zeros(3, dtype=float)
        self.mode = Mode.TAKEOFF
        self.current_goal_index = 0
        self.scaling_factor = 1.0 
        self.path = [np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value]])]

        # Properti untuk menyimpan "peta" lintasan dan jarak pandang ke depan
        self.desired_path = None
        self.look_ahead_distance = self.config.LOOK_AHEAD_DISTANCE # Asumsi ada di config, misal: 2.0

        # Jika tipe path adalah circular, buat lintasannya sekarang juga
        if self.config.PATH_TYPE == 'circular':
            self.desired_path = self._generate_circular_path(
                center=self.config.CIRCLE_CENTER,
                radius=self.config.CIRCLE_RADIUS,
                num_points=500  # Jumlah titik untuk merepresentasikan lingkaran
            )

    def _generate_circular_path(self, center, radius, num_points=500):
        """
        Menghasilkan sebuah list berisi titik-titik (waypoints) yang membentuk lintasan lingkaran.
        Fungsi ini dipanggil sekali saja saat inisialisasi.
        """
        path_points = []
        for i in range(num_points + 1):
            theta = 2 * np.pi * (i / num_points)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            z = center[2]
            path_points.append(np.array([x, y, z]))
        return np.array(path_points)

    def _find_look_ahead_point(self, current_position, path):
        """
        Mencari titik target di lintasan yang berada di depan robot (Pure Pursuit sederhana).
        """
        # 1. Cari indeks titik terdekat di path dari posisi sekarang
        distances = np.linalg.norm(path - current_position, axis=1)
        nearest_index = np.argmin(distances)
        
        # 2. Cari titik target (look-ahead point) di depan titik terdekat
        target_index = nearest_index
        while target_index < len(path) - 1:
            dist_to_target = np.linalg.norm(path[target_index] - current_position)
            if dist_to_target > self.look_ahead_distance:
                break
            target_index += 1
            
        return path[target_index]

    def update_state(self, control, dt):
        control_norm = np.linalg.norm(control)
        if control_norm > self.config.UMAX: 
            control = control/control_norm*self.config.UMAX
        self.velocity = self.velocity + control*dt
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > self.config.VMAX: 
            self.velocity = self.velocity/velocity_norm*self.config.VMAX
        self.position = self.position + self.velocity*dt
        self.stamp += dt
        self.control = control
        self.path.append(np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value]]))

    def compute_control(self, robots, dt=0.005):
        # --- Bagian Transisi State ---
        # Logika ini memeriksa apakah sudah waktunya beralih dari TAKEOFF ke FORMATION
        if self.mode == Mode.TAKEOFF:
            all_robots_ready = True
            for r in robots:
                # Periksa apakah setiap robot sudah mencapai ketinggian target
                if abs(r.position[2] - self.config.TARGET_ALTITUDE) > self.config.ALTITUDE_TOLERANCE:
                    all_robots_ready = False
                    break
            
            # Jika semua robot sudah siap, ubah mode mereka ke FORMATION
            if all_robots_ready:
                # Penting: ubah mode untuk SEMUA robot, bukan hanya 'self'
                for r in robots:
                    if r.mode == Mode.TAKEOFF: # Pastikan hanya mengubah yang masih takeoff
                        r.mode = Mode.FORMATION
                # Karena mode self mungkin sudah diubah oleh loop di atas, kita pastikan lagi
                self.mode = Mode.FORMATION

        # --- Bagian Kalkulasi Perilaku Berdasarkan Mode ---
        if self.mode == Mode.TAKEOFF:
            # Saat takeoff, hanya peduli untuk naik dan menghindari tabrakan
            v_takeoff = self.behavior_takeoff()
            v_col = self.behavior_collision(robots) # Tetap aktif untuk keamanan
            desired_velocity = v_takeoff + v_col
        
        elif self.mode == Mode.FORMATION:
            self.mode_changing() # mode_changing Anda mungkin perlu disesuaikan agar tidak kembali ke formation jika tidak ada obstacle
            v_mig = self.behavior_migration(robots)
            v_form = self.behavior_formation(robots)
            v_obs = self.behavior_obstacle()
            v_col = self.behavior_collision(robots)
            desired_velocity = v_mig + v_form + v_obs + v_col
            # Jika normanya mendekati nol, set ke nol
            if np.linalg.norm(desired_velocity) < 1e-3:
                desired_velocity = np.zeros_like(desired_velocity)
        
        else: # Mode.TAILGATING
            self.mode_changing()
            v_mig = self.behavior_migration(robots)
            v_tail = self.behavior_tailgating(robots)
            v_obs = self.behavior_obstacle()
            v_col = self.behavior_collision(robots)
            desired_velocity = v_mig + v_tail + v_obs + v_col

        desired_control = (desired_velocity - self.velocity) / dt
        self.update_state(desired_control, dt)

    def _get_dynamic_target(self, robots):
        if self.config.PATH_TYPE == 'goal':
            p_target = np.array([self.config.XGOAL, self.config.YGOAL, self.config.ZGOAL])
            center = np.mean([robot.position for robot in robots], axis=0)
            direction = p_target - center
            psi_form = math.atan2(direction[1], direction[0])
            return p_target, psi_form
        
        elif self.config.PATH_TYPE == 'multi-goal':
            waypoints = self.config.PATH_WAYPOINTS
            center = np.mean([robot.position for robot in robots], axis=0)
            
            # --- Penanganan Indeks Waypoint ---
            # Pastikan indeks tidak keluar dari rentang
            if self.current_goal_index >= len(waypoints):
                p_target = waypoints[-1]
                # Logika berhenti di goal terakhir bisa ditambahkan di sini
                return p_target, 0 # Arah hadap tidak relevan jika sudah berhenti
            
            # Tentukan titik-titik kunci untuk segmen saat ini
            current_target_idx = self.current_goal_index
            prev_target_idx = max(0, current_target_idx - 1)
            next_target_idx = min(len(waypoints) - 1, current_target_idx + 1)

            p_current = waypoints[current_target_idx]
            p_prev = waypoints[prev_target_idx]
            p_next = waypoints[next_target_idx]

            # Jarak kawanan ke target saat ini
            dist_to_current = np.linalg.norm(center - p_current)
            
            # --- Logika Transisi atau Gerak Lurus ---
            
            # Jika masih jauh dari target, bergerak lurus
            if dist_to_current > self.config.TRANSITION_RADIUS:
                p_target = p_current
                # Orientasi lurus menghadap target saat ini
                direction_vec = p_target - center
                psi_form = np.arctan2(direction_vec[1], direction_vec[0])
            
            # Jika sudah masuk zona transisi, mulai berbelok mulus
            else:
                p_target = p_next # Mulai mengejar target berikutnya
                
                # Hitung orientasi segmen lama dan baru
                vec_old = p_current - p_prev
                vec_new = p_next - p_current
                
                psi_old = np.arctan2(vec_old[1], vec_old[0])
                psi_new = np.arctan2(vec_new[1], vec_new[0])
                
                # Atasi masalah "wraparound" pada sudut (misal dari 350 ke 10 derajat)
                if abs(psi_new - psi_old) > np.pi:
                    if psi_new > psi_old:
                        psi_old += 2 * np.pi
                    else:
                        psi_new += 2 * np.pi

                # Hitung faktor interpolasi (s), dari 0 ke 1 saat mendekati target
                s = (self.config.TRANSITION_RADIUS - dist_to_current) / self.config.TRANSITION_RADIUS
                
                # Interpolasi sudut secara linear (Lerp)
                psi_form = (1 - s) * psi_old + s * psi_new
            
            # Pindah ke goal berikutnya jika sudah sangat dekat
            if dist_to_current < self.config.GOAL_REACHED_THRESHOLD:
                self.current_goal_index += 1

            return p_target, psi_form
        
        elif self.config.PATH_TYPE == 'circular':
            # Dapatkan posisi pusat kawanan saat ini
            center_of_swarm = np.mean([robot.position for robot in robots], axis=0)

            # 1. Cari titik target di "peta" lintasan yang sudah dibuat
            #    Metode ini tidak lagi menggunakan 'self.stamp'
            p_target = self._find_look_ahead_point(center_of_swarm, self.desired_path)

            # 2. Hitung orientasi formasi berdasarkan arah ke p_target
            #    Ini membuat formasi menghadap ke arah mana ia bergerak di sepanjang path.
            direction_vec = p_target - center_of_swarm
            # Hindari error jika sudah di target
            if np.linalg.norm(direction_vec) > 1e-6:
                psi_form = np.arctan2(direction_vec[1], direction_vec[0])
            else:
                # Jika sudah sangat dekat, pertahankan orientasi terakhir (atau set ke 0)
                # Di sini kita perlu cara untuk mengambil orientasi sebelumnya jika perlu,
                # atau cukup gunakan orientasi dari segmen path berikutnya.
                # Untuk simple, kita bisa gunakan orientasi dari target look-ahead itu sendiri.
                psi_form = np.arctan2(p_target[1], p_target[0]) # Simplifikasi

            return p_target, psi_form
        
    # def behavior_migration(self, robots):
    #     center = np.mean([robot.position for robot in robots], axis=0)
    #     goal = np.array([self.config.XGOAL, self.config.YGOAL, self.config.ZGOAL])
    #     error_vec = goal - center
    #     error_dist = np.linalg.norm(error_vec)
    #     if error_dist > 0:
    #         direction = error_vec / error_dist
    #     else:
    #         direction = np.zeros(3)
    #     speed = self.config.VREF
    #     if error_dist < 1.0:
    #         speed *= error_dist  # slow down near goal
    #     if speed < 0.05:
    #         speed = 0
    #     return speed * direction

    def behavior_migration(self, robots):
        p_target, _ = self._get_dynamic_target(robots)
        center = np.mean([robot.position for robot in robots], axis=0)
        error_vec = p_target - center
        error_dist = np.linalg.norm(error_vec)
        if error_dist > 0:
            direction = error_vec / error_dist
        else:
            direction = np.zeros(3)
        speed = self.config.VREF
        if self.config.PATH_TYPE == 'goal':
            if error_dist < 1.0 and error_dist > 0: 
                speed *= (error_dist / 1.0)
            if speed < 0.05: 
                speed = 0
        else:
            pass
        return speed * direction
    
    # def behavior_formation(self, robots):
    #     v_form = np.zeros(3) 
    #     for i in range(self.num_robot):
    #         desired_rel = self.config.TOPOLOGY[i,:] - self.config.TOPOLOGY[self.index,:]
    #         actual_rel = robots[i].position - self.position
    #         v_form += actual_rel - desired_rel
    #     return self.config.W_form * v_form

    def behavior_formation(self, robots):
        v_form = np.zeros(3)
        _, psi_form = self._get_dynamic_target(robots)
        # Buat matriks rotasi 2D
        c, s = np.cos(psi_form), np.sin(psi_form)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        # Ambil topologi awal dan rotasi hanya komponen x dan y
        initial_topology_xy = self.config.TOPOLOGY[:, :2]
        rotated_topology_xy = np.dot(initial_topology_xy, rotation_matrix.T)

        for i in range(self.num_robot):
            # Gabungkan kembali dengan komponen z
            desired_pos_i = np.append(rotated_topology_xy[i], self.config.TOPOLOGY[i, 2])
            desired_pos_self = np.append(rotated_topology_xy[self.index], self.config.TOPOLOGY[self.index, 2])
            
            # Hitung vektor posisi relatif yang diinginkan (sudah dirotasi)
            desired_rel = desired_pos_i - desired_pos_self
            
            actual_rel = robots[i].position - self.position
            v_form += actual_rel - desired_rel
        ### MODIFIKASI SELESAI ###
            
        return self.config.W_form * v_form
    
    def behavior_tailgating(self, robots):
        leader_idx = self.select_leader(robots)
        if leader_idx == -1:
            return np.zeros(3)
        leader_velocity = robots[leader_idx].velocity
        norm_leader_velocity = np.linalg.norm(leader_velocity)
        if norm_leader_velocity == 0:
            u_ref = np.zeros(3)
        else:
            u_ref = leader_velocity / norm_leader_velocity
        v_tail = (robots[leader_idx].position - self.position - self.config.DREF * u_ref) + robots[leader_idx].velocity
        return self.config.W_tail * v_tail
    
    def behavior_obstacle(self):
        v_obs = np.zeros(3) 
        for j in range(len(self.obstacles_2d)):
            obstacle = self.obstacles_2d[j]
            obs_point = nearest_point_to_obstacle(self.position[:2], obstacle)
            obs_rel = self.position - np.concatenate([obs_point,[self.position[2]]])
            obs_dis = np.linalg.norm(obs_rel)
            if obs_dis < self.config.ALERT_RADIUS:
                v_obs += 0.5*(1/obs_dis - 1/self.config.ALERT_RADIUS)/(obs_dis**2)*obs_rel/obs_dis
        return self.config.W_obs*v_obs
    
    def behavior_collision(self, robots):
        v_col = np.zeros(3) 
        for i in range(self.num_robot):
            if i == self.index:
                continue
            pos_rel = self.position - robots[i].position
            pos_dis = np.linalg.norm(pos_rel)
            if pos_dis < self.config.ALERT_RADIUS and pos_dis > 1e-6: 
                v_col_component = 2*(1/pos_dis - 1/self.config.ALERT_RADIUS)/(pos_dis**2)*pos_rel/pos_dis
                if not np.any(np.isnan(v_col_component)) and not np.any(np.isinf(v_col_component)):
                    v_col += v_col_component
        return self.config.W_col*v_col

    def select_leader(self, robots):
        positions = np.array([r.position for r in robots])
        center = np.mean(positions, axis=0)
        goal = np.array([self.config.XGOAL, self.config.YGOAL, self.config.ZGOAL])
        error_vec = goal - center
        error_dist = np.linalg.norm(error_vec)
        if error_dist > 0:
            direction = error_vec / error_dist
        else:
            direction = np.zeros(3)

        vec = np.dot(positions - self.position, direction)
        vec[vec <= 0] = np.inf
        if np.all(np.isinf(vec)):
            return -1
        return np.argmin(vec)
    
    def mode_changing(self):
        if len(self.obstacles_2d) == 0:
            self.mode = Mode.FORMATION
            self.scaling_factor = 1.0
            return

        we = self.estimate_environment_width()
        if we is None:
            self.mode = Mode.FORMATION
            # print("No obstacles detected, switching to Formation mode.")
            self.scaling_factor = 1.0
            return

        if we <= self.config.ALPHA * self.config.ROBOT_RADIUS:
            self.mode = Mode.TAILGATING
            # print("Switching to Tailgating mode due to narrow environment width.")
            self.scaling_factor = -1
        else:
            wf = self.estimate_formation_width()
            scaling_factor = 1.0
            if we - 2 * self.config.ROBOT_RADIUS < wf:
                scaling_factor = (we - 2 * self.config.ROBOT_RADIUS) / wf
            self.mode = Mode.FORMATION
            self.scaling_factor = scaling_factor

    def estimate_formation_width(self):
        y_left = np.min(self.config.TOPOLOGY[:,1])
        y_right = np.max(self.config.TOPOLOGY[:,1])
        return y_right - y_left
    
    def estimate_environment_width(self):
        obs_left = None; obs_right = None
        for j in range(len(self.obstacles_2d)):
            obstacle = self.obstacles_2d[j]
            obs_point = nearest_point_to_obstacle(self.position[:2], obstacle)
            # Check the nearest obstacle point is in the front of robot
            if np.dot((self.position[:2] - obs_point), self.config.UREF[:2]) <= 0:
                # Check the nearest obstacle point is in the left or right side of robot
                if np.dot((self.position[:2] - obs_point), np.array([self.config.UREF[1], self.config.UREF[0]])) < 0:
                    if obs_left is None or np.linalg.norm(self.position[:2] - obs_point) < np.linalg.norm(self.position[:2] - obs_left):
                        obs_left  = obs_point
                else:
                    if obs_right is None or np.linalg.norm(self.position[:2] - obs_point) < np.linalg.norm(self.position[:2] - obs_right):
                        obs_right = obs_point
        if obs_left is None or obs_right is None:
            return None
        theta = math.atan2(self.config.UREF[1], self.config.UREF[0])
        width = abs((obs_left[0]-obs_right[0])*np.sin(theta) + (obs_left[1]-obs_right[1])*np.cos(theta))
        return width

    def behavior_random(self):
        return self.config.W_rand*np.concatenate([np.random.rand(2),[0]])
    
    def behavior_takeoff(self):
        """
        Menghasilkan vektor kecepatan untuk naik vertikal ke target ketinggian.
        """
        # Ambil parameter dari config
        target_altitude = self.config.TARGET_ALTITUDE  # misal: 5.0 meter
        kp_takeoff = self.config.KP_TAKEOFF            # misal: 1.5 (gain proporsional)
        vmax_z_takeoff = self.config.VMAX_Z_TAKEOFF    # misal: 1.0 m/s (kecepatan naik maks)

        # Hitung error ketinggian
        error_z = target_altitude - self.position[2]
        
        # Hitung kecepatan z yang diinginkan dengan P-controller
        desired_vz = kp_takeoff * error_z
        
        # Batasi kecepatan naik maksimum
        desired_vz = np.clip(desired_vz, -vmax_z_takeoff, vmax_z_takeoff)

        # Buat vektor kecepatan, hanya bergerak di sumbu Z
        v_takeoff = np.array([0.0, 0.0, desired_vz])
        
        return v_takeoff