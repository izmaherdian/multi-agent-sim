import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Agent.Quadcopter import Quadcopter
from Agent.QuadMultiAgent import QuadMultiAgent
from Agent.MultiAgentConfig import MultiAgentConfig
from Environment.Canvas import Canvas

from mpl_toolkits.mplot3d import Axes3D

deg2rad = np.pi / 180.0

def wayPoints1():
    
    v_average = 1.6

    t_ini = 0
    t = np.array([3, 4, 7, 10])
    
    wp_ini = np.array([0, 0, -2])
    wp = np.array([[2, 2, -1],
                   [-2, 3, -3],
                   [-2, -1, -3],
                   [3, -2, -1]])

    yaw_ini = 90    
    yaw = np.array([-90, 90, 270, 90])

    t = np.hstack((t_ini, t)).astype(float)
    wp = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.hstack((yaw_ini, yaw)).astype(float)*deg2rad

    return t, wp, yaw, v_average

def wayPoints2():
    v_average = 1.6

    t_ini = 0
    t = np.array([3, 4, 7, 10])
    
    wp_ini = np.array([0, 0, -5])
    wp = np.array([[1, 1, -1],
                   [2, 3, -3],
                   [2, -1, -3],
                   [-3, 2, -1]])

    yaw_ini = 90    
    yaw = np.array([-90, 90, 270, 90])

    t = np.hstack((t_ini, t)).astype(float)
    wp = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.hstack((yaw_ini, yaw)).astype(float)*deg2rad

    return t, wp, yaw, v_average

# Menjalankan simulasi dengan dua quadcopters
def main():
    plt.ion()
    canvas = Canvas(obstacle_scheme='NONE', wind_type='NONE', controller_type='NONE', formation_type='NONE', orient='NED')

    history_quad = [
        {
            't': [],
            'x': [], 'y': [], 'z': [],
            'q0': [], 'q1': [], 'q2': [], 'q3': [],
            'phi': [], 'theta': [], 'psi': [],
            'v_x': [], 'v_y': [], 'v_z': [],
            'p': [], 'q': [], 'r': [],
            'wM1': [], 'wM2': [], 'wM3': [], 'wM4': [],

            'x_sp': [], 'y_sp': [], 'z_sp': [],
            'q0_des': [], 'q1_des': [], 'q2_des': [], 'q3_des': [],
            'phi_sp': [], 'theta_sp': [], 'psi_sp': [],
            'v_x_sp': [], 'v_y_sp': [], 'v_z_sp': [],
            'p_des': [], 'q_des': [], 'r_des': [],
            'wM1_sp': [], 'wM2_sp': [], 'wM3_sp': [], 'wM4_sp': [],

            'yaw_rate_sp': [],
            'x_thr_sp': [], 'y_thr_sp': [], 'z_thr_sp': []
        }
        # for _ in range(multi_agent.num_robot)
    ]

    # Buat dua quadcopters dengan posisi awal berbeda dan warna berbeda
    quad1 = Quadcopter(
        id=1,
        ax=canvas.ax,
        init_pos=[0.0, 0.0, 0.0],
        front_color='orange',
        ctrlType="xyz_pos",
        trajSelect=(0, 3, 0),
        wayPoints=wayPoints1()
    )
    # quad2 = Quadcopter(
    #     id=2,
    #     ax=canvas.ax,
    #     init_pos=[-5.0, 0.0, 0.0],
    #     front_color='blue',
    #     ctrlType="xyz_pos",
    #     trajSelect=(0, 3, 0),
    #     wayPoints=wayPoints2()
    # )

    quad1.initialize_draw()
    # quad2.initialize_draw()

    dt = 0.005
    fps = 30
    frame_time = 1.0 / fps

    save_video = True
    video_filename = "Simulation_single_quad.mp4"
    video_fps = int(1 / frame_time)

    if save_video:
        metadata = dict(title='Single quad simulation', artist='Matplotlib', comment='Saved simulation')
        writer = animation.FFMpegWriter(fps=video_fps, metadata=metadata)
        writer.setup(canvas.fig, video_filename, dpi=100)

    running = True

    def on_close(event):
        nonlocal running
        running = False
        print("Window closed, stopping...")

    canvas.fig.canvas.mpl_connect('close_event', on_close)

    t_prev = time.time()
    t_logic = t_prev
    t_render = t_prev
    t_sim = 0.0

    wind = canvas.wind

    while running:
        t_now = time.time()

        # Logika simulasi tiap dt detik
        if (t_now - t_logic) >= dt:
            quad1.update(t_sim, dt, wind)
            # quad2.update(t_sim, dt, wind)
            t_logic = t_now
            t_sim += dt

            # Simpan data untuk plotting
            history_quad[0]['t'].append(t_sim)
            history_quad[0]['x'].append(quad1.x)
            history_quad[0]['y'].append(quad1.y)
            history_quad[0]['z'].append(quad1.z)
            history_quad[0]['q0'].append(quad1.q0)
            history_quad[0]['q1'].append(quad1.q1)
            history_quad[0]['q2'].append(quad1.q2)
            history_quad[0]['q3'].append(quad1.q3)
            history_quad[0]['phi'].append(quad1.phi)
            history_quad[0]['theta'].append(quad1.theta)
            history_quad[0]['psi'].append(quad1.psi)
            history_quad[0]['v_x'].append(quad1.vx)
            history_quad[0]['v_y'].append(quad1.vy)
            history_quad[0]['v_z'].append(quad1.vz)
            history_quad[0]['p'].append(quad1.p)
            history_quad[0]['q'].append(quad1.q)
            history_quad[0]['r'].append(quad1.r)
            history_quad[0]['wM1'].append(quad1.wM1)
            history_quad[0]['wM2'].append(quad1.wM2)
            history_quad[0]['wM3'].append(quad1.wM3)
            history_quad[0]['wM4'].append(quad1.wM4)
            history_quad[0]['x_sp'].append(quad1.xDes)
            history_quad[0]['y_sp'].append(quad1.yDes)
            history_quad[0]['z_sp'].append(quad1.zDes)
            history_quad[0]['q0_des'].append(quad1.q0Des)
            history_quad[0]['q1_des'].append(quad1.q1Des)
            history_quad[0]['q2_des'].append(quad1.q2Des)
            history_quad[0]['q3_des'].append(quad1.q3Des)
            history_quad[0]['phi_sp'].append(quad1.phiDes)
            history_quad[0]['theta_sp'].append(quad1.thetaDes)
            history_quad[0]['psi_sp'].append(quad1.psiDes)
            history_quad[0]['v_x_sp'].append(quad1.vxDes)
            history_quad[0]['v_y_sp'].append(quad1.vyDes)
            history_quad[0]['v_z_sp'].append(quad1.vzDes)
            history_quad[0]['p_des'].append(quad1.pDes)
            history_quad[0]['q_des'].append(quad1.qDes)
            history_quad[0]['r_des'].append(quad1.rDes)
            history_quad[0]['x_thr_sp'].append(quad1.xthrDes)
            history_quad[0]['y_thr_sp'].append(quad1.ythrDes)
            history_quad[0]['z_thr_sp'].append(quad1.zthrDes)
            history_quad[0]['yaw_rate_sp'].append(quad1.yawRateDes)
            history_quad[0]['wM1_sp'].append(quad1.wM1_cmd)
            history_quad[0]['wM2_sp'].append(quad1.wM2_cmd)
            history_quad[0]['wM3_sp'].append(quad1.wM3_cmd)
            history_quad[0]['wM4_sp'].append(quad1.wM4_cmd)

        # Render frame tiap frame_time detik
        if (t_now - t_render) >= frame_time:
            quad1.drawer.update_draw(quad1.position, quad1.quat, label='Q1')
            # quad2.drawer.update_draw(quad2.position, quad2.quat, label='Q2')

            canvas.draw(t=t_sim, quadcopters=[quad1]) # , quad2])

            if save_video:
                writer.grab_frame()
            
            plt.pause(0.001)
            t_render = t_now

    plt.ioff()
    plt.show()

    if save_video:
        writer.finish()
        print(f"Video saved to {video_filename}")

    # Simpan data ke file
    data_quad_single = []
    h = history_quad[0]
    T = len(h['t'])
    arr_quad = np.zeros((T, 45))
    arr_quad[:, 0] = h['t']
    arr_quad[:, 1] = h['x']
    arr_quad[:, 2] = h['y']
    arr_quad[:, 3] = h['z']
    arr_quad[:, 4] = h['q0']
    arr_quad[:, 5] = h['q1']
    arr_quad[:, 6] = h['q2']
    arr_quad[:, 7] = h['q3']
    arr_quad[:, 8] = h['phi']
    arr_quad[:, 9] = h['theta']
    arr_quad[:, 10] = h['psi']
    arr_quad[:, 11] = h['v_x']
    arr_quad[:, 12] = h['v_y']
    arr_quad[:, 13] = h['v_z']
    arr_quad[:, 14] = h['p']
    arr_quad[:, 15] = h['q']
    arr_quad[:, 16] = h['r']
    arr_quad[:, 17] = h['wM1']
    arr_quad[:, 18] = h['wM2']
    arr_quad[:, 19] = h['wM3']
    arr_quad[:, 20] = h['wM4']
    arr_quad[:, 21] = h['x_sp']
    arr_quad[:, 22] = h['y_sp']
    arr_quad[:, 23] = h['z_sp']
    arr_quad[:, 24] = h['q0_des']
    arr_quad[:, 25] = h['q1_des']
    arr_quad[:, 26] = h['q2_des']
    arr_quad[:, 27] = h['q3_des']
    arr_quad[:, 28] = h['phi_sp']
    arr_quad[:, 29] = h['theta_sp']
    arr_quad[:, 30] = h['psi_sp']
    arr_quad[:, 31] = h['v_x_sp']
    arr_quad[:, 32] = h['v_y_sp']
    arr_quad[:, 33] = h['v_z_sp']
    arr_quad[:, 34] = h['p_des']
    arr_quad[:, 35] = h['q_des']
    arr_quad[:, 36] = h['r_des']
    arr_quad[:, 37] = h['yaw_rate_sp']
    arr_quad[:, 38] = h['x_thr_sp']
    arr_quad[:, 39] = h['y_thr_sp']
    arr_quad[:, 40] = h['z_thr_sp']
    arr_quad[:, 41] = h['wM1_sp']
    arr_quad[:, 42] = h['wM2_sp']
    arr_quad[:, 43] = h['wM3_sp']
    arr_quad[:, 44] = h['wM4_sp']
    data_quad_single.append(
        {
            'path': arr_quad
        }
    )

    file_name_quad = "single_quad_data.pkl"

    import pickle
    with open(file_name_quad, 'wb') as f:
        pickle.dump({
            'data_quad_single': data_quad_single
        }, f)
    print(f"Data saved to {file_name_quad}")


def main_multi_agent():
    plt.ion()
    multi_agent_config = MultiAgentConfig()
    obstacle_scheme = multi_agent_config.OBSTACLE_SCHEME
    wind_type       = multi_agent_config.WIND_TYPE
    controller_type = multi_agent_config.CONTROLLER
    formation_type  = multi_agent_config.FORMATION_TYPE
    orient          = multi_agent_config.ORIENT

    canvas = Canvas(obstacle_scheme=obstacle_scheme, wind_type=wind_type, controller_type=controller_type, formation_type=formation_type, orient=orient)

    agents_config = multi_agent_config.agents_config

    multi_agent = QuadMultiAgent(canvas.ax, agents_config, obstacle_scheme=obstacle_scheme)
    multi_agent.initialize_draw()

    history_agent = [
        { 
            't': [], 
            'xa': [], 'ya': [], 'za': [], 
            'v_xa': [], 'v_ya': [], 'v_za': [], 
            'mode': [], 'scale': [] 
        }
        for _ in range(multi_agent.num_robot)
    ]

    history_quad = [
        {
            't': [],
            'x': [], 'y': [], 'z': [],
            'q0': [], 'q1': [], 'q2': [], 'q3': [],
            'phi': [], 'theta': [], 'psi': [],
            'v_x': [], 'v_y': [], 'v_z': [],
            'p': [], 'q': [], 'r': [],
            'wM1': [], 'wM2': [], 'wM3': [], 'wM4': [],

            'x_sp': [], 'y_sp': [], 'z_sp': [],
            'q0_des': [], 'q1_des': [], 'q2_des': [], 'q3_des': [],
            'phi_sp': [], 'theta_sp': [], 'psi_sp': [],
            'v_x_sp': [], 'v_y_sp': [], 'v_z_sp': [],
            'p_des': [], 'q_des': [], 'r_des': [],
            'wM1_sp': [], 'wM2_sp': [], 'wM3_sp': [], 'wM4_sp': [],

            'yaw_rate_sp': [],
            'x_thr_sp': [], 'y_thr_sp': [], 'z_thr_sp': []
        }
        for _ in range(multi_agent.num_robot)
    ]

    dt = multi_agent_config.dt
    frame_time = multi_agent_config.frame_time

    running = True

    def on_close(event):
        nonlocal running
        running = False
        print("Window closed, stopping...")

    canvas.fig.canvas.mpl_connect('close_event', on_close)

    save_video = multi_agent_config.SAVE_VIDEO  
    video_filename = f"Simulation_{obstacle_scheme}_{wind_type}_{controller_type}_formation{formation_type}_{orient}_erc_scheme1.mp4"
    video_fps = int(1 / frame_time)

    if save_video:
        metadata = dict(title='Multi-agent Simulation', artist='Matplotlib', comment='Saved simulation')
        writer = animation.FFMpegWriter(fps=video_fps, metadata=metadata)
        writer.setup(canvas.fig, video_filename, dpi=100)

    t_prev = time.time()
    t_logic = t_prev
    t_render = t_prev
    t_sim = 0.0

    wind = canvas.wind

    prev_center = None
    stable_count = 0
    iteration = 0

    while running:
        t_now = time.time()

        if (t_now - t_logic) >= dt:
            multi_agent.update(t_sim, dt, wind)
            # break
            for idx, agent in enumerate(multi_agent.agents):
                # Simpan data untuk plotting
                history_agent[idx]['t'].append(t_sim)

                history_agent[idx]['xa'].append(agent.position[0])
                history_agent[idx]['ya'].append(agent.position[1])
                history_agent[idx]['za'].append(agent.position[2])
                history_agent[idx]['v_xa'].append(agent.velocity[0])
                history_agent[idx]['v_ya'].append(agent.velocity[1])
                history_agent[idx]['v_za'].append(agent.velocity[2])
                history_agent[idx]['mode'].append(agent.mode.value)
                history_agent[idx]['scale'].append(agent.scaling_factor)

            for idx, quad in enumerate(multi_agent.quadcopters):
                # Simpan data untuk plotting
                history_quad[idx]['t'].append(t_sim)
                history_quad[idx]['x'].append(quad.x)
                history_quad[idx]['y'].append(quad.y)
                history_quad[idx]['z'].append(quad.z)
                history_quad[idx]['q0'].append(quad.q0)
                history_quad[idx]['q1'].append(quad.q1)
                history_quad[idx]['q2'].append(quad.q2)
                history_quad[idx]['q3'].append(quad.q3)
                history_quad[idx]['phi'].append(quad.phi)
                history_quad[idx]['theta'].append(quad.theta)
                history_quad[idx]['psi'].append(quad.psi)
                history_quad[idx]['v_x'].append(quad.vx)
                history_quad[idx]['v_y'].append(quad.vy)
                history_quad[idx]['v_z'].append(quad.vz)                
                history_quad[idx]['p'].append(quad.p)
                history_quad[idx]['q'].append(quad.q)
                history_quad[idx]['r'].append(quad.r)
                history_quad[idx]['wM1'].append(quad.wM1)
                history_quad[idx]['wM2'].append(quad.wM2)
                history_quad[idx]['wM3'].append(quad.wM3)
                history_quad[idx]['wM4'].append(quad.wM4)

                history_quad[idx]['x_sp'].append(quad.xDes)
                history_quad[idx]['y_sp'].append(quad.yDes)
                history_quad[idx]['z_sp'].append(quad.zDes)
                history_quad[idx]['q0_des'].append(quad.q0Des)
                history_quad[idx]['q1_des'].append(quad.q1Des)
                history_quad[idx]['q2_des'].append(quad.q2Des)
                history_quad[idx]['q3_des'].append(quad.q3Des)
                history_quad[idx]['phi_sp'].append(quad.phiDes)
                history_quad[idx]['theta_sp'].append(quad.thetaDes)
                history_quad[idx]['psi_sp'].append(quad.psiDes)
                history_quad[idx]['v_x_sp'].append(quad.vxDes)
                history_quad[idx]['v_y_sp'].append(quad.vyDes)
                history_quad[idx]['v_z_sp'].append(quad.vzDes)
                history_quad[idx]['p_des'].append(quad.pDes)
                history_quad[idx]['q_des'].append(quad.qDes)
                history_quad[idx]['r_des'].append(quad.rDes)
                history_quad[idx]['x_thr_sp'].append(quad.xthrDes)
                history_quad[idx]['y_thr_sp'].append(quad.ythrDes)
                history_quad[idx]['z_thr_sp'].append(quad.zthrDes)
                history_quad[idx]['yaw_rate_sp'].append(quad.yawRateDes)
                history_quad[idx]['wM1_sp'].append(quad.wM1_cmd)
                history_quad[idx]['wM2_sp'].append(quad.wM2_cmd)
                history_quad[idx]['wM3_sp'].append(quad.wM3_cmd)
                history_quad[idx]['wM4_sp'].append(quad.wM4_cmd)

            t_logic = t_now
            t_sim += dt

            # Cek tabrakan
            if multi_agent.check_collision():
                print("Collision detected! Stopping simulation.")
                running = False
                break

            # Cek apakah sudah mencapai goal dan formasi stabil
            goal_reached, center, stable_count = multi_agent.check_reach_goal(prev_center, stable_count)
            prev_center = center
            # print("Center:", center)

            if goal_reached:
                print("Goal reached and formation stable. Stopping simulation.")
                running = False
                break

            iteration += 1

        if (t_now - t_render) >= frame_time and (iteration % 100 == 0):
            # Update drawing tiap quadcopter
            for quad in multi_agent.quadcopters:
                quad.drawer.update_draw(quad.position, quad.quat, label=f'Q{quad.id}')
            
            canvas.draw(t=t_sim, center=center, quadcopters=multi_agent.quadcopters)

            if save_video:
                writer.grab_frame()

            plt.pause(0.001)
            t_render = t_now

    plt.ioff()
    plt.show()

    if save_video:
        writer.finish()
        print(f"Video saved to {video_filename}")

    data_agent = []
    for idx in range(multi_agent.num_robot):
        h = history_agent[idx]
        T = len(h['t'])
        arr_agent = np.zeros((T, 9))

        arr_agent[:, 0] = h['t']
        arr_agent[:, 1] = h['xa']
        arr_agent[:, 2] = h['ya']
        arr_agent[:, 3] = h['za']
        arr_agent[:, 4] = h['v_xa']
        arr_agent[:, 5] = h['v_ya']
        arr_agent[:, 6] = h['v_za']
        arr_agent[:, 7] = h['mode']
        scale_arr = np.array(h['scale'])
        scale_arr[scale_arr == -1] = 0
        arr_agent[:, 8] = scale_arr
        data_agent.append({'path': arr_agent})

    data_quad = []
    for idx in range(multi_agent.num_robot):
        h = history_quad[idx]
        T = len(h['t'])
        arr_quad = np.zeros((T, 45))

        arr_quad[:, 0] = h['t']
        arr_quad[:, 1] = h['x']
        arr_quad[:, 2] = h['y']
        arr_quad[:, 3] = h['z']
        arr_quad[:, 4] = h['q0']
        arr_quad[:, 5] = h['q1']
        arr_quad[:, 6] = h['q2']
        arr_quad[:, 7] = h['q3']
        arr_quad[:, 8] = h['phi']
        arr_quad[:, 9] = h['theta']
        arr_quad[:, 10] = h['psi']
        arr_quad[:, 11] = h['v_x']
        arr_quad[:, 12] = h['v_y']
        arr_quad[:, 13] = h['v_z']
        arr_quad[:, 14] = h['p']
        arr_quad[:, 15] = h['q']
        arr_quad[:, 16] = h['r']
        arr_quad[:, 17] = h['wM1']
        arr_quad[:, 18] = h['wM2']
        arr_quad[:, 19] = h['wM3']
        arr_quad[:, 20] = h['wM4']

        arr_quad[:, 21] = h['x_sp']
        arr_quad[:, 22] = h['y_sp']
        arr_quad[:, 23] = h['z_sp']
        arr_quad[:, 24] = h['q0_des']
        arr_quad[:, 25] = h['q1_des']
        arr_quad[:, 26] = h['q2_des']
        arr_quad[:, 27] = h['q3_des']
        arr_quad[:, 28] = h['phi_sp']
        arr_quad[:, 29] = h['theta_sp']
        arr_quad[:, 30] = h['psi_sp']
        arr_quad[:, 31] = h['v_x_sp']
        arr_quad[:, 32] = h['v_y_sp']
        arr_quad[:, 33] = h['v_z_sp']
        arr_quad[:, 34] = h['p_des']
        arr_quad[:, 35] = h['q_des']
        arr_quad[:, 36] = h['r_des']
        arr_quad[:, 37] = h['yaw_rate_sp']
        arr_quad[:, 38] = h['x_thr_sp']
        arr_quad[:, 39] = h['y_thr_sp']
        arr_quad[:, 40] = h['z_thr_sp']
        arr_quad[:, 41] = h['wM1_sp']
        arr_quad[:, 42] = h['wM2_sp']
        arr_quad[:, 43] = h['wM3_sp']
        arr_quad[:, 44] = h['wM4_sp']

        data_quad.append(
            {
                'id': idx + 1,
                'path': arr_quad
            })


    # file_name_agent = f"multi_agent_data_{obstacle_scheme}_{wind_type}_{controller_type}_formation{formation_type}_{orient}_erc_scheme1.pkl"
    # file_name_quad = f"multi_quad_data_{obstacle_scheme}_{wind_type}_{controller_type}_formation{formation_type}_{orient}_erc_scheme1.pkl"
    
    # import pickle
    # with open(file_name_agent, 'wb') as f:
    #     pickle.dump({
    #         'data_agent': data_agent
    #     }, f)
    # print(f"Data saved to {file_name_agent}")

    # with open(file_name_quad, 'wb') as f:
    #     pickle.dump({
    #         'data_quad': data_quad
    #     }, f)
    # print(f"Data saved to {file_name_quad}")

if __name__ == "__main__":
    main()
    # main_multi_agent()