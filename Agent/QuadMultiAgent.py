import numpy as np

from Agent.Quadcopter import Quadcopter
from Agent.MultiAgentConfig import MultiAgentConfig
from Agent.MultiAgentUtils import nearest_point_to_obstacle
from Environment.Obstacles import Obstacles

class QuadMultiAgent:
    def __init__(self, ax, agents_config, obstacle_scheme='scheme1'):
        self.quadcopters = []
        list_init_pos = []
        for params in agents_config:
            init_pos = params.get('init_pos', None)
            quad = Quadcopter(
                            id=params.get('id'),
                            ax=ax,
                            init_pos=params.get('init_pos', None),
                            front_color=params.get('front_color', 'red'),
                            ctrlType=params.get('ctrlType', "xyz_pos"),
                            trajSelect=params.get('trajSelect', (0,0,0)),
                            wayPoints=params.get('wayPoints', None)
                            )
            self.quadcopters.append(quad)

            if init_pos is not None:
                list_init_pos.append((init_pos))

        self.initial_positions = np.array(list_init_pos)
        self.num_robot = len(self.initial_positions)

        obstacles = Obstacles(scheme=obstacle_scheme)
        self.obstacles_2d = obstacles.obstacles_2d
        self.obstacle_heights = obstacles.obstacle_heights

        self.multi_agent_config = MultiAgentConfig()
        self.multi_agent_config.set_num_robot(self.num_robot)

        self.agents = []
        for i in range(self.num_robot):
            if self.multi_agent_config.CONTROLLER == 'iapf':
                from Agent.MultiAgentIAPF import MultiAgentMethod
            elif self.multi_agent_config.CONTROLLER == 'erc':
                from Agent.MultiAgentERC import MultiAgentMethod

            agent = MultiAgentMethod(
                index=i,
                position=self.initial_positions[i],
                config=self.multi_agent_config,
                num_robot=self.num_robot,
                initial_positions=self.initial_positions,
                obstacles_2d=self.obstacles_2d,
                obstacle_heights=self.obstacle_heights
            )
            self.agents.append(agent)

    def initialize_draw(self):
        for quad in self.quadcopters:
            quad.initialize_draw()

    def update_agents(self, t, dt):
        """
        Update posisi agent MultiAgentIAPF dan set posisi desired ke QuadTrajectory.
        """
        for i, agent in enumerate(self.agents):
            agent.compute_control(self.agents, dt=dt)
            desired_pos = agent.position
            self.quadcopters[i].trajectory.set_external_desired_pos(desired_pos)

    def update_quadcopters(self, t, dt, wind):
        """
        Update posisi fisik dan visual quadcopter.
        """
        for quad in self.quadcopters:
            quad.update(t, dt, wind)

    def update(self, t, dt, wind):
        self.update_agents(t, dt)
        self.update_quadcopters(t, dt, wind)

    def check_collision(self):
        for i, quad in enumerate(self.quadcopters):
            quad_pos = quad.position
            for obs_idx, obstacle in enumerate(self.obstacles_2d):
                z_min, z_max = self.obstacle_heights[obs_idx]  
                if not (quad_pos[2] + self.multi_agent_config.ROBOT_RADIUS < z_min or
                        quad_pos[2] - self.multi_agent_config.ROBOT_RADIUS > z_max):
                    obs_point_2d = nearest_point_to_obstacle(quad_pos[:2], obstacle)
                    if np.linalg.norm(quad_pos[:2] - obs_point_2d) < self.multi_agent_config.ROBOT_RADIUS:
                        print(f"Tabrakan: Agent {i} dengan obstacle {obs_idx}")
                        return True

        for i in range(len(self.quadcopters) - 1):
            for j in range(i + 1, len(self.quadcopters)):
                if np.linalg.norm(self.quadcopters[i].position - self.quadcopters[j].position) < 2 * self.multi_agent_config.ROBOT_RADIUS:
                    print(f"Tabrakan: Agent {i} dengan Agent {j}")
                    return True
        return False


    def check_reach_goal(self, prev_center, stable_count):
        all_positions = np.array([quad.position for quad in self.quadcopters])
        center = np.mean(all_positions, axis=0)

        all_velocities = np.array([quad.velocity for quad in self.quadcopters])
        sum_vel = np.sum(all_velocities, axis=0)
        avg_velocity = np.linalg.norm(sum_vel) / self.num_robot

        goal = np.array([
            self.multi_agent_config.XGOAL,
            self.multi_agent_config.YGOAL,
            self.multi_agent_config.ZGOAL
        ])
        dist_to_goal = np.linalg.norm(center - goal)

        new_stable_count = stable_count
        goal_reached = False

        if dist_to_goal < 0.1 and avg_velocity < 0.1:
            if prev_center is not None and np.linalg.norm(center - prev_center) < 0.1:
                new_stable_count += 1
            else: 
                new_stable_count = 0
            if new_stable_count >= 3:
                print("Formasi sudah stabil di goal. Simulasi berhenti.")
                goal_reached = True
        else: 
            new_stable_count = 0
        return goal_reached, center.copy(), new_stable_count



