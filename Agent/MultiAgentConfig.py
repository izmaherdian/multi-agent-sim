import numpy as np

class MultiAgentConfig:
    def __init__(self):
        self.dt                 = 0.005
        self.fps                = 30
        self.frame_time         = 1.0 / self.fps

        self.ROBOT_RADIUS       = 0.2
        self.ALERT_RADIUS       = 3 * self.ROBOT_RADIUS
        self.SENSING_RADIUS     = 3.0

        # self.EPSILON            = 0.1
        self.ALPHA              = 8
        self.VREF               = 0.5
        self.VMAX               = 1.0
        self.UMAX               = 2.0
        self.UREF               = np.array([1,0,0])
        self.DREF               = 1.0
        self.W_form             = 1.0
        self.W_tail             = 1.2
        self.W_obs              = 8.0
        self.W_col              = 8.0
        self.W_rand             = 2e-2
        self.KP_TAKEOFF         = 1.5
        self.TARGET_ALTITUDE    = -2.0
        self.ALTITUDE_TOLERANCE = 0.1
        self.VMAX_Z_TAKEOFF     = 1.0

        self.PATH_TYPE          = 'goal' # or goal, multi-goal, circular

        # Parameter untuk 'goal'
        self.XGOAL = 22.0 # 22.0
        self.YGOAL = 3.0 # 3.0
        self.ZGOAL = -5.0

        # Parameter untuk 'multi-goal'
        self.PATH_WAYPOINTS = np.array([
            [0.0, 5.0, -5.0],   # Titik 1
            [0.0, 0.0, -5.0],   # Titik 2
            [3.0, -5.0, -5.0],  # Titik 3
            [0.0, -6.0, -5.0],  # Titik 4
            [0.0, 0.0, -5.0]    # Titik 5
        ])
        self.GOAL_REACHED_THRESHOLD = 0.5
        self.TRANSITION_RADIUS = 0.5

        # Parameter untuk 'circular'
        self.CIRCLE_CENTER = np.array([0.0, 0.0, -5.0])  # Pusat lingkaran (x, y, z)
        self.CIRCLE_RADIUS = 4.0                         # Radius lingkaran
        self.LOOK_AHEAD_DISTANCE = 0.5         # Jarak pandang ke depan

        self.OBSTACLE_SCHEME    = 'scheme1' 
        self.WIND_TYPE          = 'NONE' 
        self.CONTROLLER         = 'erc' 
        self.FORMATION_TYPE     = 1 
        self.ORIENT             = "NED"
        self.SAVE_VIDEO         = False

        self.agents_config      = self.default_agents_config()

    def set_num_robot(self, num_robot):
        self.NUM_ROBOT = num_robot
        if self.FORMATION_TYPE == 2:
            self.TOPOLOGY = np.array([[np.cos(2*np.pi/self.NUM_ROBOT*i), np.sin(2*np.pi/self.NUM_ROBOT*i), 0.0] for i in range(self.NUM_ROBOT)])
        elif self.FORMATION_TYPE == 1:
            if self.NUM_ROBOT == 5:
                self.TOPOLOGY = np.array([
                                            [ 1.0, 0.0, 0.0], 
                                            [ 0.0,-0.5, 0.0],
                                            [ 0.0, 0.5, 0.0], 
                                            [-1.0, 1.0, 0.0],
                                            [-1.0,-1.0, 0.0]
                                            ])
            else:
                print(f"Peringatan: Topologi V-shape untuk {self.NUM_ROBOT} robot tidak standar, menggunakan formasi garis.")
                self.TOPOLOGY = np.array([[float(-i) * 0.5, 0.0, 0.0] for i in range(self.NUM_ROBOT)])

    def default_agents_config(self):
        return [
            {
                'id': 1,
                'init_pos': [-3.0, 4.0, 0.0],
                'front_color': 'orange',
                'ctrlType': "xyz_pos",
                'trajSelect': (99, 3, 0),
                'wayPoints': None
            },
            {
                'id': 2,
                'init_pos': [-5.0, 5.0, 0.0],
                'front_color': 'red',
                'ctrlType': "xyz_pos",
                'trajSelect': (99, 3, 0),
                'wayPoints': None
            },
            {
                'id': 3,
                'init_pos': [-2.0, 4.0, 0.0],
                'front_color': 'blue',
                'ctrlType': "xyz_pos",
                'trajSelect': (99, 3, 0),
                'wayPoints': None
            },
            {
                'id': 4,
                'init_pos': [-5.0, -2.0, 0.0],
                'front_color': 'green',
                'ctrlType': "xyz_pos",
                'trajSelect': (99, 3, 0),
                'wayPoints': None
            },
            {
                'id': 5,
                'init_pos': [-2.0, -2.0, 0.0],
                'front_color': 'purple',
                'ctrlType': "xyz_pos",
                'trajSelect': (99, 3, 0),
                'wayPoints': None
            }
        ]