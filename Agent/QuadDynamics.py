import numpy as np

from numpy import pi, sign, sin, cos, arctan2, arcsin
from numpy.linalg import inv
from scipy.integrate import ode

from Agent.QuadUtils.rotationConversion import YPRToQuat, quat2Dcm, quatToYPR_ZYX


orient = "NED"
usePrecession = bool(True)

# Degrees to radians
deg2rad = pi/180.0

def sys_params():
    mB  = 1.2                                   # mass (kg)
    g   = 9.81                                  # gravity (m/s/s)
    dxm = 0.16                                  # arm length (m)
    dym = 0.16                                  # arm length (m)
    dzm = 0.05                                  # motor height (m)
    IB  = np.array([[0.0123, 0,      0     ],
                    [0,      0.0123, 0     ],
                    [0,      0,      0.0224]])  # Inertial tensor (kg*m^2)
    IRzz = 2.7e-5                               # Rotor moment of inertia (kg*m^2)

    params = {}
    params["mB"]   = mB
    params["g"]    = g
    params["dxm"]  = dxm
    params["dym"]  = dym
    params["dzm"]  = dzm
    params["IB"]   = IB
    params["invI"] = inv(IB)
    params["IRzz"] = IRzz
    params["useIntergral"] = bool(False)        # Include integral gains in linear velocity control
    # params["interpYaw"] = bool(False)         # Interpolate Yaw setpoints in waypoint trajectory

    params["Cd"]         = 0.1
    params["kTh"]        = 1.076e-5                 # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
    params["kTo"]        = 1.632e-7                 # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
    params["mixerFM"]    = makeMixerFM(params)      # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds
    params["mixerFMinv"] = inv(params["mixerFM"])
    params["minThr"]     = 0.1*4                    # Minimum total thrust
    params["maxThr"]     = 9.18*4                   # Maximum total thrust
    params["minWmotor"]  = 75                       # Minimum motor rotation speed (rad/s)
    params["maxWmotor"]  = 925                      # Maximum motor rotation speed (rad/s)
    params["tau"]        = 0.015                    # Value for second order system for Motor dynamics
    params["kp"]         = 1.0                      # Value for second order system for Motor dynamics
    params["damp"]       = 1.0                      # Value for second order system for Motor dynamics
    
    params["motorc1"]    = 8.49                     # w (rad/s) = cmd*c1 + c0 (cmd in %)
    params["motorc0"]    = 74.7
    params["motordeadband"] = 1   
    # params["ifexpo"] = bool(False)
    # if params["ifexpo"]:
    #     params["maxCmd"] = 100                    # cmd (%) min and max
    #     params["minCmd"] = 0.01
    # else:
    #     params["maxCmd"] = 100
    #     params["minCmd"] = 1
    
    return params


def makeMixerFM(params):
    dxm = params["dxm"]
    dym = params["dym"]
    kTh = params["kTh"]
    kTo = params["kTo"] 

    # Motor 1 is front left, then clockwise numbering.
    # A mixer like this one allows to find the exact RPM of each motor 
    # given a desired thrust and desired moments.
    # Inspiration for this mixer (or coefficient matrix) and how it is used : 
    # https://link.springer.com/article/10.1007/s13369-017-2433-2 (https://sci-hub.tw/10.1007/s13369-017-2433-2)
    if (orient == "NED"):
        mixerFM = np.array([[    kTh,      kTh,      kTh,      kTh],
                            [dym*kTh, -dym*kTh,  -dym*kTh, dym*kTh],
                            [dxm*kTh,  dxm*kTh, -dxm*kTh, -dxm*kTh],
                            [   -kTo,      kTo,     -kTo,      kTo]])
    elif (orient == "ENU"):
        mixerFM = np.array([[     kTh,      kTh,      kTh,     kTh],
                            [ dym*kTh, -dym*kTh, -dym*kTh, dym*kTh],
                            [-dxm*kTh, -dxm*kTh,  dxm*kTh, dxm*kTh],
                            [     kTo,     -kTo,      kTo,    -kTo]])
    
    return mixerFM


def init_cmd(params):
    mB = params["mB"]
    g = params["g"]
    kTh = params["kTh"]
    kTo = params["kTo"]
    c1 = params["motorc1"]
    c0 = params["motorc0"]
    
    # w = cmd*c1 + c0   and   m*g/4 = kTh*w^2   and   torque = kTo*w^2
    thr_hover = mB*g/4.0
    w_hover   = np.sqrt(thr_hover/kTh)
    tor_hover = kTo*w_hover*w_hover
    cmd_hover = (w_hover-c0)/c1

    return [cmd_hover, w_hover, thr_hover, tor_hover]


def init_state(params):
    
    x0     = 0.  # m
    y0     = 0.  # m
    z0     = 0.  # m
    phi0   = 0.  # rad
    theta0 = 0.  # rad
    psi0   = 0.  # rad

    quat = YPRToQuat(psi0, theta0, phi0)
    
    if (orient == "ENU"):
        z0 = -z0

    s = np.zeros(21)
    s[0]  = x0       # x
    s[1]  = y0       # y
    s[2]  = z0       # z
    s[3]  = quat[0]  # q0
    s[4]  = quat[1]  # q1
    s[5]  = quat[2]  # q2
    s[6]  = quat[3]  # q3
    s[7]  = 0.       # xdot
    s[8]  = 0.       # ydot
    s[9]  = 0.       # zdot
    s[10] = 0.       # p
    s[11] = 0.       # q
    s[12] = 0.       # r

    w_hover = params["w_hover"]     # Hovering motor speed
    wdot_hover = 0.                 # Hovering motor acc

    s[13] = w_hover
    s[14] = wdot_hover
    s[15] = w_hover
    s[16] = wdot_hover
    s[17] = w_hover
    s[18] = wdot_hover
    s[19] = w_hover
    s[20] = wdot_hover
    
    return s


def init_state_custom_pos(params, pos):
    x0 = init_state(params)   # ambil default state
    x0[0:3] = pos             # ubah hanya posisi (x, y, z)
    return x0


class QuadDynamics:
    def __init__(self, Ti, custom_pos=None, id=None):

        # Quad Params
        # ---------------------------
        self.params = sys_params()
        self.id = id

        # Command for initial stable hover
        # ---------------------------
        ini_hover = init_cmd(self.params)

        self.params["FF"]           = ini_hover[0]              # Feed-Forward Command for Hover
        self.params["w_hover"]      = ini_hover[1]              # Motor Speed for Hover
        self.params["thr_hover"]    = ini_hover[2]              # Motor Thrust for Hover
        self.thr                    = np.ones(4)*ini_hover[2]
        self.params["tor_hover"]    = ini_hover[3]              # Motor Torque for Hover
        self.tor                    = np.ones(4)*ini_hover[3]

        # Initial State
        # ---------------------------
        # Custom posisi saja
        if custom_pos is not None:
            self.state = init_state_custom_pos(self.params, custom_pos)
        else:
            self.state = init_state(self.params)

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]

        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        self.vel_dot = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.acc = np.zeros(3)

        self.extended_state()
        self.forces()

        # Set Integrator
        # ---------------------------
        self.integrator = ode(self.state_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, Ti)

    def extended_state(self):

        # Rotation Matrix of current state (Direct Cosine Matrix)
        self.dcm = quat2Dcm(self.quat)

        # Euler angles of current state
        YPR = quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1] # flip YPR so that euler state = phi, theta, psi
        self.psi   = YPR[0]
        self.theta = YPR[1]
        self.phi   = YPR[2]

    def forces(self):
        
        # Rotor thrusts and torques
        self.thr = self.params["kTh"]*self.wMotor*self.wMotor
        self.tor = self.params["kTo"]*self.wMotor*self.wMotor

    def state_dot(self, t, state, cmd, wind):

        # Import Params
        # ---------------------------    
        mB   = self.params["mB"]
        g    = self.params["g"]
        dxm  = self.params["dxm"]
        dym  = self.params["dym"]
        IB   = self.params["IB"]
        IBxx = IB[0,0]
        IByy = IB[1,1]
        IBzz = IB[2,2]
        Cd   = self.params["Cd"]
        
        kTh  = self.params["kTh"]
        kTo  = self.params["kTo"]
        tau  = self.params["tau"]
        kp   = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if (usePrecession):
            uP = 1
        else:
            uP = 0

        # Import State Vector
        # ---------------------------  
        x      = state[0]
        y      = state[1]
        z      = state[2]
        q0     = state[3]
        q1     = state[4]
        q2     = state[5]
        q3     = state[6]
        xdot   = state[7]
        ydot   = state[8]
        zdot   = state[9]
        p      = state[10]
        q      = state[11]
        r      = state[12]
        wM1    = state[13]
        wdotM1 = state[14]
        wM2    = state[15]
        wdotM2 = state[16]
        wM3    = state[17]
        wdotM3 = state[18]
        wM4    = state[19]
        wdotM4 = state[20]

        # Motor Dynamics and Rotor forces
        # ---------------------------
        
        uMotor = cmd
        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*uMotor[0])/(tau**2)
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*uMotor[1])/(tau**2)
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*uMotor[2])/(tau**2)
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*uMotor[3])/(tau**2)
    
        wMotor = np.array([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh*wMotor*wMotor
        torque = kTo*wMotor*wMotor
    
        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]
        TorM1 = torque[0]
        TorM2 = torque[1]
        TorM3 = torque[2]
        TorM4 = torque[3]

        # Wind Model
        # ---------------------------
        [velW, qW1, qW2] = wind.evaluate_wind(t)
        # velW = 0          # m/s
        # velW = 5          # m/s
        # qW1 = 0*deg2rad   # Wind heading
        # qW2 = 60*deg2rad  # Wind elevation (positive = upwards wind in NED, positive = downwards wind in ENU)

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        if (orient == "NED"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 - 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 + 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 - (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + ( ThrM1 + ThrM2 - ThrM3 - ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IByy)*p*q - TorM1 + TorM2 - TorM3 + TorM4)/IBzz]])
        elif (orient == "ENU"):
            DynamicsDot = np.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 + 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 - 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 + (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) - g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (-ThrM1 - ThrM2 + ThrM3 + ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IBzz)*p*q + TorM1 - TorM2 + TorM3 - TorM4)/IBzz]])

        # State Derivative Vector
        # ---------------------------
        DynamicsDot = DynamicsDot.flatten()
        sdot     = np.zeros([21])
        sdot[0:13] = DynamicsDot
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4

        self.acc = sdot[7:10]

        return sdot
    
    def update(self, t, Ts, cmd, wind):

        prev_vel   = self.vel
        prev_omega = self.omega

        self.integrator.set_f_params(cmd, wind)
        self.state = self.integrator.integrate(t, t+Ts)

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], 
                                self.state[15], 
                                self.state[17], 
                                self.state[19]])

        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts

        self.extended_state()
        self.forces()
