import numpy as np
from numpy import sin, cos, pi

deg2rad = pi / 180.0

class Wind:
    def __init__(self, wind_type='NONE', n_points=5):
        self.windType = wind_type.upper()
        self.xlim = (-5, 5)
        self.ylim = (-5, 5)
        self.zlim = (0, 5)
        self.n_points = n_points

        self.generate_grid()

        if self.windType == 'FIXED':
            self.velW_med = 1.0
            self.qW1_med = 180 * deg2rad
            self.qW2_med = 0 * deg2rad

        elif self.windType == 'GUST':
            self.velW_med = 2.0
            self.qW1_med = 180 * deg2rad
            self.qW2_med = -5 * deg2rad

            self.gust_amplitude = 2.0
            self.gust_frequency = 0.25  # Hz
            self.gust_phase = 0.0

        elif self.windType == 'NONE':
            self.velW_med = 0.0
            self.qW1_med = 0.0
            self.qW2_med = 0.0

        else:
            raise Exception(f"Invalid wind type: {self.windType}")

    def set_limits(self, xlim=None, ylim=None, zlim=None):
        if xlim is not None:
            self.xlim = xlim
        if ylim is not None:
            self.ylim = ylim
        if zlim is not None:
            self.zlim = zlim
        self.generate_grid()

    def generate_grid(self):
        self.xs = np.linspace(self.xlim[0], self.xlim[1], self.n_points)
        self.ys = np.linspace(self.ylim[0], self.ylim[1], self.n_points)
        self.zs = np.linspace(self.zlim[0], self.zlim[1], self.n_points)

        self.positions = [np.array([x, y, z]) for x in self.xs for y in self.ys for z in self.zs]
        self.vectors = [np.zeros(3) for _ in self.positions]

    def evaluate_wind(self, t):
        if self.windType == 'GUST':
            # Burst muncul setiap 15 detik, bertahan selama 5 detik
            burst_interval = 10.0  # detik
            burst_duration = 4.0   # detik
            time_in_cycle = t % burst_interval

            if time_in_cycle < burst_duration:
                base = self.velW_med  # kecepatan dasar
                fluctuation = self.gust_amplitude * abs(sin(2 * pi * self.gust_frequency * t + self.gust_phase))
                velW = base + fluctuation
            else:
                # Di luar burst → tidak ada angin
                velW = 0.0
        else:
            velW = self.velW_med

        return velW, self.qW1_med, self.qW2_med

    def get_wind_vector(self, t):
        vel, qW1, qW2 = self.evaluate_wind(t)
        vx = vel * cos(qW2) * cos(qW1)
        vy = vel * cos(qW2) * sin(qW1)
        vz = vel * sin(qW2)
        return np.array([vx, vy, vz])

    def update(self, t):
        for i, pos in enumerate(self.positions):
            self.vectors[i] = self.get_wind_vector(t)

    def draw(self, ax, scale=1.0):
        for pos, vec in zip(self.positions, self.vectors):
            if np.linalg.norm(vec) < 1e-6:
                continue
            ax.quiver(pos[0], pos[1], pos[2],
                      vec[0], vec[1], vec[2],
                      length=scale, normalize=True,
                      color='blue', alpha=0.1,
                      arrow_length_ratio=0.3)

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    wind = Wind(wind_type='GUST')

    times = np.linspace(0, 60, 1000)
    speeds = []

    for t in times:
        speed, _, _ = wind.evaluate_wind(t)
        speeds.append(speed)

    plt.figure(figsize=(6.85, 2.85))
    plt.plot(times, speeds, label='Wind Speed (m/s)', color='blue')
    # plt.title('Wind Gust Model')
    plt.xlabel('Time (seconds)', fontname='Arial', fontsize=12)
    plt.ylabel('Wind Speed (m/s)', fontname='Arial', fontsize=12)
    plt.xticks(fontname='Arial', fontsize=12)
    plt.yticks(fontname='Arial', fontsize=12)
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.show()