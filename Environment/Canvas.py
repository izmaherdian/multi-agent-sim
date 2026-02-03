import numpy as np
import matplotlib.pyplot as plt

from Environment.Obstacles import Obstacles
from Environment.Wind import Wind

class Canvas:
    def __init__(self, obstacle_scheme='scheme1', wind_type='SINE', controller_type='erc', formation_type=2, orient="NED"):
        self.fig = plt.figure()
        self.obstacle_scheme = obstacle_scheme
        self.wind_type = wind_type
        self.controller_type = controller_type
        self.formation_type = formation_type
        self.orient = orient
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        # Set default limits (nanti bisa diupdate dengan data)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(0, 5)

        # Buat obstacles dengan skema yang dipilih
        self.obstacles = Obstacles(scheme=obstacle_scheme)
        self.obstacles.draw(self.ax)

        # Buat wind dengan tipe yang dipilih
        self.wind = Wind(wind_type, n_points=5)

        # Buat waktu
        # self.time_text = self.ax.text2D(...)  # hapus ini
        self.time_text = self.fig.text(0.9, 0.95, "", ha='right', va='top', fontsize=10, color='blue')


    def set_axes_limits_center(self, center, x, y, z, orient="NED"):
        # Hitung selisih (range) di tiap sumbu
        dx = x.max() - x.min()
        dy = y.max() - y.min()
        dz = z.max() - z.min()

        # Ambil range terbesar kemudian bagi dua untuk setengah panjang,
        # lalu tambahkan margin (misal extra_each_side=0.5)
        extra_each_side = 3
        max_range = 0.5 * np.array([dx, dy, dz]).max() + extra_each_side

        # Ambil koordinat pusat
        cx, cy, cz = center

        # Atur sumbu X: [cx - max_range, cx + max_range]
        self.ax.set_xlim3d([cx - max_range, cx + max_range])
        self.ax.set_xlabel('X (m)')

        # Atur sumbu Y: [cy - max_range, cy + max_range] (atau dibalik jika NED)
        if orient == "ENU":
            self.ax.set_ylim3d([cy + max_range, cy - max_range])
        else:  
            self.ax.set_ylim3d([cy - max_range, cy + max_range])
        self.ax.set_ylabel('Y (m)')

        # Atur sumbu Z: [cz - max_range, cz + max_range]
        self.ax.set_zlim3d([-cz - max_range, -cz + max_range])
        self.ax.set_zlabel('Altitude (m)')

    def draw(self, t=0, center=None, quadcopters=[]):
        self.ax.cla()
        if (center is not None) and (quadcopters is not None):
            # Kumpulkan array x,y,z dari semua quad
            x = np.array([q.position[0] for q in quadcopters])
            y = np.array([q.position[1] for q in quadcopters])
            z = np.array([q.position[2] for q in quadcopters])

            # Atur limit 3D agar mengikuti center dan sebaran data
            self.set_axes_limits_center(center, x, y, z, orient=self.orient)

            xlim = self.ax.get_xlim3d()
            ylim = self.ax.get_ylim3d()
            zlim = self.ax.get_zlim3d()
            self.wind.set_limits(xlim=xlim, ylim=ylim, zlim=zlim)
        else:
            # Jika center belum tersedia, pakai default
            # (opsional: bisa skip, atau tetap gunakan axis awal)
            self.ax.set_xlim3d(-5, 5)
            self.ax.set_ylim3d(-5, 5)
            self.ax.set_zlim3d(0, 5)
            self.wind.set_limits(xlim=(-5, 5), ylim=(-5, 5), zlim=(0, 5))

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        # Gambar obstacles
        self.obstacles.draw(self.ax)

        # Gambar wind
        self.wind.update(t)
        self.wind.draw(self.ax, scale=0.5)

        # Gambar quadcopters
        for quad in quadcopters:
            quad.initialize_draw()

        self.ax.set_title(
                f'3D Environment\n'
                f'Obs: {self.obstacle_scheme} - Wind: {self.wind_type} - Contr: {self.controller_type}\n'
                f'Formation: {self.formation_type} - Orient: {self.orient} - Time: {t:.2f}s',
                fontsize=10, 
                fontname='Arial',
                )

    def show(self):
        self.draw(t=0)
        plt.show()