import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Obstacles:
    def __init__(self, scheme='scheme1'):
        # Pilih skema obstacles
        if scheme == 'scheme5':
            # buatkan 3 obstacles saja di -2.5,2.5 lalu di 0,2.5 (kecil aja bentuknya jangan besar-besar)
            self.obstacles_2d = [
                np.array([[0.0, 2.5], [0.2, 2.5], [0.2, 3.0], [0.0, 3.0]]),  # persegi kecil
                np.array([[-2.0, 2.5], [-1.5, 2.5], [-1.5, 3.0], [-2.0, 3.0]]),  # persegi kecil
                np.array([[2.0, -1.5], [2.5, -1.5], [2.5, -2.0], [2.0, -2.0]])   # persegi kecil
            ]
            self.obstacle_heights = [
                (-1, 7),
                (-1, 6),
                (0, 5)
            ] 
        elif scheme == 'scheme4':
            # buatkan 3 obstacles saja di -2.5,2.5 lalu di 0,2.5 (kecil aja bentuknya jangan besar-besar)
            self.obstacles_2d = [
                np.array([[0.0, 2.5], [0.2, 2.5], [0.2, 3.0], [0.0, 3.0]]),  # persegi kecil
                np.array([[-2.5, 2.5], [-1.5, 2.5], [-1.5, 3.0], [-2.5, 3.0]]),  # persegi kecil
                np.array([[2.0, -1.5], [2.5, -1.5], [2.5, -2.0], [2.0, -2.0]])   # persegi kecil
            ]
            self.obstacle_heights = [
                (-1, 7),
                (-1, 6),
                (0, 5)
            ] 
        elif scheme == 'scheme3':
            self.obstacles_2d = [
                # Dinding atas (horizontal atas)
                np.array([[11.0, 5.5], [15.0, 5.5], [15.0, 6.0], [11.0, 6.0]]),

                # Dinding bawah (horizontal bawah)
                np.array([[11.0, 0.5], [15.0, 0.5], [15.0, 1.0], [11.0, 1.0]]),

                # Dinding kanan (penutup U)
                np.array([[14.5, 0.5], [15.0, 0.5], [15.0, 6.0], [14.5, 6.0]])
            ]

            self.obstacle_heights = [
                (0, 7),  # atas
                (0, 7),  # bawah
                (0, 7)   # penutup kanan
            ]

        elif scheme == 'scheme2':
            self.obstacles_2d = [
                np.array([[0.0, 0.0], [ 5.0, 3.0], [15.0, 3.0], [20.0, 0.0]]),
                np.array([[0.0, 6.5], [10.0, 4.0], [15.0, 4.0], [20.0, 6.5]])
            ]
            self.obstacle_heights = [
                (-1, 7),
                (-1, 6)
            ]

        elif scheme == 'scheme1':
            self.obstacles_2d = [
                np.array([[5.0, 4.0], [5.7, 4.8], [6.3, 4.5], [5.9, 3.7]]),
                np.array([[9.1, 1.27], [9.4, 1.3], [9.8, 1.2], [9.6, 1.1]]),
                np.array([[11.5, 5.2], [12.0, 4.9], [12.4, 5.4], [12.1, 5.8]]),
                np.array([[7.0, 4.5], [7.4, 5.0], [8.1, 4.6]]),  # segitiga
                np.array([[14.0, 5.0], [14.5, 5.3], [15.0, 5.1], [15.2, 4.6], [14.6, 4.4]]),  # pentagon acak
                np.array([[13.0, 1.7], [13.5, 0.6], [15.0, 0.5]]),
                np.array([[3.5, 0.0], [4.0, 0.7], [4.8, 0.5], [4.6, -0.1], [4.0, -0.4]]),
                np.array([[16.0, 4.0], [16.6, 4.3], [17.0, 4.0], [16.8, 3.5], [16.2, 3.4]])
            ]
            self.obstacle_heights = [
                (0, 4),
                (0, 5),
                (0, 4.5),
                (0, 5.5),
                (0, 6),
                (0, 5.5),
                (0, 6),
                (0, 7.5)
            ]

        elif scheme == 'NONE':
            # Default empty obstacles
            self.obstacles_2d = []
            self.obstacle_heights = []

        else:
            raise ValueError(f"Skema '{scheme}' tidak dikenali. Pilih 'scheme1', 'scheme2', atau 'NONE'.")

    def _plot_prism(self, ax, polygon_2d, z_min=0, z_max=3, color='gray', alpha=0.3, edge_color='k'):
        xs = [p[0] for p in polygon_2d]
        ys = [p[1] for p in polygon_2d]

        verts_bottom = list(zip(xs, ys, [z_min]*len(xs)))
        verts_top = list(zip(xs, ys, [z_max]*len(xs)))

        ax.add_collection3d(Poly3DCollection([verts_bottom], facecolors=color, alpha=alpha))
        ax.add_collection3d(Poly3DCollection([verts_top], facecolors=color, alpha=alpha))

        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            side = [verts_bottom[i], verts_bottom[j], verts_top[j], verts_top[i]]
            ax.add_collection3d(Poly3DCollection([side], facecolors=color, alpha=alpha))

        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            ax.plot([verts_bottom[i][0], verts_bottom[j][0]], [verts_bottom[i][1], verts_bottom[j][1]], [verts_bottom[i][2], verts_bottom[j][2]], color=edge_color, alpha=0.5)
            ax.plot([verts_top[i][0], verts_top[j][0]], [verts_top[i][1], verts_top[j][1]], [verts_top[i][2], verts_top[j][2]], color=edge_color, alpha=0.5)
            ax.plot([verts_bottom[i][0], verts_top[i][0]], [verts_bottom[i][1], verts_top[i][1]], [verts_bottom[i][2], verts_top[i][2]], color=edge_color, alpha=0.5)

    def draw(self, ax):
        for i, obstacle in enumerate(self.obstacles_2d):
            z_min, z_max = self.obstacle_heights[i]
            self._plot_prism(ax, obstacle, z_min=z_min, z_max=z_max, color='gray', alpha=0.3, edge_color='k')
