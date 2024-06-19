import numpy as np
import matplotlib.pyplot as plt
import os
import random
from numba import prange, int32, float32, types, njit, objmode
from numba.experimental import jitclass



# spec = [
#     ('width', int32),
#     ('length', int32),
#     ('resolution', float32),
#     ('thermalheight_avg', float32),
#     ('thermalheight_std', float32),
#     ('thermalstrength_avg', float32),
#     ('thermalstrength_std', float32),
#     ('thermalrad_avg', float32),
#     ('thermalrad_std', float32),
#     ('roi', float32),
#     ('name', types.unicode_type),
#     ('thermals', float32[:,:]),
#     ('field', float32[:,:])
# ]
# @jitclass(spec)
class Windfield():
    def __init__(self, size= 10000, resolution = 10., thermalheight_avg = 1000., thermalheight_std = 30.,
                 thermalstrength_avg = 2., thermalstrength_std = 0.5,
                  thermalrad_avg = 200., thermalrad_std = 50., name = 'noname'):
        self.width = size
        self.length = size
        self.resolution = resolution
        self.thermalheight_avg = thermalheight_avg
        self.thermalheight_std = thermalheight_std
        self.thermalstrength_avg = thermalstrength_avg
        self.thermalstrength_std = thermalstrength_std
        self.thermalrad_avg = thermalrad_avg
        self.thermalrad_std = thermalrad_std
        self.roi = thermalrad_avg * 2.5
        self.name = name
        self.field = np.zeros((round(self.width/resolution), round(self.length/resolution)), dtype=np.single)




    def create_thermals(self):
        N = round(.6 * self.length * self.width / (self.thermalheight_avg * self.thermalrad_avg))  # Number of thermals https://websites.isae-supaero.fr/IMG/pdf/report.pdf
        thermal_coord = np.random.rand(N, 2) * np.array([self.width, self.length])
        thermal_strength = np.random.normal(self.thermalstrength_avg, self.thermalstrength_std, size=N)
        thermal_rad = np.random.normal(self.thermalrad_avg, self.thermalrad_std, size=N)
        self.thermals = np.hstack((thermal_coord, np.column_stack((thermal_strength, thermal_rad))))


    def generate_field(self):
        calculate_field_optimized(self.field, self.resolution, self.thermals)
        # for t in range(len(self.thermals)):
        #     xt, yt, w, R = self.thermals[t,:]
        #
        #     # Calculate the index range of the subarray
        #     delta = (2.5 * R)
        #     i_min = round((xt - delta) / self.resolution)
        #     if i_min < 0:
        #         continue
        #     i_max = round((xt + delta) / self.resolution)
        #     if i_max >= field.shape[0]:
        #         continue
        #     j_min = round((yt - delta) / self.resolution)
        #     if j_min < 0:
        #         continue
        #     j_max = round((yt + delta) / self.resolution)
        #     if j_max > field.shape[1]:
        #         continue
        #
        #     # Calculate the distances and update the field within the subarray
        #     for i in range(i_min, i_max):
        #         for j in range(j_min, j_max):
        #             x = i * self.resolution
        #             y = j * self.resolution
        #             r = np.sqrt((x - xt) * (x - xt) + (y - yt) * (y - yt))
        #             if r < 2.5 * R:
        #                 self.field[i, j] += w * np.exp(-((r * r) / (R * R))) * (1 - ((r * r) / (R * R)))

    def save_field(self, filename=None):
        if filename:
            np.save(f'custom_gym/Windfields/{filename}', self.field)
        else:
            np.save(f'custom_gym/Windfields/{self.name}.npy', self.field)

    def load_field(self, filename = None):
        if filename:
            self.field = np.load(f'custom_gym/Windfields/{filename}')
        else:
            random_file = random.choice(os.listdir("./custom_gym/Windfields"))
            self.field = np.load(f'custom_gym/Windfields/{random_file}')

    def updraft(self, x,y,z):
        if x >= self.width or y >= self.length:
            return -1
        if x < 0 or y < 0:
            return -1
        if z > self.thermalheight_avg: # later height of individual thermals can be taken into account
            return 0

        xf = round(x/self.resolution)
        yf = round(y/self.resolution)
        if xf > self.field.shape[0]-1 or yf > self.field.shape[1]-1:
            return -1

        return self.field[xf, yf]






# @njit(parallel=True)
# def calculate_field(field, width, length, resolution, thermal_strength, thermal_coord, thermal_rad):
#     for i in prange(round(width / resolution)):
#         for j in prange(round(length / resolution)):
#             for t in range(len(thermal_strength)):
#                 x = i * resolution
#                 y = j * resolution
#                 xt = thermal_coord[t, 0]
#                 yt = thermal_coord[t, 1]
#                 R = thermal_rad[t]
#                 if (x-xt)*(x-xt) > 2.5*2.5*R*R:
#                     continue
#                 elif (y-yt)*(y-yt) > 2.5*2.5*R*R:
#                     continue
#                 r = np.sqrt((x - xt) ** 2 + (y - yt) ** 2)
#                 if r < 2.5*R:
#                     field[i, j] += thermal_strength[t] * np.exp(-(r / R) ** 2) * (1 - (r / R) ** 2)

@njit
def calculate_field_optimized(field, resolution, thermals):
    thermal_coord = thermals[:, 0:2]
    thermal_strength = thermals[:, 2]
    thermal_rad = thermals[:, 3]
    for t in range(len(thermal_strength)):
        xt = thermal_coord[t, 0]
        yt = thermal_coord[t, 1]
        R = thermal_rad[t]
        w = thermal_strength[t]

        # Calculate the index range of the subarray
        delta = (2.5 * R)
        i_min = round((xt - delta)/resolution)
        if i_min < 0:
            continue
        i_max = round((xt + delta)/resolution)
        if i_max >= field.shape[0]:
            continue
        j_min = round((yt - delta)/resolution)
        if j_min < 0:
            continue
        j_max = round((yt + delta)/resolution)
        if j_max > field.shape[1]:
            continue

        # Calculate the distances and update the field within the subarray
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                x = i * resolution
                y = j * resolution
                r = np.sqrt((x - xt) * (x - xt) + (y - yt) * (y - yt))
                if r < 2.5 * R:
                    field[i, j] += w * np.exp(-((r * r)/(R * R))) * (1 - ((r * r) / (R * R)))




if __name__ == "__main__":
    length, width = 10000., 10000. # m
    thermalheigth_avg = 1500 # m
    thermalheigth_std = 100 # m
    thermalstrength_avg = 2 # m/s
    thermalstrength_std = 1 # m/s
    thermalwidth_avg = 200 # m
    thermalwidth_std = 50 # m
    resolution = 10 #m
    N = round(.6 * length * width / (thermalheigth_avg * thermalwidth_avg)) # Number of thermals https://websites.isae-supaero.fr/IMG/pdf/report.pdf
    thermal_coord = np.random.rand(N, 2) * np.array([width,length])
    thermal_strength = np.random.normal(thermalstrength_avg, thermalstrength_std, size=N)
    thermal_rad = np.random.normal(thermalwidth_avg, thermalwidth_std, size=N)
    field = np.zeros((round(width/resolution), round(length/resolution))) # lenght y rows, width x columns

    #_____________Numba method
    # Assuming the following variables are defined: width, length, resolution, thermal_strength, thermal_coord, thermal_rad, field
    field = np.zeros((round(width / resolution), round(length / resolution)))
    calculate_field_optimized(field, resolution, thermal_strength, thermal_coord, thermal_rad)

    plt.imshow(field)
    plt.colorbar()
    plt.show()



