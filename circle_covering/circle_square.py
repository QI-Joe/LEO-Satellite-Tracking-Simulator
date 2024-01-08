import math
from typing import *
import os
print(os.getcwd())

from gdclass import GD


# unit is set as km per hour
statellite_speed = 28_800
cruising_aircraft = 926
number_of_statellite = 5
escape_time = 2 # hours
R_Earth: int = 6_371_000 # in meters

# ---------basic settings-----------
i_o = math.radians(97)
Omega_o = 0
e_o = 0
omega_o = 0
M_o = 0
circle_o = 14
off_nadir = 45
T = 86400/circle_o
ground_lat_list = [80, 45, 10]
ground_long_list = [10*x for x in range(36)]

# ----------观测区域经纬度
gd_lines = []
obs_f = open('settings/OBSERVATION.txt', 'r')
for line in obs_f.readlines():
    gd_lines.append(line.split(' '))
obs_f.close()
gd_accounts = len(gd_lines)
gd_list = []
for g in range(gd_accounts):
    region_lat = float(gd_lines[g][0])
    region_long = float(gd_lines[g][1])
    region_lat_rad = math.radians(region_lat)       # 弧度
    region_long_rad = math.radians(region_long)     # 弧度
    gd = GD(region_lat_rad, region_long_rad)
    gd_list.append(gd)

print(gd_list)

def curving_area_from_earth_centre(phi: Union[float, int], extra_r: int):
    angle = phi*2 # phi is expected to be in 360 degree
    plane_prob = angle/360
    area = plane_prob * math.pi * R_Earth**2

    sub_radius = R_Earth * math.sin(phi)
    h = R_Earth - R_Earth * math.cos(phi)
    hemisphere = 2*math.pi*h*R_Earth

    return round(area/10e3, 2), round(hemisphere/(10e2*1.5), 2)

def curving_area_from_statellite(beta: Union[float, int]):
    ...



    