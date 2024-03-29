import random
import os
import xlwt
import time

# import sys
# sys.path.insert(1, 'D:\Code Working Area\Python\FYP\satellite-2022-master\include')

from include import imaging
from include import communication
from include import satcompute
from include import gsexclude
from include import curve
from include import service
from include import validation
from include.greenwich import *
from include.satclass import *
from include.gdclass import *
from include.gsclass import *
from include.requestclass import *


start_time = time.time()
# ---------read start time and end time
time_f = open('../settings/TIME_INTERVAL.txt', 'r')
time_lines = []
for line in time_f.readlines():
    time_lines.append(line.split())
start_time_julian = julian2(int(time_lines[0][0]), int(time_lines[0][1]), int(time_lines[0][2]),
                                      int(time_lines[0][3]), int(time_lines[0][4]), int(time_lines[0][5]))
end_time_julian = julian2(int(time_lines[1][0]), int(time_lines[1][1]), int(time_lines[1][2]),
                                    int(time_lines[1][3]), int(time_lines[1][4]), int(time_lines[1][5]))
time_interval = (end_time_julian-start_time_julian)*86400  # 单位s
start_greenwich = (greenwich(start_time_julian)) % 360   # 转到0到360°



# ----------观测区域经纬度
# gd_lines = []
# obs_f = open('../settings/OBSERVATION.txt', 'r')
# for line in obs_f.readlines():
#     gd_lines.append(line.split(' '))
# obs_f.close()
# gd_accounts = len(gd_lines)
# gd_list = []
# for g in range(gd_accounts):
#     region_lat = float(gd_lines[g][0])
#     region_long = float(gd_lines[g][1])
#     region_lat_rad = math.radians(region_lat)       # 弧度
#     region_long_rad = math.radians(region_long)     # 弧度
#     gd = GD(region_lat_rad, region_long_rad)
#     gd_list.append(gd)
gd_accounts = 1
gd_list = []
region_lat = random.randint(-90, 90)
region_long = random.randint(0, 360)
print("lat:", region_lat)
print("long:", region_long)
region_lat_rad = math.radians(region_lat)       # 弧度
region_long_rad = math.radians(region_long)     # 弧度
gd = GD(region_lat_rad, region_long_rad)
gd_list.append(gd)

# ---------read ground stations
gs_lines = []
input_f2 = open('../settings/SELECT_GROUND_STATION2.txt', 'r')
for line in input_f2.readlines():
    tmpl = line.strip()
    gs_lines.append(tmpl.split(' '))
input_f2.close()
gs_accounts = len(gs_lines)
gs_list = []
for g in range(gs_accounts):
    gs_lat = float(gs_lines[g][0])
    gs_long = float(gs_lines[g][1])
    if gs_long < 0:
        gs_long = 360 + gs_long
    # gs_ele = float(gs_lines[g][3])
    gs_ele = 10
    gs_lat_rad = math.radians(gs_lat)  # 弧度
    gs_long_rad = math.radians(gs_long)  # 弧度
    gs_ele_rad = math.radians(gs_ele)  # 弧度
    gs = GS(gs_lat_rad, gs_long_rad, gs_ele_rad)
    gs_list.append(gs)

# ----------main section
off_nadir = math.radians(45)
i_o = math.radians(97)
e_o = 0
omega_o = 0
circle_o = 14
m = 9
n = 25

request_period = 600   # request period (s)
request_postpone = 60  # request postpone (s)

img_cost = 1           # imaging cost (s)
com_cost = 15          # communication cost (s)
ser_ddl = 150          # service delay (s)

# 删除output文件
if os.path.exists("../results/merge_result_improve.xls"):
    os.remove("../results/merge_result_improve.xls")
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('merge_result', cell_overwrite_ok=True)
col = ('ground latitude', 'ground longitude', 'feasible')
for i in range(0, 3):
    sheet.write(0, i, col[i])

col_num = 1
# ------初始化所有卫星
sat_list = []
first_Omega = 0  # 第一个轨道的升交点赤经
even_Omega = 180 / (m-1)
for orbit_id in range(m):
    Omega_o = math.radians(first_Omega + orbit_id * even_Omega)
    first_M = 0  # 轨道上的第一个卫星的位置
    even_M = 360 / n
    for sat_id in range(n):
        M_o = math.radians(first_M + sat_id * even_M)
        # 令卫星的当前时间为simulation的开始时间
        s = Sat(start_time_julian, i_o, Omega_o, e_o, omega_o, M_o, circle_o, start_time_julian)
        sat_list = sat_list + [s]

for i in range(gd_accounts):
    sheet.write(col_num, 0, math.degrees(gd_list[i].lat_rad))
    sheet.write(col_num, 1, math.degrees(gd_list[i].long_rad))
    # ------求总的image curve
    imaging_curve_lst = []
    psi_gd_lst = []
    start_time1 = time.time()
    for sid in range(m * n):
        imaging_curve = imaging.visible(time_interval + request_period, start_greenwich, sat_list[sid], gd_list[i],
                                        off_nadir)
        imaging_curve_lst.append(imaging_curve)
        psi_gd, phi1, phi2 = satcompute.get_sat_phi_range(math.radians(off_nadir), sat_list[sid].r, gd_list[i].lat_rad)
        psi_gd_lst.append(psi_gd)
    end_time1 = time.time()
    print('imaging overall time:', end_time1 - start_time1)
    # ------累加地面站
    service_curve = [[] for i in range(m*n)]
    valid_flag = False
    for g in range(gs_accounts):
        gs = gs_list[g]
        obs_flag = False
        for sid in range(m * n):
            gs_off_nadir = math.asin(Re * math.cos(gs.ele_rad) / sat_list[sid].r)
            psi_gs, phi3, phi4 = satcompute.get_sat_phi_range(gs_off_nadir, sat_list[sid].r, gs.lat_rad)
            max_rad = sat_list[sid].n_o * ser_ddl + psi_gd_lst[sid] + psi_gs
            if gsexclude.exclude_ground_station(gd_list[i], gs, max_rad):
                continue
            sat_gs_curve = communication.communicable(time_interval + request_period, start_greenwich, sat_list[sid],
                                                      gs)
            sat_obs_curve = service.get_obs_curve(imaging_curve_lst[sid], sat_gs_curve, img_cost, com_cost, ser_ddl)
            if sat_obs_curve:
                obs_flag = True
                service_curve[sid] = curve.merge_curve(service_curve[sid], sat_obs_curve)
        # -------validation
        print("service curve:", service_curve)
        if obs_flag:
            request_num = int(time_interval / request_period)
            request = Request(0, request_period, request_postpone, request_num * request_period)
            vote_list = validation.get_vote_list(service_curve, request, img_cost)
            valid = validation.validation_by_vote_simple(vote_list, request_num, m * n)
            if valid:
                valid_flag = True
                sheet.write(col_num, 2, 'V')
                print('valid')
                break
    if not valid_flag:
        sheet.write(col_num, 2, '-')
        print('invalid')
    col_num = col_num + 1
end_time = time.time()
print('overall time:', end_time - start_time)
book.save('../results/merge_result_improve.xls')






