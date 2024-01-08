from scipy.spatial import Voronoi, Delaunay
from scipy import optimize

import numpy as np
from numpy.linalg import norm
from itertools import islice
from constr import *
from plot_and_thread import *
from circle_square import *
from satclass import * 
from imaging import *    
import time


N = 5
R = 4

rng = np.random.default_rng()


def normalize(v):
    return v / norm(v)


def in_circle(p):
    return p.dot(p) <= R**2

def line_circle_intersection(p1, p2, finite=True):
    '''
    |(1-t)p1 + t p2|^2 == R^2
    |dp t + p1|^2 - R^2 == 0 (dp = p2-p1)
    dp.dp t^2 + 2 dp.p1 t + p1.p1 - R^2
    at^2 - 2bt + c == 0 (a = dp.dp, b = -dp.p1, c = p1.p1 - R^2)
    delta = b^2-ac = (dp.p1)^2 - (dp.dp)(p1.p1) + (dp.dp)R^2
    t = (b +/- sqrt(delta)) / a
    '''
    dp = p2 - p1
    b = -p1.dot(dp)
    a = dp.dot(dp)
    delta = R**2 * a - np.cross(p1, p2)**2
    if delta < 0:
        return []
    delta_sqrt = np.sqrt(delta)
    pts = []
    for s in (-1, +1):
        t = (b + s*delta_sqrt)/a
        if t >= 0 and (not finite or t <= 1):
            pts.append(p1 + t*dp)
    return pts


def circle_circle_intersection(p1, p2, r):
    dp = (p2 - p1)/2
    d_sq = np.dot(dp, dp)
    if d_sq > r**2:
        return None
    m = (p1 + p2) / 2
    perp = perp_2d(dp)
    perp = perp * np.sqrt(r**2/d_sq - 1)
    return (m + perp, m - perp)


def min_cover_radius(points):
    points = np.reshape(points, (-1, 2))
    # make sure all points are inside the circle
    if not all(in_circle(p) for p in points):
        return max(norm(p) for p in points)
    # filter out near duplicate points that may cause an issue with
    # computing the voronoi diagram
    points_dedup = []
    for point in points:
        if all(norm(point - existing) > 1e-3*R for existing in points_dedup):
            points_dedup.append(point)
    points = points_dedup
    # if there are too few points, we cannot compute the voronoi diagram;
    # but the result would be bad anyway so we just reject those cases by
    # returning a high value
    if len(points) < 4:
        return R
    vor = Voronoi(points)
    center = np.mean(vor.vertices, axis=0)
    radius = min(norm(p) for p in points)
    # for every edge of the voronoi diagram that crosses the circle, compute
    # its intersection and compare that points distance to the centers of
    # the adjacent cells
    for (i1, i2), vert in zip(vor.ridge_points, vor.ridge_vertices):
        p1 = vor.points[i1]
        p2 = vor.points[i2]
        assert len(vert) == 2
        # if a vertex index is less than zero, the edge goes to infinity
        v1 = vor.vertices[vert[0]] if vert[0] >= 0 else None
        v2 = vor.vertices[vert[1]] if vert[1] >= 0 else None
        assert v1 is not None or v2 is not None
        in1 = v1 is not None and in_circle(v1)
        in2 = v2 is not None and in_circle(v2)
        # test if the edge crosses the circle
        if not in1 or not in2:
            finite = True
            # make v1 finite and v2 finite or infinite
            if v1 is None:
                (v1, v2) = (v2, v1)
            # find effective endpoint of half-infinite line
            if v2 is None:
                # the edge is perpendicular to the line between the two points
                n = p2 - p1
                t = np.array((-n[1], n[0]))
                # m = (p1 + p2) / 2
                # if the edge direction points towards the center, reverse it
                if (v1-center).dot(t) < 0:
                    t = -t
                # compute "effective" endpoint
                v2 = v1 + t
                finite = False
            for v in line_circle_intersection(v1, v2, finite):
                radius = max(radius, norm(v - p1), norm(v - p2))
    # for every vertex of the voronoi diagram inside the circle, compare its
    # distance to the centers of the adjacent cells
    for p, ri in zip(vor.points, vor.point_region):
        for vi in vor.regions[ri]:
            if vi >= 0:
                v = vor.vertices[vi]
                if in_circle(v):
                    radius = max(radius, norm(v - p))

    return radius


def local_minimize(fun, x0, *args, **kwargs):
    p0 = np.reshape(x0, (-1, 2))
    r0 = fun(x0)
    print('starting local optimization...')

    triang = Delaunay(p0)
    assert len(triang.coplanar) == 0
    triple_points = []
    for idxs in triang.simplices:
        p = p0[idxs]
        q = np.roll(p, 1, axis=0)
        if all(norm(p-q, axis=1) <= 2*r0):
            triple_points.append(idxs)
    triple_points = np.array(triple_points)

    edge_points = []
    for i1, p1 in enumerate(p0):
        for i2, p2 in islice(enumerate(p0), i1 + 1, None):
            qs = circle_circle_intersection(p1, p2, r0)
            if qs is None:
                continue
            for q in qs:
                if norm(q) >= R:
                    edge_points.append((i1, i2))
                    # ax.add_patch(plt.Circle(q, 0.1, color='blue'))

    constr = []
    if len(triple_points) > 0:
        constr.append(triple_point_constr(triple_points))

    if len(edge_points) > 0:
        constr.append(edge_point_constr(edge_points))

    x = np.concatenate((x0.flatten(), (r0,)))

    n = len(x)
    j = np.zeros((n,))
    j[-1] = 1
    h = np.zeros((n, n))
    result = optimize.minimize(lambda x: x[-1], x, method='SLSQP',
                               jac=lambda _: j,  # hess=lambda _: h,
                               constraints=[{'type': 'ineq', 'fun': c.fun,
                                             'jac': c.jacobian} for c in constr]
                               )
    # print(result)
    x = result.x[:-1]
    r = result.x[-1]
    r_actual = fun(x)
    # print(r_actual, r, r_actual-r)
    # if r < R and r_actual < R:
    #    assert(abs(r_actual-r) < 1e-6)
    print(
        f'improved from {r0:.6f} to {r:.6f} / {r_actual:.6f}: {result.message}')
    result.fun = r_actual
    result.x = x
    return result

def area_map(actual, shrink, coordinating):
    expanding_ratio = actual/shrink
    coor = [coor * expanding_ratio for coor in coordinating]
    return coor


# 儒略日计算
# 输入：年月日时分秒
# 输出：当前时刻的儒略日
def julian2(year, month, day, hour, min, sec):
    if month == 1 or month == 2:
        f = year - 1
        g = month + 12
    if month >= 3:
        f = year
        g = month
    mid1 = math.floor(365.25 * f)
    mid2 = math.floor(30.6001 * (g + 1))
    A = 2 - math.floor(f / 100) + math.floor(f / 400)
    J = mid1 + mid2 + day + A + 1720994.5
    JDE = float(J+hour/24+min/1440+sec/86400)
    return JDE

def greenwich(jd):
    T = (jd-2451545.0)/36525
    return 280.46061837+360.98564736629*(jd-2451545.0)+0.000387933*T*T-T*T*T/38710000

if __name__ == '__main__':
    fun = plotting_wrapper(min_cover_radius)
    try:
        bounds = [(-R, +R) for _ in range(2*N)]
        res = optimize.dual_annealing(fun, bounds,
                                      maxiter=2_000, initial_temp=7000,
                                      minimizer_kwargs={'method': local_minimize})
        fun.print_best()
        print('done!')
    except KeyboardInterrupt:
        pass
    print("The one last result is:\n", res)

    # -----------------------------------------------------------------------------
    # unit is set as km per hour

    statellite_speed = 28_800
    cruising_aircraft = 926
    number_of_statellite = 5
    escape_time = 2 # hours
    R_Earth: int = 6_371_000 # in meters

    start_time = time.time()
    requestNum = 144

    # ---------read start time and end time------------------
    time_f = open('settings/TIME_INTERVAL.txt', 'r')
    time_lines = []
    for line in time_f.readlines():
        time_lines.append(line.split())
    start_time_julian = julian2(int(time_lines[0][0]), int(time_lines[0][1]), int(time_lines[0][2]),
                                        int(time_lines[0][3]), int(time_lines[0][4]), int(time_lines[0][5]))
    end_time_julian = julian2(int(time_lines[1][0]), int(time_lines[1][1]), int(time_lines[1][2]),
                                        int(time_lines[1][3]), int(time_lines[1][4]), int(time_lines[1][5]))
    time_interval = (end_time_julian-start_time_julian)*86400  # 单位s
    start_greenwich = (greenwich(start_time_julian)) % 360   # 转到0到360°

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

    # -----------------------------------------------------------------------------

    statellite = Sat(start_time_julian, i_o, Omega_o, e_o, omega_o, M_o, circle_o, start_time_julian)

    bool_v, phi, beta = is_visible(1, statellite, gd_list[0], off_nadir, start_greenwich)

    area = curving_area_from_earth_centre(phi=phi, extra_r=160_000)
    new_destinate = area_map(area[0], R, res["x"])
    print(new_destinate)