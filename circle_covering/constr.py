import numpy as np

N = 5
R = 4

def perp_2d(x, axis=None):
    # return np.array((-x[1], x[0]))
    return np.concatenate((
        -np.take(x, (1,), axis=axis),
        np.take(x, (0,), axis=axis)),
        axis=axis)


def squared_norm(x, axis=None):
    return np.sum(np.square(x), axis=axis)


class triple_point_constr:
    '''
    r >= circumradius of triangle formed by circle centers
    circumradius = (|p1-p2| |p2-p3| |p3-p1|) / (4 area)
    area = 1/2 | x1y2 - x1y3 + x2y3 - x2y1 + x3y1 - x3y2 |
    so:
    (2r(x1y2 - x2y1 + x2y3 - x3y2 + x3y1 - x1y3))^2 - (x1^2+y1^2)(x2^2+y2^2)(x3^2+y3^2) >= 0
    '''

    def __init__(self, indexes):
        self.indexes = np.array(indexes)

    def lb(self):
        # (m,)
        return np.zeros((self.indexes.shape[0],))

    def ub(self):
        # (m,)
        return np.full((self.indexes.shape[0],), np.inf)

    def fun(self, x):
        # (m,)
        p = np.reshape(x[:-1], (-1, 2))
        r = x[-1]

        p = p[self.indexes]
        q = np.roll(p, 1, axis=1)
        d = p-q

        area = np.sum(np.cross(p, q, axisa=2, axisb=2), axis=1)
        sqnorm = squared_norm(d, axis=2)

        return np.square(2*r*area) - np.prod(sqnorm, axis=1)

    def jacobian(self, x):
        # (m,n); d(fun)/dx
        m = len(self.indexes)
        n = len(x)
        k = (n-1)//2

        p = np.reshape(x[:-1], (k, 2))
        r = x[-1]

        q = p[self.indexes]
        d = q - np.roll(q, -1, axis=1)
        s = squared_norm(d, axis=2)[..., np.newaxis]
        area = np.sum(q[:, :, 0]*np.roll(d[:, :, 1], -1, axis=1), axis=1)
        perp = perp_2d(d, axis=2)
        jac_elem = -8*(r**2)*np.roll(perp, -1, axis=1)*area[..., np.newaxis, np.newaxis] - \
            2*np.roll(s, -1, axis=1)*(d*np.roll(s, 1, axis=1) -
                                      np.roll(d, 1, axis=1)*s)

        jac = np.zeros((m, k, 2))
        i = np.repeat(np.arange(m)[..., np.newaxis], 3, axis=1)
        j = self.indexes
        jac[i, j] = jac_elem
        jac = np.reshape(jac, (m, n-1))

        return np.concatenate((jac, 8*r*np.square(area)[..., np.newaxis]), axis=1)

    def hessian(self, x, v):
        # (n,n)
        pass


class edge_point_constr:
    '''
    |intersection of circles|^2 >= R^2
    dp = (p2 - p1)/2
    d_sq = np.dot(dp, dp)
    m = (p1 + p2) / 2
    perp = np.array((-dp[1], dp[0]))
    intersection = m +/- perp * np.sqrt(r**2/d_sq - 1)
    |intersection|^2 = m.m + 2*abs(m.perp sqrt(r**2/d_sq - 1)) + dp.dp (r**2/d_sq - 1) >= R^2
    2*abs(m.perp sqrt(r**2/d_sq - 1)) >= R^2 - m.m - (r**2 - dp.dp)
    (x1y2-x2y1)^2*(r**2/d_sq - 1) >= (R^2 - r**2 - p1.p2)^2
    (x1y2-x2y1)^2*(r**2 - d_sq) >= (R^2 - r**2 - p1.p2)^2*d_sq
    '''

    def __init__(self, indexes):
        self.indexes = np.array(indexes)

    def lb(self):
        # (m,)
        return np.zeros((self.indexes.shape[0],))

    def ub(self):
        # (m,)
        return np.full((self.indexes.shape[0],), np.inf)

    def fun(self, x):
        # (m,)
        p = np.reshape(x[:-1], (-1, 2))
        r = x[-1]

        p = p[self.indexes]
        dp = (p[:, 1] - p[:, 0])/2
        det = np.cross(p[:, 0], p[:, 1], axisa=1, axisb=1)
        dp_sq = squared_norm(dp, axis=1)
        dot = np.sum(p[:, 0]*p[:, 1], axis=1)

        return np.square(det)*(r**2 - dp_sq) - np.square(R**2 - r**2 - dot)*dp_sq

    def jacobian(self, x):
        '''
        1/2 (-("cross"^2 + ("dot" + r^2 - R^2)^2) {x1 - x2, y1 - y2} - 
        "dsq" ("dot" + r^2 - R^2) {x2, y2} - 
        "cross" ("dsq" - 4 r^2) {-y2, x2})
        1/2 (-("cross"^2 + ("dot" + r^2 - R^2)^2) {x2 - x1, y2 - y1} - 
        "dsq" ("dot" + r^2 - R^2) {x1, y1} + <===
        "cross" ("dsq" - 4 r^2) {-y1, x1})
        '''
        # (m,n); d(fun)/dx
        m = len(self.indexes)
        n = len(x)
        k = (n-1)//2

        p = np.reshape(x[:-1], (k, 2))
        j_p = np.reshape(np.eye(2*k, n), (k, 2, n))
        r = x[-1]
        j_r = np.zeros((n,))
        j_r[-1] = 1

        p = p[self.indexes]
        j_p = j_p[self.indexes]
        dp = (p[:, 1] - p[:, 0])/2
        j_dp = (j_p[:, 1]-j_p[:, 0])/2
        det = np.cross(p[:, 0], p[:, 1], axisa=1, axisb=1)
        j_det = \
            p[:, 1, 1, np.newaxis]*j_p[:, 0, 0] + \
            p[:, 0, 0, np.newaxis]*j_p[:, 1, 1] - \
            p[:, 1, 0, np.newaxis]*j_p[:, 0, 1] - \
            p[:, 0, 1, np.newaxis]*j_p[:, 1, 0]
        dp_sq = squared_norm(dp, axis=1)
        j_sq = np.sum(2 * dp[..., np.newaxis] * j_dp, axis=1)
        dot = np.sum(p[:, 0]*p[:, 1], axis=1)
        j_dot = np.sum(p[:, 1, :, np.newaxis]*j_p[:, 0] +
                       p[:, 0, :, np.newaxis]*j_p[:, 1], axis=1)

        return 2*det[..., np.newaxis]*j_det*(r**2 - dp_sq)[..., np.newaxis] + \
            np.square(det)[..., np.newaxis]*(2*r[..., np.newaxis]*j_r - j_sq) - \
            (2*(R**2 - r**2 - dot)[..., np.newaxis]*(-2*r[..., np.newaxis]*j_r-j_dot)*dp_sq[..., np.newaxis] +
             np.square(R**2 - r**2 - dot)[..., np.newaxis]*j_sq)

    def hessian(self, x, v):
        # (n,n)
        pass