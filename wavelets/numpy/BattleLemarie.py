import numpy as np
from ..base import WaveletFrame

class BattleLemarieWavelet(WaveletFrame):

    def __init__(self, n_scales, n_samples, order):
        super().__init__()

        self.n_scales = n_scales
        self.n_samples = n_samples
        self.m = order

        assert self.n_scales > 1, "The minimum number of scales is 2"

        self.w = np.linspace(
            0,
            4 * np.pi / 3 * 2**(self.n_scales - 2),
            self.n_samples // 2 + 1
        )

        self.construct_filters()

    def phi(self, w):

        b_hat = self.B_hat(w)

        h = np.zeros(len(w))

        for k in range(1-self.m, self.m):
            h += self.B(np.array([k]), 2*self.m) * np.cos(k * w)

        return b_hat / np.sqrt(h)
    
    def psi(self, w):

        return np.abs(
            self.phi(w + 2 * np.pi) * self.phi(w / 2) / self.phi(w / 2 + np.pi)
        )

    def B_hat(self, w):

        return np.sinc(w / (2 * np.pi))**self.m
    
    def B(self, k, m):

        if m == 2:
            return self.N2(k)
        elif m == 4:
            return self.N4(k)
        elif m == 6:
            return self.N6(k)
        elif m == 8:
            return self.N8(k)
        elif m == 10:
            return self.N10(k)
        else:
            raise ValueError(f"B-spline of order {m} is not supported. Current implementation only supports Battle-Lemarie of order 1 to 4")

    def N2(self, x):

        y = np.zeros(len(x))

        interval = (x >= -1) & (x < 0)
        y += interval * np.polyval([1, 1], x)

        interval = (x >= 0) & (x < 1)
        y += interval * np.polyval([-1, 1], x)

        return y

    def N4(self, x):

        y = np.zeros(len(x))

        interval = (x >= -2) & (x < -1)
        y += interval * np.polyval([1/6, 1, 2, 4/3], x)

        interval = (x >= -1) & (x < 0)
        y += interval * np.polyval([-1/2, -1, 0, 2/3], x)

        interval = (x >= 0) & (x < 1)
        y += interval * np.polyval([1/2, -1, 0, 2/3], x)

        interval = (x >= 1) & (x < 2)
        y += interval * np.polyval([-1/6, 1, -2, 4/3], x)

        return y
    
    def N6(self, x):

        y = np.zeros(len(x))

        interval = (x >= -3) & (x < -2)
        y += interval * np.polyval([1/120, 1/8, 3/4, 9/4, 27/8, 81/40], x)

        interval = (x >= -2) & (x < -1)
        y += interval * np.polyval([-1/24, -3/8, -5/4, -7/4, -5/8, 17/40], x)

        interval = (x >= -1) & (x < 0)
        y += interval * np.polyval([1/12, 1/4, 0, -1/2, 0, 11/20], x)

        interval = (x >= 0) & (x < 1)
        y += interval * np.polyval([-1/12, 1/4, 0, -1/2, 0, 11/20], x)

        interval = (x >= 1) & (x < 2)
        y += interval * np.polyval([1/24, -3/8, 5/4, -7/4, 5/8, 17/40], x)

        interval = (x >= 2) & (x < 3)
        y += interval * np.polyval([-1/120, 1/8, -3/4, 9/4, -27/8, 81/40], x)

        return y
    
    def N8(self, x):

        y = np.zeros(len(x))

        interval = (x >= -4) & (x < -3)
        y += interval * np.polyval([1/5040, 1/180, 1/15, 4/9, 16/9, 64/15, 256/45, 1024/315, ], x)

        interval = (x >= -3) & (x < -2)
        y += interval * np.polyval([-1/720, -1/36, -7/30, -19/18, -49/18, -23/6, -217/90, -139/630, ], x)

        interval = (x >= -2) & (x < -1)
        y += interval * np.polyval([1/240, 1/20, 7/30, 1/2, 7/18, -1/10, 7/90, 103/210, ], x)

        interval = (x >= -1) & (x < 0)
        y += interval * np.polyval([-1/144, -1/36, 0, 1/9, 0/1, -1/3, 0, 151/315, ], x)

        interval = (x >= 0) & (x < 1)
        y += interval * np.polyval([1/144, -1/36, 0, 1/9, 0, -1/3, 0, 151/315, ], x)

        interval = (x >= 1) & (x < 2)
        y += interval * np.polyval([-1/240, 1/20, -7/30, 1/2, -7/18, -1/10, -7/90, 103/210, ], x)

        interval = (x >= 2) & (x < 3)
        y += interval * np.polyval([1/720, -1/36, 7/30, -19/18, 49/18, -23/6, 217/90, -139/630, ], x)

        interval = (x >= 3) & (x < 4)
        y += interval * np.polyval([-1/5040, 1/180, -1/15, 4/9, -16/9, 64/15, -256/45, 1024/315, ], x)

        return y
    
    def N10(self, x):

        y = np.zeros(len(x))

        interval = (x >= -5) & (x < -4)
        y += interval * np.polyval([1/362880, 1/8064, 5/2016, 25/864, 125/576, 625/576, 3125/864, 15625/2016, 78125/8064, 390625/72576, ], x)

        interval = (x >= -4) & (x < -3)
        y += interval * np.polyval([-1/40320, -1/1152, -3/224, -103/864, -43/64, -1423/576, -563/96, -2449/288, -5883/896, -133663/72576, ], x)

        interval = (x >= -3) & (x < -2)
        y += interval * np.polyval([1/10080, 5/2016, 3/112, 35/216, 19/32, 191/144, 83/48, 635/504, 339/448, 1553/2592, ], x)

        interval = (x >= -2) & (x < -1)
        y += interval * np.polyval([-1/4320, -1/288, -1/48, -13/216, -7/96, -1/144, -7/144, -19/72, -1/192, 7799/18144, ], x)

        interval = (x >= -1) & (x < 0)
        y += interval * np.polyval([1/2880, 1/576, 0/1, -5/432, 0/1, 19/288, 0/1, -35/144, 0/1, 15619/36288, ], x)

        interval = (x >= 0) & (x < 1)
        y += interval * np.polyval([-1/2880, 1/576, 0/1, -5/432, 0/1, 19/288, 0/1, -35/144, 0/1, 15619/36288, ], x)

        interval = (x >= 1) & (x < 2)
        y += interval * np.polyval([1/4320, -1/288, 1/48, -13/216, 7/96, -1/144, 7/144, -19/72, 1/192, 7799/18144, ], x)

        interval = (x >= 2) & (x < 3)
        y += interval * np.polyval([-1/10080, 5/2016, -3/112, 35/216, -19/32, 191/144, -83/48, 635/504, -339/448, 1553/2592, ], x)

        interval = (x >= 3) & (x < 4)
        y += interval * np.polyval([1/40320, -1/1152, 3/224, -103/864, 43/64, -1423/576, 563/96, -2449/288, 5883/896, -133663/72576, ], x)

        interval = (x >= 4) & (x < 5)
        y += interval * np.polyval([-1/362880, 1/8064, -5/2016, 25/864, -125/576, 625/576, -3125/864, 15625/2016, -78125/8064, 390625/72576, ], x)

        return y