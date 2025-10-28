import numpy as np
from ..base import WaveletFrame

class MeyerWavelet(WaveletFrame):

    def __init__(self, n_scales, n_samples):
        super().__init__()

        self.n_scales = n_scales
        self.n_samples = n_samples

        self.w = np.linspace(
            0,
            8 * np.pi / 3 * 2**(self.n_scales-2),
            self.n_samples // 2 + 1
        )

        self.construct_filters()

    def v(self, x):

        wave_int = (x >= 0) & (x < 1)
        ones_int = (x >= 1)
        zeros_int = (x < 0)

        y = wave_int * np.sin(np.pi * x / 2)**2
        y += ones_int * 1.0
        y += zeros_int * 0.0

        return y
    
    """Define the scaling function."""
    def phi(self, w):

        w = np.abs(w)

        wave_int = (w >= 2 * np.pi / 3) & (w < 4 * np.pi / 3)
        ones_int = (w < 2 * np.pi / 3)

        phi = wave_int * np.cos(np.pi / 2 * self.v(3 * w / (2 * np.pi) - 1))
        phi += ones_int * 1.0

        return phi
    
    """Define the mother wavelet function."""
    def psi(self, w):

        w = np.abs(w)

        sin_int = (w >= 2 * np.pi / 3) & (w < 4 * np.pi / 3)
        cos_int = (w >= 4 * np.pi / 3) & (w < 8 * np.pi / 3)

        psi = sin_int * np.sin(np.pi / 2 * self.v(3 * w / (2 * np.pi) - 1))
        psi += cos_int * np.cos(np.pi / 2 * self.v(3 * w / (4 * np.pi) - 1))

        return psi