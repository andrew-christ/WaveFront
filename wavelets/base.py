from abc import ABC, abstractmethod
import numpy as np

class WaveletFrame(ABC):

    def transform(self, s):
        # Compute the Fourier Transform along the last dimension
        s_fft       = np.fft.rfft(s, axis=-1)

        # Expand the dimensions in the second to last position
        # so we can multiple the signal with the wavelet filters
        s_scales    = np.expand_dims(s_fft, axis=-2)

        # Multiple the Frequency domain representation of the signal
        # by the wavelet filters and then compute the Inverse Fourier Transform
        return np.fft.irfft(s_scales * self.W, n=self.n_samples, axis=-1)

    def inverse_transform(self, s):

        # Compute the Fourier Transform along the last dimension
        s_fft       = np.fft.rfft(s, axis=-1)

        # Reconstruct the signals by multiply the Frequency Domain 
        # representation of the Wavelet filters with the 'n+1' signal 
        # scales (low-pass included)
        s_recon     = np.sum(s_fft * self.W, axis=-2)
        
        # Compute the Inverse Fourier Transform of the reconstructed signal
        return np.fft.irfft(s_recon, n=self.n_samples, axis=-1)
    
    def construct_filters(self):

        self.n_scales = int(self.n_scales)

        self.W = np.zeros(
            (self.n_scales + 1, len(self.w))
        )

        # Scaling function (Low-frequency)
        self.W[0, :] = self.phi(self.w)

        for j in range(self.n_scales):

            # Wavelet function at jth scale
            self.W[j + 1, :] = self.psi(2**(-j) * self.w)

    @abstractmethod
    def psi(self, w):
        """Define the mother wavelet function."""
        pass

    @abstractmethod
    def phi(self, w):
        """Define the scaling function."""
        pass