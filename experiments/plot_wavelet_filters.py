import argparse
import numpy as np
import matplotlib.pyplot as plt

from wavelets.numpy.Meyer import MeyerWavelet

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Wavelet Filters")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    # parser.add_argument("--rank", type=int, default=10)
    # parser.add_argument("--max_iter", type=int, default=100)
    # parser.add_argument("--matrix_size", type=int, nargs=2, default=[1000, 1000])
    # parser.add_argument("--missing_fraction", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    n_rows = 4

    fig, ax = plt.subplots(n_rows, 2, figsize=(12, 8))

    n_samples = 256
    # n_scales = np.floor(np.log2(n_samples)) - 2
    n_scales = 3

    print(f'Number of Scales: {n_scales}')

    wavelet = MeyerWavelet(
        n_scales=n_scales, 
        n_samples=n_samples
    )

    colors = plt.cm.seismic(np.linspace(0, 1, n_rows))

    for i, c in enumerate(colors):

        w_t = np.fft.irfft(wavelet.W[i], n=n_samples)
        w_t = np.fft.ifftshift(w_t)

        t = np.linspace(-1, 1, n_samples)

        ax[i, 0].plot(wavelet.w, wavelet.W[i], color=c, linewidth=2)
        ax[i, 1].plot(t, w_t, color=c, linewidth=2)

        ax[i, 0].set_xlabel('Freq [Hz]')
        ax[i, 1].set_xlabel('Time [Hz]')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()