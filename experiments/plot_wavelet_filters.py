import argparse
import numpy as np
import matplotlib.pyplot as plt

from wavelets.numpy.Meyer import MeyerWavelet

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Wavelet Filters")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--wavelet", choices=["meyer", "battle-lemarie"], default="meyer")
    parser.add_argument("--n_scales", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=256)
    return parser.parse_args()


def main():

    args = parse_args()

    n_samples   = args.n_samples
    n_scales    = args.n_scales
    n_rows      = n_scales + 1

    print(f'Number of Scales: {n_scales}')

    wavelet = MeyerWavelet(
        n_scales=n_scales, 
        n_samples=n_samples
    )

    colors = plt.cm.seismic(np.linspace(0, 1, n_rows))

    fig, ax = plt.subplots(n_rows, 2, figsize=(12, 9))

    for i, c in enumerate(colors):

        w_t = np.fft.irfft(wavelet.W[i], n=n_samples)
        w_t = np.fft.ifftshift(w_t)

        t = np.linspace(-1, 1, n_samples)

        ax[i, 0].plot(wavelet.w, wavelet.W[i], color=c, linewidth=2)
        ax[i, 1].plot(t, w_t, color=c, linewidth=2)

        ax[i, 0].set_xlabel('Freq [Hz]')
        ax[i, 1].set_xlabel('Time [Hz]')

    ax[0, 0].set_title('Wavelet Filter in Frequency Domain')
    ax[0, 1].set_title('Wavelet Filter in Time Domain')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()