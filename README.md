# WaveFront

## A Historical Introduction to Time-Frequency Representations

In applications such as computer vision, it is difficult to analyze local and global information from individual pixel value of an image. Examining pixels in isolation fails to capture the neighborhood structure around them such as edges, contours, and textures. Moreover, indovidual pixels are highly susceptible to corruption from sources like camera noise and quantization. These challenges motivated the development of transform-based representations that provide a more holistic view of the image's components


In many introductory Digital Signal Processing courses, the Discrete Fourier Transform (DFT) is introduced as an alternative representation for analyzing periodic signals. The DFT is widely used because it decomposes a signal into a sum of sinusoidal waves. However, while effective for stationary or periodic signals, it performs poorly for signals with transient features, oftern requiring many Fourier coeffcients to represent discontinuities.

Beginning with the work of Dennis Gabir in the 1940s, physicists, mathematicians, and engineers have proposed alternative transforms that could simultaneously capture time and frequency characteristics. These functions, which trade off between time and frequency resolution, are know as _wavelets_. A wavelet is a localized, oscillatory function that rapidly decays to zero outside a short interval. One of the earliest examples is the _Gabor wavelet_, defined as

$$
\begin{equation}
    f(t) = e^{\frac{(t - t_0)^2}{s^2}} e^{-2\pi \imath \zeta (t - t_0)}
\end{equation}
$$

which represents a complex exponential of frequency $\zeta$ modulated by a Gaussian window. 

The 1980s saw an explosion of research that established the modern field of _multiscale signal processing_, leading to the development of several important wavelet families, such as as the Meyer, Battle-Lemarié, and Daubechies wavelets.

In this repository, we explore and implement some of these foundational wavelet transforms, demonstrating how they can be combined with convex optimization to perform tasks such as image denoising.