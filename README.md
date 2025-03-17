# ISP stitcher
Routines for stitching anime screenshots and then processing the images. The routines are primarily tested on a MacBookPro M4 Max but CUDA support is also tentatively tested in my old Linux machine from time to time. The principles of stitching are carried out using the generalized cross correlation (GCC) of signals:

$$
\mathrm{gcc}(x, y, \Delta t) = \frac{\int x(t) y(\Delta t - t) \mathrm{d}t}{\sqrt{\int x^2(t) \mathrm{d}t \int y^2(t) \mathrm{d} t}}
$$

as well as variations based on either the presence of a moving average or the use of phase transformation. These routines are used to find correlation peaks that indicate the best stitching point(s) between two images, identified as the correlation peak.

## Standard GCC

The standard generalized cross correlation algorithm employed here uses the FFT to turn the time complexity from $N^2$ to $N\mathrm{log}(N)$. It is the simplest and least resource-heavy method available, though they all have the same time complexity. It is also the least precise, but for most shots it is enough.

## Moving average GCC

This formulation subtracts the average of the block from the signals, which essentially makes it so brighter colors (higher RGB values) don't pull the average up. Considering we zero-pad on the leading dimension, the calculation of these means also adds up more $N\mathrm{log}(N)$ steps. It is negligibly more costly and performs better than the GCC.

## GCC PHAT

The GCC **PHAT** (phase transformed) involves taking only the phase of the frequency domain info and inverting the FFT, resulting in by far the most extreme criterion. Because the act of normalizing the signals is not linear, the operation dimensions can only be reduced in the frequency domain, meaning even though it is the most performant and has no need to calculate means, the GCC-PHAT is the costliest.

# Disclaimer

This is not a full-fledged package, it is mostly a playground where I do my hobby that I thought could be useful for other people who wish there was a faster and precise way to stitch anime panning shots. Now that correlations have been implemented the next step should be implementing simple blending (feathering).
