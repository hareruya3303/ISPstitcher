# ISP stitcher
Routines for stitching anime screenshots and then processing the images. The routines are primarily tested on a MacBookPro M4 Max but CUDA support is also tentatively tested in my old Linux machine from time to time. The principles of stitching are carried out using the generalized cross correlation of signals:

$$
\mathrm{gcc}(x, y, \Delta t) = \frac{\int x(t) y(\Delta t - t) \mathrm{d}t}{\sqrt{\int x^2(t) \mathrm{d}t \int y^2(t) \mathrm{d} t}}
$$

as well as variations based on either the presence of a moving average or the use of phase transformation. These routines are used to find correlation peaks that indicate the best stitching point(s) between two images, identified as the correlation peak.
