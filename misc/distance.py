import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1., 1., 1001)
XY = np.dstack(np.meshgrid(x, x))


# vec3 r = max(abs(p - vec3(i, j, k)) - 0.25, 0.);
r = np.maximum(abs(XY) - 0.25, 0.)
# float di = length(r) - 0.25;
d = np.sqrt(np.sum(r ** 2, axis=2)) - 0.25

plt.contour(x, x, d, [-0.25, 0])
plt.axis('equal')

xy = np.random.random(2) * 2. - 1.
r = np.maximum(abs(xy) - 0.25, 0.)
d = np.sqrt(np.sum(r ** 2)) - 0.25
