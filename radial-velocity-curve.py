import numpy as np
import matplotlib.pyplot as plt

v_star = np.load('star-velocity-array.npy')
t = np.load('star-time-array.npy')

v_max = np.max(v_star)
v_star_noise = v_star[:,0] + np.random.normal(scale=1/5*v_max, size=len(t))

plt.figure()
plt.title('x-component of star velocity')
plt.plot(t, v_star[:,0])
plt.xlabel('time')
plt.ylabel('AU/yr')
plt.savefig('star-velocity.png')

plt.figure()
plt.title('x-component of star velocity with noise')
plt.plot(t, v_star_noise)
plt.xlabel('time')
plt.ylabel('AU/yr')
plt.savefig('star-velocity-noise.png')

plt.figure()
plt.title('x-component of star velocity with and without noise')
plt.plot(t, v_star_noise)
plt.plot(t, v_star[:,0])
plt.xlabel('time')
plt.ylabel('AU/yr')
plt.savefig('star-velocity-noise-and-clean.png')
plt.show()
