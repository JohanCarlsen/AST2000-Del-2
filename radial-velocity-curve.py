'''
EGEN kode!
beklager at det ikke er noen kommentarer her, men jeg fikk ikke tid.
prøvde å få kontakt på forumet, men til ingen nytte.
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
from numba import njit
from time import time

v_star = np.load('star-velocity-array.npy')
t = np.load('star-time-array.npy')

v_max = np.max(v_star)
v_star_noise = v_star[:,0] + np.random.normal(scale=1/5*v_max, size=len(t))

other_group_velocity_curve = pickle.load(open('array_med_støy.dat', 'rb'))
peculiar_velocity = 8.875

N = len(other_group_velocity_curve)
t_other_group = np.linspace(0, N/1000, N)

t_0_max = 60
t_0_min = 40
dt = 20
t_0_list = np.linspace(t_0_min, t_0_max, dt)

P_max = 80
P_min = 40
dP = 20
P_list = np.linspace(P_min, P_max, dP)

v_r_max = 9.75 - peculiar_velocity
v_r_min = 8.75 - peculiar_velocity
dv = 20
v_r_list = np.linspace(v_r_min, v_r_max, dv)

matrix = np.array([v_r_list, P_list, t_0_list])

t1 = time()
@njit
def delta():
    v_r = 0.
    P = 0.
    t_0 = 0.
    delta = 0.
    v = other_group_velocity_curve
    t = t_other_group
    v_model_old = matrix[0,0] * np.cos((2 * np.pi / matrix[1,0]) * (t - matrix[2,0]))
    delta_old = np.sum((v_model_old - v)**2)
    for i in range(len(matrix[0,:])):
        for j in range(len(matrix[1,:])):
            for k in range(len(matrix[2,:])):
                v_model = matrix[0,i] * np.cos((2 * np.pi / matrix[1,j]) * (t - matrix[2,k]))
                delta_new = np.sum((v_model - v)**2)
                if delta_new < delta_old:
                    v_r = matrix[0,i]
                    P = matrix[1,j]
                    t_0 = matrix[2,k]
                    delta_old = delta_new
                else:
                    continue
    return v_r, P, t_0


v_r, P, t_0 = delta()

t2 = time()

print(f'v_r={v_r}, P={P}, t_0={t_0}')

print(int(t2-t1))

def v_model_best(v_r, P, t_0):
    t = t_other_group
    return v_r * np.cos((2 * np.pi / P) * (t - t_0)) + peculiar_velocity

plt.figure()
plt.title('Finding $v_r$')
plt.xlabel('Time in years')
plt.ylabel('Velocity in AU/yrs')
plt.plot(t_other_group, other_group_velocity_curve, 'k')
plt.plot(t_other_group, v_model_best(v_r, P, t_0), 'r')
plt.savefig('v-r-with-formula.png')

plt.figure()
plt.title('Finding $v_r$ with eye-measure')
plt.xlabel('Time in years')
plt.ylabel('Velocity in AU/yrs')
plt.plot(t_other_group, other_group_velocity_curve, 'k')
plt.plot(t_other_group, 0.375 * np.cos((2 * np.pi / 65) * (t_other_group - 50)) + peculiar_velocity, 'r')
plt.savefig('v-r-with-eye-measure.png')

plt.show()

'''
what we got from the formula:
v_r=0.875, P=50.526315789473685, t_0=60.0
'''

# plt.figure(figsize=(8,6))
# plt.title('x-component of star velocity')
# plt.plot(t, v_star[:,0])
# plt.xlabel('years on home planet')
# plt.ylabel('AU/yr')
# plt.savefig('star-velocity.png')
#
# plt.figure(figsize=(8,6))
# plt.title('x-component of star velocity with noise')
# plt.plot(t, v_star_noise)
# plt.xlabel('years on home planet')
# plt.ylabel('AU/yr')
# plt.savefig('star-velocity-noise.png')
#
# plt.figure(figsize=(8,6))
# plt.title('x-component of star velocity with and without noise')
# plt.plot(t, v_star_noise)
# plt.plot(t, v_star[:,0])
# plt.xlabel('years on home planet')
# plt.ylabel('AU/yr')
# plt.savefig('star-velocity-noise-and-clean.png')
# plt.show()
