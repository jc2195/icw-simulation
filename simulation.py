import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import time

# The start and end frequencies in hz for the range of absorbers in the model
# If total number of absorbers is 1, then absorber is tuned to hz_start
# If total number of absorbers is 2, then absorbers are tuned to hz_start and hz_end
# Otherwise the absorbers are evenly distributed between hz_start and hz_end (inclusive)
hz_start = 1 
hz_end = 5

# Mass of the building model in kg
building_mass = 3.98

# Spring stiffness of the building model in N/m
building_k = 2100

# Damping rate of building dashpot in Ns/m
building_l = 1.98

# Disturbance force applied to building in N
building_force = 0.25

# Damping rate of each absorber in Ns/m
absorber_l = 0.01

# The total mass of all absorbers. The mass of each absorber is the same (total mass / number of absorbers), as the frequency
# they are tuned to is controlled by the spring stiffness (which is dictated by the supplied start and end frequencies)
total_absorber_mass = 0.15

# The number of total absorbers in the model - the model will have this many degrees of freedom plus 1.
no_of_absorbers = 10

def MLKF(md, ld, kd, f):
    num = len(md)

    M = [[0 for i in range(num)] for j in range(num)]
    L = [[0 for i in range(num)] for j in range(num)]
    K = [[0 for i in range(num)] for j in range(num)]
    F = [0 for i in range(num)]

    F[0] = f

    for i in range(num):
        M[i][i] = md[i]

        for j in range(num):
            if i == 0:
                if j == 0:
                    L[i][j] = np.sum(ld)
                    K[i][j] = np.sum(kd)
                else:
                    L[i][j] = -ld[j]
                    K[i][j] = -kd[j]
            else:
                if j == 0:
                    L[i][j] = -ld[i]
                    K[i][j] = -kd[i]
                elif j == i:
                    L[i][j] = ld[j]
                    K[i][j] = kd[j]
                else:
                    L[i][j] = 0
                    K[i][j] = 0
    M = np.array(M)
    L = np.array(L)
    K = np.array(K)
    F = np.array(F)

    return M, L, K, F

def MLKF_1dof(m1, l1, k1, f1):

    """Return mass, damping, stiffness & force matrices for 1DOF system"""

    M = np.array([[m1]])
    L = np.array([[l1]])
    K = np.array([[k1]])
    F = np.array([f1])

    return M, L, K, F


def MLKF_2dof(m1, l1, k1, f1, m2, l2, k2, f2):

    """Return mass, damping, stiffness & force matrices for 2DOF system"""

    M = np.array([[m1, 0], [0, m2]])
    L = np.array([[l1+l2, -l2], [-l2, l2]])
    K = np.array([[k1+k2, -k2], [-k2, k2]])
    F = np.array([f1, f2])

    return M, L, K, F

def MLKF_3dof(m1, l1, k1, f1, m2, l2, k2, f2, m3, l3, k3, f3):

    """Return mass, damping, stiffness & force matrices for 3DOF system"""

    M = np.array([[m1, 0, 0], [0, m2, 0], [0, 0, m3]])
    L = np.array([[l1+l2, -l2, 0], [-l2, l2+l3, -l3], [0, -l3, l3]])
    K = np.array([[k1+k2, -k2, 0], [-k2, k2+k3, -k3], [0, -k3, k3]])
    F = np.array([f1, f2, f3])

    return M, L, K, F


def freq_response(w_list, M, L, K, F):

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )


def time_response(t_list, M, L, K, F):

    """Return time response of system"""

    mm = M.diagonal()

    def slope(t, y):
        xv = y.reshape((2, -1))
        a = (F - L@xv[1] - K@xv[0]) / mm
        s = np.concatenate((xv[1], a))
        return s

    solution = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(len(mm) * 2),
        method='Radau',
        t_eval=t_list
    )

    return solution.y[0:len(mm), :].T


def last_nonzero(arr, axis, invalid_val=-1):

    """Return index of last non-zero element of an array"""

    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def plot(hz, sec, M, L, K, F):

    """Plot frequency and time domain responses"""

    # Generate response data
    start_time = time.time()
    f_response = np.abs(freq_response(hz * 2*np.pi, M, L, K, F))
    t_response = time_response(sec, M, L, K, F)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Determine suitable legends

    f_legends = (
        'm{} peak {:.4g} metre at {:.4g} Hz'.format(
            i+1,
            f_response[m][i],
            hz[m]
        )
        for i, m in enumerate(np.argmax(f_response, axis=0))
    )

    equilib = np.abs(freq_response([0], M, L, K, F))[0]         # Zero Hz
    toobig = abs(100 * (t_response - equilib) / equilib) >= 2
    lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)

    t_legends = (
        'm{} settled to 2% beyond {:.4g} sec'.format(
            i+1,
            sec[lastbig[i]]
        )
        for i, _ in enumerate(t_response.T)
    )

    # Create plot

    fig, ax = plt.subplots(2, 1, figsize=(11.0, 7.7))

    ax[0].set_title('Frequency domain response')
    ax[0].set_xlabel('Frequency/hertz')
    ax[0].set_ylabel('Amplitude/metre')
    # ax[0].plot(hz, f_response)
    ax[0].legend(ax[0].plot(hz, f_response), f_legends)

    ax[1].set_title('Time domain response')
    ax[1].set_xlabel('Time/second')
    ax[1].set_ylabel('Displacement/metre')
    # ax[1].plot(sec, t_response)
    ax[1].legend(ax[1].plot(sec, t_response), t_legends)

    fig.tight_layout()
    plt.show()


def main():

    """Main program"""

    # Parse arguments

    ap = argparse.ArgumentParser('Plot response curves')

    ap.add_argument(
        '--hz', type=float, nargs=2, default=(0, 5),
        help='Frequency range [0 5]'
    )
    ap.add_argument(
        '--sec', type=float, default=30,
        help='Time limit [30]'
    )

    args = ap.parse_args()

    absorber_mass = (total_absorber_mass / no_of_absorbers)

    if no_of_absorbers != 1:
        hz_gap = ((hz_end - hz_start) / (no_of_absorbers - 1))

    if no_of_absorbers == 1:
        w_list = [(hz_start * 2 * np.pi)]
    elif no_of_absorbers == 2:
        w_list = [(hz_start * 2 * np.pi), (hz_end * 2 * np.pi)]
    else:
        w_list = [(hz_start * 2 * np.pi) + (hz_gap * 2 * np.pi * i) for i in range(no_of_absorbers)]

    l_list = [building_l] + [absorber_l for i in range(no_of_absorbers)]
    k_list = [building_k] + [absorber_mass * (i ** 2) for i in w_list]
    m_list = [building_mass] + [absorber_mass for i in range(no_of_absorbers)]

    # Generate matrices describing the system

    M, L, K, F = MLKF(m_list, l_list, k_list, building_force)
    # Generate frequency and time arrays

    hz = np.linspace(args.hz[0], args.hz[1], 10001)
    sec = np.linspace(0, args.sec, 10001)

    # Plot results

    plot(hz, sec, M, L, K, F)


if __name__ == '__main__':
    main()