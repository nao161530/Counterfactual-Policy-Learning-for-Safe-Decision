import numpy as np

def circle_points(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


def I_fun(x):
    res = []
    for i in range(len(x)):
        if x[i] > 0:
            res.append(1)
        else:
            res.append(0)
    res = np.reshape(res,(x.shape[0],1))
    return res