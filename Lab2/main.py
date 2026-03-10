import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline


def f(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.sin(r)


def make_grid(nx, ny, x_min, x_max, y_min, y_max):
    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    return x, y


def make_pseudo_regular_grid(nx, ny, x_min, x_max, y_min, y_max, noise=0.28):
    rng = np.random.default_rng()
    x_base = np.linspace(x_min, x_max, nx + 1)
    y_base = np.linspace(y_min, y_max, ny + 1)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x = x_base + rng.uniform(-noise * dx, noise * dx, nx + 1)
    y = y_base + rng.uniform(-noise * dy, noise * dy, ny + 1)
    x = np.sort(x)
    y = np.sort(y)
    x[0], x[-1] = x_min, x_max
    y[0], y[-1] = y_min, y_max
    return x, y


def get_refined_coords(x, y):
    nx, ny = len(x) - 1, len(y) - 1
    x_ref = np.empty(2 * nx + 1)
    y_ref = np.empty(2 * ny + 1)
    for i in range(nx):
        x_ref[2 * i] = x[i]
        x_ref[2 * i + 1] = 0.5 * (x[i] + x[i + 1])
    x_ref[2 * nx] = x[nx]
    for j in range(ny):
        y_ref[2 * j] = y[j]
        y_ref[2 * j + 1] = 0.5 * (y[j] + y[j + 1])
    y_ref[2 * ny] = y[ny]
    return x_ref, y_ref


def refine_grid_interp(x, y, z, interp):
    nx, ny = len(x) - 1, len(y) - 1
    centers_x = []
    centers_y = []
    z_centers = []
    for i in range(nx):
        for j in range(ny):
            xc = 0.5 * (x[i] + x[i + 1])
            yc = 0.5 * (y[j] + y[j + 1])
            zc = interp.ev(xc, yc)
            centers_x.append(xc)
            centers_y.append(yc)
            z_centers.append(zc)
    return np.array(centers_x), np.array(centers_y), np.array(z_centers), interp


def build_refined_grid(x, y, z, interp, levels=1):
    nx, ny = len(x) - 1, len(y) - 1
    x_ref, y_ref = get_refined_coords(x, y)
    nrx, nry = 2 * nx + 1, 2 * ny + 1
    z_ref = np.empty((nrx, nry))
    for i in range(nrx):
        for j in range(nry):
            if i % 2 == 0 and j % 2 == 0:
                z_ref[i, j] = z[i // 2, j // 2]
            else:
                z_ref[i, j] = interp.ev(x_ref[i], y_ref[j])
    if levels > 1:
        _, _, _, interp_next = refine_grid_interp(x_ref, y_ref, z_ref, RectBivariateSpline(x_ref, y_ref, z_ref, kx=3, ky=3, s=0))
        return build_refined_grid(x_ref, y_ref, z_ref, interp_next, levels=levels - 1)
    return x_ref, y_ref, z_ref


def main():
    nx, ny = 9, 9
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    x, y = make_pseudo_regular_grid(nx, ny, x_min, x_max, y_min, y_max)
    X, Y = np.meshgrid(x, y, indexing="ij")
    z = f(X, Y)
    nodal_err = 1e-6
    rng = np.random.default_rng(42)
    z_noisy = z + rng.uniform(-nodal_err, nodal_err, z.shape)

    X_nodes, Y_nodes = np.meshgrid(x, y, indexing="ij")
    centers_x = np.array([0.5 * (x[i] + x[i + 1]) for i in range(nx) for j in range(ny)])
    centers_y = np.array([0.5 * (y[j] + y[j + 1]) for i in range(nx) for j in range(ny)])
    x_all = np.concatenate([X_nodes.ravel(), centers_x])
    y_all = np.concatenate([Y_nodes.ravel(), centers_y])
    z_true_nodes = f(x_all, y_all)
    z_true_centers = f(centers_x, centers_y)

    interp = RectBivariateSpline(x, y, z_noisy, kx=3, ky=3, s=0)
    z_interp_all = interp.ev(x_all, y_all)
    z_interp_centers = interp.ev(centers_x, centers_y)
    mean_with = np.mean(np.abs(z_interp_all - z_true_nodes))
    diff_centers = np.abs(z_interp_centers - z_true_centers)
    mean_without = np.mean(diff_centers)
    min_without = np.min(diff_centers)
    max_without = np.max(diff_centers)
    #print("с учётом исходных:", mean_with)
    #print("без исходных:", mean_without)
    print("минимальная абсолютная погрешность", min_without)
    print("максимальная абсолютная погрешность", max_without)
    centers_x, centers_y, z_centers, interp = refine_grid_interp(x, y, z_noisy, interp)
    x_ref, y_ref, z_ref = build_refined_grid(x, y, z_noisy, interp)

    fig, ax = plt.subplots()
    x_edges = np.concatenate([x_ref, [2 * x_ref[-1] - x_ref[-2]]])
    y_edges = np.concatenate([y_ref, [2 * y_ref[-1] - y_ref[-2]]])
    im = ax.pcolormesh(x_edges, y_edges, z_ref, shading="flat")
    plt.colorbar(im, ax=ax, label="z")
    X_nodes, Y_nodes = np.meshgrid(x, y, indexing="ij")
    ax.scatter(X_nodes.ravel(), Y_nodes.ravel(), c="white", s=40, edgecolors="black", linewidths=0.8, label="узлы сетки", zorder=2)
    ax.scatter(centers_x, centers_y, c="red", marker="x", s=30, label="центры ячеек", zorder=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig("result.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
