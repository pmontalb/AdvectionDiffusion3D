import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib
import numpy as np


def single_run():
    n_iterations = 600
    solution = np.load("/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/sol.txt")

    Nx = solution.shape[0] // n_iterations
    actual = np.zeros((Nx, solution.shape[1], solution.shape[2], n_iterations))
    for n in range(n_iterations):
        actual[:, :, :, n] = solution[n * Nx:(n + 1) * Nx, :, :]
    solution = actual
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, solution.shape[1])
    X, Y = np.meshgrid(x, y)

    def update_plot(frame_number, z, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, z[:, :, 0, frame_number], cmap="coolwarm", )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_surface(X, Y, solution[:, :, 0, 0], cmap="coolwarm", color='0.75', rstride=1, cstride=1)]
    # ax.set_zlim(0, 0.21)
    ani = animation.FuncAnimation(fig, update_plot, solution.shape[-1], fargs=(solution, plot), interval=1000 / 200)
    plt.show()


def stability_analysis():
    n_iterations = 600
    solution_ee = np.load("/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/explicitEuler.txt")
    solution_ie = np.load("/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/implicitEuler.txt")
    solution_cn = np.load("/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/crankNicolson.txt")
    solution_lw = np.load("/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/laxWendroff.txt")
    solution_ad = np.load("/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/adi.txt")

    Nx = solution_ee.shape[0] // n_iterations
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, solution_ee.shape[1])
    X, Y = np.meshgrid(x, y)

    def reshape_solution(solution):
        actual = np.zeros((Nx, solution.shape[1], solution.shape[2], n_iterations))
        for n in range(n_iterations):
            actual[:, :, :, n] = solution[n * Nx:(n + 1) * Nx, :, :]
        return actual
    solution_ee = reshape_solution(solution_ee)
    solution_ie = reshape_solution(solution_ie)
    solution_cn = reshape_solution(solution_cn)
    solution_lw = reshape_solution(solution_lw)
    solution_ad = reshape_solution(solution_ad)

    fig = plt.figure()

    def update_plot(frame_number, z, plot, ax):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, z[:, :, 0, frame_number], cmap="coolwarm", )

    def worker(solution, axis_idx):
        ax = fig.add_subplot(2, 3, axis_idx, projection='3d')

        return ax, [ax.plot_surface(X, Y, solution[:, :, 0, 0], cmap="coolwarm", color='0.75', rstride=1, cstride=1)]
        # ax.set_zlim(0, 0.21)

    ax_ee, plot_ee = worker(solution_ee, 1)
    ani_ee = animation.FuncAnimation(fig, update_plot, solution_ee.shape[-1], fargs=(solution_ee, plot_ee, ax_ee),
                                  interval=1000 / 200)

    ax_ie, plot_ie = worker(solution_ie, 2)
    ani_ie = animation.FuncAnimation(fig, update_plot, solution_ie.shape[-1], fargs=(solution_ie, plot_ie, ax_ie),
                                  interval=1000 / 200)

    ax_cn, plot_cn = worker(solution_cn, 3)
    ani_cn = animation.FuncAnimation(fig, update_plot, solution_cn.shape[-1], fargs=(solution_cn, plot_cn, ax_cn),
                                  interval=1000 / 200)

    ax_lw, plot_lw = worker(solution_lw, 4)
    ani_lw = animation.FuncAnimation(fig, update_plot, solution_lw.shape[-1], fargs=(solution_lw, plot_lw, ax_lw),
                                     interval=1000 / 200)

    ax_ad, plot_ad = worker(solution_ad, 5)
    ani_ad = animation.FuncAnimation(fig, update_plot, solution_ad.shape[-1], fargs=(solution_ad, plot_ad, ax_ad),
                                  interval=1000 / 200)
    plt.show()


if __name__ == "__main__":
    stability_analysis()
