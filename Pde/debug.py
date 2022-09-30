
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib
import numpy as np

if __name__ == "__main__":
    n_iterations = 600
    solution = np.load("/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/sol.txt")

    Nx = solution.shape[0] // n_iterations
    actual =np.zeros((Nx,solution.shape[1], solution.shape[2], n_iterations))
    for n in range(n_iterations):
        actual[:, :, :, n] = solution[n*Nx:(n+1)*Nx, :, :]
    solution = actual
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, solution.shape[1])
    X, Y = np.meshgrid(x, y)
    def update_plot(frame_number, z, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, z[:,:,0,frame_number], cmap="coolwarm",)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_surface(X, Y, solution[:,:,0,0], cmap="coolwarm", color='0.75', rstride=1, cstride=1)]
    # ax.set_zlim(0, 0.21)
    ani = animation.FuncAnimation(fig, update_plot, solution.shape[-1], fargs=(solution, plot), interval=1000/200)
    plt.show()
