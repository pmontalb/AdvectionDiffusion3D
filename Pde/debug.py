
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib
import numpy as np

if __name__ == "__main__":
    solution = np.zeros((16, 16, 16, 600))
    with open("/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/sol.txt", "r") as f:
        lines = f.readlines()

        counter = 0
        for line in lines:
            line = line.replace("\n", "")
            if "solution" not in line:
                continue
            if line.strip() == "":
                continue
            tokens = line.split("np.array([")[1].split("])")[0].split(",")
            solution[:, :, :, counter] = np.array([float(x) for x in tokens if x.strip() != ""]).reshape(solution.shape[:-1])
            solution[:, :, :, counter] = solution[:, :, :, counter].transpose()
            counter += 1

    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    X, Y = np.meshgrid(x, y)
    def update_plot(frame_number, z, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, z[:,:,2,frame_number], cmap="coolwarm",)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_surface(X, Y, solution[:,:,2,0], cmap="coolwarm", color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(0, 0.21)
    ani = animation.FuncAnimation(fig, update_plot, counter, fargs=(solution, plot), interval=1000/200)
    plt.show()
