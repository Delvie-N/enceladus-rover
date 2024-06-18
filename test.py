from map import EnceladusEnvironment
import matplotlib.pyplot as plt
from matplotlib import animation

def test_manual():
    env = EnceladusEnvironment()

    plt.figure(figsize=(6, 6))
    plt.title('Exploring Enceladus')
    plt.imshow(env.surface_grid.transpose(), cmap=env.cmap)
    plt.scatter(env.start_x, env.start_y, color='springgreen', label='Start', marker='s')
    plt.scatter(env.end_x, env.end_y, color='red', label='End', marker='s')
    plt.legend()
    plt.show()

    env.step(4)

    plt.figure(figsize=(6, 6))
    plt.title('Exploring Enceladus')
    plt.imshow(env.surface_grid.transpose(), cmap=env.cmap)
    plt.scatter(env.start_x, env.start_y, color='springgreen', label='Start', marker='s')
    plt.scatter(env.end_x, env.end_y, color='red', label='End', marker='s')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_manual()