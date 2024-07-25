from map import EnceladusEnvironment
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import numpy as np

def test_manual():
    env = EnceladusEnvironment()

    figure, axes = plt.subplots(figsize=(6, 6))

    frames = []
    fps = 10

    n_steps = 100
    for step in range(n_steps):
        print('Step:', step+1)
        observations, reward, done, _, _ = env.step(np.random.randint(0,7))
        print('Reward:', reward)
        print('Done?:', done, '\n')

        new_frame = [axes.imshow(env.surface_grid.transpose(), cmap=env.cmap),
                    axes.scatter(env.start_x, env.start_y, color='springgreen', label='Start', marker='s'),
                    axes.scatter(env.end_x, env.end_y, color='red', label='End', marker='s')]
        
        frames.append(new_frame)

        if done:
            print('Done')
            break

    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    rover_animation = animation.ArtistAnimation(figure, frames, interval=int(1000/fps), blit=True, repeat_delay=1000)

    file_path = 'visuals/test/rover_animation_test.gif'
    file_version = 1

    while os.path.isfile(file_path) is True:
        if file_version == 1:
            file_path = file_path.split('.')[0] + f'-{file_version}.gif'
        else:
            file_path = file_path.replace(f'-{file_version-1}.gif', f'-{file_version}.gif')
        file_version += 1

    rover_animation.save(file_path, dpi=150)

if __name__ == "__main__":
    test_manual()