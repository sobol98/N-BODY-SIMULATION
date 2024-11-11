'''
File path: 
.../N-body-simulation/openmp

To run:
python3 plot.py

'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation



# ------------------- settings -------------------
axis_size = 5e6
interval_size =1  # 1, 10, 100, 1000, or similiar -> smaller is faster

filename='cuda/results.txt'
# filename='openmp/results.txt'

# ------------------- main -------------------
data=np.loadtxt(filename)

bodies = np.unique(data[:, 0]).astype(int)
timesteps = int(data.shape[0] / len(bodies))
positions = {body: data[data[:, 0] == body, 2:] for body in bodies}

# print(bodies)
# print(timesteps)
# print(positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


all_positions = np.vstack(list(positions.values()))
min_x, max_x = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
min_y, max_y = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
min_z, max_z = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])

ax.set_xlim(-axis_size, axis_size)
ax.set_ylim(-axis_size, axis_size)
ax.set_zlim(-axis_size, axis_size)

# Initialize points for each body
points = {body: ax.plot([], [], [], 'o', label=f'Body {body}')[0] for body in bodies}
trajectories = {body: ax.plot([], [], [], '-')[0] for body in bodies}



def update(frame):
    for body in bodies:
        pos = positions[body][frame]  #
        points[body].set_data([pos[0]], [pos[1]])
        points[body].set_3d_properties(float(pos[2]))
        
        trajectory_x, trajectory_y, trajectory_z = trajectories[body].get_data_3d()
        trajectory_x = np.append(trajectory_x, pos[0])
        trajectory_y = np.append(trajectory_y, pos[1])
        trajectory_z = np.append(trajectory_z, pos[2])

        trajectories[body].set_data(trajectory_x, trajectory_y)
        trajectories[body].set_3d_properties(trajectory_z)
 
    ax.legend()
    return list(points.values()) +list(trajectories.values())


def init():
    for body in bodies:
        points[body].set_data([], [])
        points[body].set_3d_properties([])
        
        trajectories[body].set_data([], [])
        trajectories[body].set_3d_properties([])
    return list(points.values()) + list(trajectories.values())

# Create the animation
ani = FuncAnimation(fig, update, frames=timesteps, init_func=init, interval=interval_size,blit=True)

# Show the animation
plt.show()