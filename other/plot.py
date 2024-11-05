import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

n_body = 50

# Read the data
with open("output.txt", "r") as file:
    lines = file.readlines()

# Prepare data structures
positions = [[] for _ in range(n_body)]
current_body = 0
timesteps = []

# Parse the data
for line in lines:
    if line.strip():  # If the line is not empty
        positions[current_body].append([float(val) for val in line.split()])
        if current_body == n_body - 1:  # Last body for the timestep
            timesteps.append([pos[-1] for pos in positions])
        current_body = (current_body + 1) % n_body
    else:  # Empty line indicates new timestep
        current_body = 0

# Add the last timestep if not empty
if positions[0]:
    timesteps.append([pos[-1] for pos in positions])

# Set up the figure and axis for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Initializing scatter plots and trails for each body
scatters = []
trails = []

for i in range(n_body):
    x, y, z = zip(*positions[i])
#    if(i==0){
#    	markersize0=8
#    if(i==1)
#    	markersize1=4
    	
    scatter, = ax.plot([x[0]], [y[0]], [z[0]], 'o', markersize=4)  # Initial position
    trail, = ax.plot(x, y, z, '-', linewidth=1, alpha=0.5)  # Trail
    
    scatters.append(scatter)
    trails.append(trail)

# Update function for animation
def update(frame):
    for scatter, trail, body_index in zip(scatters, trails, range(n_body)):
        x, y, z = zip(*positions[body_index][:frame+1])
        scatter.set_data(x[-1], y[-1])
        scatter.set_3d_properties(z[-1], 'z')
        trail.set_data(x, y)
        trail.set_3d_properties(z, 'z')
    return scatters + trails

# Create animation
ani = FuncAnimation(fig, update, frames=len(timesteps), interval=10, blit=False)

# Set labels and title
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')
ax.set_title('Animated Trajectories of Bodies')

# Improve visualization
ax.grid(True)
ax.set_xlim([min(pos[0][0] for pos in positions), 5*max(pos[0][0] for pos in positions)])
ax.set_ylim([min(pos[0][1] for pos in positions), 5*max(pos[0][1] for pos in positions)])
ax.set_zlim([min(pos[0][2] for pos in positions), 5*max(pos[0][2] for pos in positions)])

#ax.set_xlim([-100,100])
#ax.set_ylim([-100,100])
#ax.set_zlim([-100,100])

plt.show()
