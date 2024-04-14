import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env = gym.make('CartPole-v1', render_mode='rgb_array')

observation = env.reset()
rgb_buffer = []
for i in range(1000):
  rgb_data = env.render()
  rgb_buffer.append(rgb_data)

  action = env.action_space.sample()
  print("action:",action)
  observation, reward, info, done, _ = env.step(action)
  print("obs:",observation)
  print("reward:",reward)
  print("info:",info)
  print("done:",done)
  print("---------")

  if done:
    break

fig = plt.figure()
ax = fig.add_subplot(111)

def update(rgb_frame):
    ax.cla()
    ax.imshow(rgb_frame)
    
ani = animation.FuncAnimation(fig, update, frames=rgb_buffer, interval=50)
plt.show()

env.close()
