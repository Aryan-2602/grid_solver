from src.gridworld import GridWorld
from src.q_learning import QLearningAgent
import matplotlib.pyplot as plt
import imageio

grid_size = 20
env = GridWorld(size=grid_size, goals=[(0, grid_size - 1), (grid_size - 1, 0)])
agent = QLearningAgent(n_states=grid_size * grid_size, n_actions=4)


episodes = 500
rewards_per_episode = []
episode_lengths = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        steps += 1
    rewards_per_episode.append(total_reward)
    episode_lengths.append(steps)

print("Training complete! Final Q-table:")
print(agent.q_table)

print("\nLearned policy:")
actions = ['↑', '↓', '←', '→']
for i in range(env.size):
    row = ''
    for j in range(env.size):
        s = i * env.size + j
        row += actions[agent.q_table[s].argmax()] + ' '
    print(row)

# 📊 Plot reward and episode length
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.title("Episode Length per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid()

plt.tight_layout()
plt.savefig("assets/reward_plot.png")
plt.show()

# 🎥 Save agent run as GIF
frames = []
state = env.reset()
done = False
while not done:
    frames.append(env.render_image())
    action = agent.choose_action(state)
    state, _, done = env.step(action)

imageio.mimsave("assets/agent_run.gif", frames, fps=2)
print("🎥 Saved: assets/agent_run.gif")

