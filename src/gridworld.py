import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import io
from PIL import Image

class GridWorld:
    def __init__(self, size=5, goals=[(4, 0), (0, 4)], obstacles=None):
        self.size = size
        self.goals = goals
        self.obstacles = obstacles or [(1, 2), (2, 2), (3, 1)]
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.current_goal_idx = 0
        return self._get_state()

    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action):
        # 80% chance of intended action
        if np.random.rand() > 0.8:
            action = np.random.choice([a for a in range(4) if a != action])

        x, y = self.agent_pos
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.size - 1: x += 1
        elif action == 2 and y > 0: y -= 1
        elif action == 3 and y < self.size - 1: y += 1

        if (x, y) not in self.obstacles:
            self.agent_pos = [x, y]

        reward = -1
        done = False
        if tuple(self.agent_pos) == self.goals[self.current_goal_idx]:
            reward = 10
            self.current_goal_idx += 1
            if self.current_goal_idx >= len(self.goals):
                done = True

        return self._get_state(), reward, done

    def render(self):
        grid = np.full((self.size, self.size), '.', dtype=str)
        for ox, oy in self.obstacles:
            grid[ox][oy] = '#'
        gx, gy = self.goals[min(self.current_goal_idx, len(self.goals) - 1)]
        grid[gx][gy] = 'G'
        x, y = self.agent_pos
        grid[x][y] = 'A'
        for row in grid:
            print(' '.join(row))
        print()

    def render_image(self):
        cmap = colors.ListedColormap(['white', 'black', 'green', 'blue', 'red'])
        grid = np.zeros((self.size, self.size))
        for (ox, oy) in self.obstacles:
            grid[ox][oy] = 1
        gx, gy = self.goals[min(self.current_goal_idx, len(self.goals)-1)]
        grid[gx][gy] = 2
        ax, ay = self.agent_pos
        grid[ax][ay] = 4

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(grid, cmap=cmap)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        plt.close(fig)
        return np.array(image)

