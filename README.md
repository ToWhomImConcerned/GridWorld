# Q-Learning Visualizer

![demo](Heatmap_Animation.gif)

An interactive reinforcement learning visualizer built from scratch in Python.
Draw a custom maze, then watch a Q-learning agent learn to solve it in real time
through a live heatmap that shows the agent's knowledge as it builds.

## Features
- Draw custom mazes with left click
- Place the goal anywhere with right click
- Live Q-value heatmap updates every step during training
- Control training speed in real time with W/S
- Trained agent demo runs after training completes

## How it works
The agent learns using Q-learning — a reinforcement learning algorithm that builds
a table of values for every state-action pair. Early in training the agent explores
randomly. Over time it exploits what it has learned, converging on the optimal path
through any maze you draw. The heatmap visualizes this process — warm colors mean
the agent thinks a cell is valuable, cold colors mean the opposite.

## Run it yourself
1. Clone the repo
2. Install dependencies: `pip install pygame numpy`
3. Run: `python live_visualizer.py`
4. Draw your maze, then press Space to train

## Built with
- Python
- Pygame
- NumPy