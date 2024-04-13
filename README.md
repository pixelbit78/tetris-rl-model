# Tetris Reinforcement Learning Model

## Overview

This repository hosts a Python-based Tetris game, enhanced with a reinforcement learning (RL) model designed to master the game. The project aims to demonstrate the application of RL principles to video games, showcasing how an AI can develop strategies to maximize its score in Tetris.

## Features

- **Tetris Python Implementation**: Fully functional Tetris game coded in Python, using pygame.
- **Reinforcement Learning Integration**: A robust RL model trained to optimize gameplay strategies.
- **Performance Metrics**: Tracking of AI performance and improvement over time.

## Getting Started

### Installation

To get started with this project, you'll need to have Python installed on your machine. It is recommended to use a virtual environment:

```bash
# Clone the repo
git clone https://github.com/pixelbit78/tetris-rl-model.git
cd tetris-rl-model

# Create a virtual environment
python -m venv env

# Install the dependencies
pip install -r requirements.txt

```

### Usage

To play Tetris manually or watch the AI play:
```bash
# To play the game yourself
python main.py

# To watch the AI play
python main.py --model model/checkpoint.pth

# To train the AI
python train.py
```
