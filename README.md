# Project: DQN with Transformer for CartPole

**Undergraduate Project (2020)**

This repository contains an implementation of a Deep Q-Network (DQN) augmented with a Transformer block, originally developed as an undergraduate project in 2020. The agent learns to solve the CartPole-v1 environment using a combination of self-attention and feed-forward neural network layers.

## Features

* **Transformer Block**: Integrates multi-head self-attention and feed-forward layers for richer state representation.
* **Experience Replay**: Stores transitions and samples mini-batches for stable learning.
* **Target Network**: Uses a separate target model to stabilize Q-value targets.
* **Epsilon-Greedy Exploration**: Balances exploration and exploitation with decaying epsilon.

## Requirements

* Python 3.8+
* `tensorflow` 2.x
* `gymnasium` (Farama’s new Gym): replaces classic `gym`

To install dependencies:

```bash
pip install tensorflow gymnasium
```

> **Note:** We now recommend using [Gymnasium](https://gymnasium.farama.org/) instead of the deprecated OpenAI Gym. All environment imports in the code have been updated accordingly.

## Installation

1. Clone the repository:

   ```bash
   ```

git clone [https://github.com/yourusername/dqn-transformer.git](https://github.com/yourusername/dqn-transformer.git)
cd dqn-transformer

````
2. Install Python packages:
   ```bash
pip install -r requirements.txt
````

## Usage

Run the training script:

```bash
python main.py
```

* The agent will train for 500 episodes by default.
* Training logs (episode number, score, epsilon) will be printed to the console.

## File Structure

```
├── main.py                # Entry point: trains and evaluates the DQN agent
├── agent.py               # DQNAgent class with TransformerBlock
├── transformer_block.py   # Definition of TransformerBlock layer
├── requirements.txt       # List of Python dependencies
└── README.md              # This file
```

## Acknowledgments

* Developed during my undergraduate studies in 2020.
* Gymnasium by Farama for modernized RL environments.

## License

MIT License

