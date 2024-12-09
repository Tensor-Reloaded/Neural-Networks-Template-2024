# Assignment 5

Implement and train a neural network using the Q learning algorithm to control an agent in the Flappy Bird game.

## Environment

You can use a Flappy Bird environment from [here](https://pypi.org/project/flappy-bird-gymnasium/) or [here](https://github.com/Talendar/flappy-bird-gym) or another environment. Why not create your own? If you use a pre-made environment, make sure you can render the environment and interact with it.

## Specifications

You can train the model directly on images (the model receives the pixels) or you can extract helpful features. If you use preprocessed data (extract certain features such as direction, distance to the pipe, etc.), you can receive a maximum score of 20 points.

It is not necessary to implement the neural network from scratch (you can use PyTorch), but you must implement the Q learning algorithm.

## Evaluation

Deadline: January 6th, 2025

Teams of 2 people.

The 30-point score for the assignment will include the following:
- training based on pixels
- explanations for the architecture and implementation of the neural network
- explanations for the implementation of the Q learning algorithm
- the agent's performance in the game environment

## Suggestions

If you train on pixels, it will be helpful to experiment with convolutional neural networks.

Don't forget to use: replay buffer, epsilon greedy, and other techniques discussed in the course to improve convergence.

If you do not have sufficient hardware resources, you can try Google Colab for faster experimentation.

You could reduce the inference frequency to a number of frames. Also, when taking the action to jump, you can skip a number of frames since it has a long-lasting effect. Experiment with multiple such modifications and combinations to achieve a performant agent.
