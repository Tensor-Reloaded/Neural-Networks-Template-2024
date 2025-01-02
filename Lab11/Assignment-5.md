# Assignment 5

Implement and train a neural network using the Q learning algorithm to control an agent in the Flappy Bird game.

## Environment

You can use a Flappy Bird environment from [here](https://pypi.org/project/flappy-bird-gymnasium/) or [here](https://github.com/Talendar/flappy-bird-gym) or another environment. Why not create your own? If you use a pre-made environment, make sure you can render the environment and interact with it.

## Specifications

You can train the model directly on images (the model receives the pixels) or you can extract helpful features. Based on the input you are using for the model, the maximum score is capped to:

- 20 points: if you provide the game state directly (this might include positions of the pipes, bird, direction, simple distances)
- 25 points: if you provide preprocessed features (this might include more complex features extracted from the image: e.g. sensors/lidar for the bird)
- 30 points: if you use the image as input, eventually preprocessed, if needed (resizing, grayscale conversion, thresholding, dilation, erosion, background removal, etc.)

It is not necessary to implement the neural network from scratch (you can use PyTorch), but you must implement the Q learning algorithm.

## Evaluation

Deadline: January 13th, 2025

Teams of 2 people.

You should create a markdown/pdf report explaining the architecture that you used alongside the hyperparameters. Also include various experimentation attempts and the score that the agent achieves on multiple runs.

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
