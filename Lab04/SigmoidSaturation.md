### Forward Pass

1. **Input Layer:** The input neurons don't have activation functions. Given inputs are:
   - $x_1 = -3$
   - $x_2 = 1$

2. **Calculations for the Hidden Layer:**
   - **Neuron 1:**
     $$z_1^2 = 2 \cdot (-3) + 6 \cdot 1 = -6 + 6 = 0$$
     
     $$y_1^2 = \sigma(z_1^2) = \sigma(0) = \frac{1}{1 + e^0} = 0.5$$
   - **Neuron 2:**
     $$z_2^2 = 2 \cdot 6 + 6 \cdot (-2) = 12 - 12 = 0$$
     $$y_2^2 = \sigma(z_2^2) = \sigma(0) = \frac{1}{1 + e^0} = 0.5$$

3. **Calculations for the Output Layer:**
   - The weighted sum at the output neuron:
     $$z_3 = 8 \cdot 0.5 + 4 \cdot 0.5 = 4 + 2 = 6$$
   - Applying the sigmoid activation function:
     $$y_3 = \sigma(z_3) = \frac{1}{1 + e^{-6}} \approx 1$$

### Compute the Error in the Output Layer

The error term for the output layer neuron is calculated as:
$$\delta_3 = y_3 - t = 1 - 0 = 1$$

### Backward Pass

To update the weights, we need to calculate the error term for each hidden layer neuron and then propagate it back to the input layer.

#### Step 1: Error Term for Each Hidden Neuron

The error term for a neuron in a hidden layer is given by:
$$\delta_i^l = y_i^l (1 - y_i^l) \sum_k \delta_k^{l+1} w_{ik}^{l+1}$$
Where:
- $y_i^l$ is the output of neuron $i$ in layer $l$
- $\delta_k^{l+1}$ is the error term for the neuron in the next layer
- $w_{ik}^{l+1}$ is the weight connecting neuron $i$ in layer $l$ to neuron $k$ in the next layer

#### Step 2: Calculate the Error Terms for Hidden Neurons

Given:
- $y_1^2 = 0.5$, $y_2^2 = 0.5$
- $\delta_3 = 1$
- Weights from hidden neurons to the output: $w_{13}^2 = 8$, $w_{23}^2 = 4$

1. **Calculate $\delta_1^2$:**
   $$\delta_1^2 = y_1^2 (1 - y_1^2) \sum_k \delta_k^3 w_{1k}^3 = 0.5 (1 - 0.5) \cdot 1 \cdot 8 = 0.5 \cdot 0.5 \cdot 8 = 2$$

2. **Calculate $\delta_2^2$:**
   $$\delta_2^2 = y_2^2 (1 - y_2^2) \sum_k \delta_k^3 w_{2k}^3 = 0.5 (1 - 0.5) \cdot 1 \cdot 4 = 0.5 \cdot 0.5 \cdot 4 = 1$$

### Weight Updates Using Gradient Descent

The weight update rule is:
$$w_{ij}^l = w_{ij}^l - \eta \cdot \frac{\partial C}{\partial w_{ij}^l}$$
Where:
$$\frac{\partial C}{\partial w_{ij}^l} = \delta_j^l y_i^{l-1}$$
Given the learning rate $\eta = 0.5$, we can update the weights.

#### Update the Weights from the Hidden Layer to the Output Layer

1. **Update $w_{13}^2$:**
   $$\Delta w_{13}^2 = \eta \cdot \delta_3 \cdot y_1^2 = 0.5 \cdot 1 \cdot 0.5 = 0.25$$
   $$w_{13}^2 = 8 - 0.25 = 7.75$$

2. **Update $w_{23}^2$:**
   $$\Delta w_{23}^2 = \eta \cdot \delta_3 \cdot y_2^2 = 0.5 \cdot 1 \cdot 0.5 = 0.25$$
   $$w_{23}^2 = 4 - 0.25 = 3.75$$

#### Update the Weights from the Input Layer to the Hidden Layer

1. **Update $w_{11}^1$:**
   $$\Delta w_{11}^1 = \eta \cdot \delta_1^2 \cdot x_1 = 0.5 \cdot 2 \cdot (-3) = -3$$
   $$w_{11}^1 = 2 - (-3) = 5$$

2. **Update $w_{12}^1$:**
   $$\Delta w_{12}^1 = \eta \cdot \delta_1^2 \cdot x_2 = 0.5 \cdot 2 \cdot 1 = 1$$
   $$w_{12}^1 = 6 - 1 = 5$$

3. **Update $w_{21}^1$:**
   $$\Delta w_{21}^1 = \eta \cdot \delta_2^2 \cdot x_1 = 0.5 \cdot 1 \cdot (-3) = -1.5$$
   $$w_{21}^1 = 2 - (-1.5) = 3.5$$

4. **Update $w_{22}^1$:**
   $$\Delta w_{22}^1 = \eta \cdot \delta_2^2 \cdot x_2 = 0.5 \cdot 1 \cdot 1 = 0.5$$
   $$w_{22}^1 = 6 - 0.5 = 5.5$$

### Saturation Issue and Zero Gradient

After computing the forward pass, we observed that the output of the sigmoid function at the output neuron was $y_3 = 1$. The derivative of the sigmoid function is:

$$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z)) = y_3 \cdot (1 - y_3)$$

For $y_3 = 1$:

$$\sigma'(z_3) = 1 \cdot (1 - 1) = 0$$

This results in a zero gradient at the output neuron, causing the weight updates to become zero, effectively halting the training process. This issue arises because the sigmoid function saturates near 0 and 1, leading to vanishing gradients.

To mitigate this, try to clip values for the activation input, alternative activation functions such as ReLU or different initialization strategies for weights can help avoid saturation and allow for more effective training.
