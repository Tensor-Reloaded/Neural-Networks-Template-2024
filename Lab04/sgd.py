from typing import List, Optional
from numpy import ndarray


def sgd(weights: List[ndarray],
        grads: List[ndarray],
        momentum_buffer_list: List[Optional[ndarray]],
        weight_decay: float,
        momentum: float,
        lr: float):
    # weight_decay is usually 0.0001 or so
    # momentum is usually 0.9

    for i, weight in enumerate(weights):
        dW = grads[i]  # The gradient for weight i
        # Other forms of regularization can be included in the gradient

        if weight_decay != 0:
            # This is the simplest form of regularization without depending on the gradient
            dW = dW + weight * weight_decay

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:  # The first step
                buf = dW.clone()
                momentum_buffer_list[i] = buf
            else:
                buf = buf * momentum + dW
                momentum_buffer_list[i] = buf

            dW = buf

        weight = weight - lr * dW
        weights[i] = weight
