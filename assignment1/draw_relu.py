import math
import random


import numpy as np

import matplotlib.pyplot as plt

def compute_activation(input, w, b):
    linear_value = w * input + b
    relu = np.maximum(0, linear_value)

    return relu

x = np.array([-5, -4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7,7,7, 7.8])
relu_1 = compute_activation(x, 1, 1)
plt.plot(x, relu_1, 'b')
relu_2 = compute_activation(relu_1, -2, 2)
plt.plot(x, relu_2, 'g')
relu_3 = compute_activation(relu_2, 3, 2)
plt.plot(x, relu_3, 'y')
relu_4 = compute_activation(relu_3, 2, -1)
plt.plot(x, relu_4, 'r')
plt.show()

#
# import matplotlib.pyplot as plt
# import numpy as np
# import random
#
# def f(x,weights,biases):
#   for w,b in zip(weights,biases):
#     x = max(0.01*x, w*x+b)
#   return x
#
# x = np.linspace(-10,10,300)
# # w = [(random.random()-0.5)*10 for _ in range(5)]
# # b = [random.random()-0.5 for _ in range(5)]
# w=[1,-2,3,4,5, -6]
# b=[1,2,-3,4,5, 5]
# y = [f(a,w,b) for a in x]
# plt.plot(x,y)
# plt.show()
