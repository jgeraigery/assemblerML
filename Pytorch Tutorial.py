from __future__ import print_function
import torch
x = tch.empty(5, 3)
print(x)
x = tch.rand(5, 3)
print(x)
x = tch.zeros(5, 3, dtype=torch.long)
print(x)
x = tch.tensor([5.5, 3])
print(x)



# Define the leaf nodes
a = tch.tensor([4])

weights = tch. requires_grad=True) for i in (2, 5, 9, 7)]

# unpack the weights for nicer assignment
w1, w2, w3, w4 = weights

b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
L = (10 - d)

L.backward()

for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print(f"Gradient of w{index} w.r.t to L: {gradient}")