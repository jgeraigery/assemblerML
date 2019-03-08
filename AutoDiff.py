#Auto differentiation
from __future__ import print_function
import torch as tch
from torch import FloatTensor
from torch.autograd import Variable


# Define the leaf nodes
a = Variable(FloatTensor([4]))
#weights=list([])
l=0
# for i in (2,5,9,7):
#     wt = Variable(FloatTensor([i]), requires_grad=True)
#     wt.retain_grad()
#     weights=[weights,wt]
#weights = Variable(FloatTensor([2.,5.,9.,7.], requires_grad=True))
weights=[Variable(FloatTensor([i]), requires_grad=True) for i in (2, 5, 9, 7)]
#weights.retain_grad()
# unpack the weights for nicer assignment
w1, w2, w3, w4 = weights

b = w1 * a
b.retain_grad()
c = w2 * a
c.retain_grad()
d = w3 * b + w4 * c
d.retain_grad()
L = (10. - d)



b.register_hook(print)
c.register_hook(print)
d.register_hook(print)
print(b.grad,c.grad,d.grad)
L.backward(retain_graph=True)
print(b.grad,c.grad,d.grad)
for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print("Gradient of w{",index,"} w.r.t to L: ",gradient)
L.backward(retain_graph=True)
print(b.grad,c.grad,d.grad)
for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print("Gradient of w{", index, "} w.r.t to L: ", gradient)
L.backward(retain_graph=True)
print(b.grad,c.grad,d.grad)
for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print("Gradient of w{", index, "} w.r.t to L: ", gradient)
