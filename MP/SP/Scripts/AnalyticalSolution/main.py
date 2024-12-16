from analytical import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

pgen = 0.01
pswap = 0.596
pfusion = 1.0

x = [2, 3, 4, 8]
y = []
for n in x:
    if n == 2:
        y.append((1/pfusion)*two_segment_solution(pgen, pswap, "entanglement")[0])
    elif n == 3:
        y.append((1/pfusion)*three_segment_solution(pgen, pswap, "entanglement")[0])
    else:
        y.append((1/pfusion)*n_segment_solution(n, pgen, pswap)[0])


x2 = [2, 3]
y2 = []
for n in x2:
    if n == 2:
        y2.append((1/pfusion)*two_segment_solution(pgen, pswap, "swap")[0])
    elif n == 3:
        y2.append((1/pfusion)*three_segment_solution(pgen, pswap, "swap")[0])

plt.figure(figsize=(8, 6))

# Plot first function output
plt.scatter(x, y, color='hotpink', label = r"Analytical Solution ($\epsilon$)")
plt.plot(x, y, color='hotpink', linestyle = "dashed")
plt.scatter(x2, y2, color='blue', label = r"Analytical Solution ($\sigma$)")
plt.plot(x2, y2, color='blue', linestyle = "dashed")
# Labels and legend
plt.xticks(x, list(map(lambda x: r"{n} $\times$ {n}".format(n = x +1), x)))
plt.xlabel(r'Grid Size')
plt.ylabel(r'Latency')
plt.grid(True)
plt.legend(loc = "best")
plt.show()