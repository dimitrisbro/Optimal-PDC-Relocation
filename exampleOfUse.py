import alg
import numpy as np

# PMU Placement Example
'''
bus = 7
branches = np.array(
    [(1, 2), (2, 3), (2, 6), (3, 7), (3, 4), (3, 6), (4, 5), (4, 7)])
injection = [2]
flow = [(2, 3)]
v=alg.pmuPlacement(branches, bus, injection, flow)
print(v)
'''

# PDC Placement Example
bus = 7
pmu = [2, 5]
branches = np.array(
    [(1, 2), (2, 3), (2, 6), (2, 7), (3, 4), (3, 6), (4, 5), (4, 7)])
dist = np.array([[5, 0, 5, 10, 12, 3, 7], [12, 12, 2, 2, 0, 6, 4]])
d=5
v = alg.pdcPlacement(branches, bus, pmu, dist, d)
print(v)