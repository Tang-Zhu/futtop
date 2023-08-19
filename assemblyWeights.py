import numpy as np

# 8*8*8
prolongationWeights =np.array([
[[1,0,0,0,0,0,0,0],
[0.5,0.5,0,0,0,0,0,0],
[0.25,0.25,0.25,0.25,0,0,0,0],
[0.5,0,0,0.5,0,0,0,0],
[0.5,0,0,0,0.5,0,0,0],
[0.25,0.25,0,0,0.25,0.25,0,0],
[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
[0.25,0,0,0.25,0.25,0,0,0.25]],
[[0.5,0.5,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0],
[0,0.5,0.5,0,0,0,0,0],
[0.25,0.25,0.25,0.25,0,0,0,0],
[0.25,0.25,0,0,0.25,0.25,0,0],
[0,0.5,0,0,0,0.5,0,0],
[0,0.25,0.25,0,0,0.25,0.25,0],
[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]],
[[0.25,0.25,0.25,0.25,0,0,0,0],
[0,0.5,0.5,0,0,0,0,0],
[0,0,1,0,0,0,0,0],
[0,0,0.5,0.5,0,0,0,0],
[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
[0,0.25,0.25,0,0,0.25,0.25,0],
[0,0,0.5,0,0,0,0.5,0],
[0,0,0.25,0.25,0,0,0.25,0.25]],
[[0.5,0,0,0.5,0,0,0,0],
[0.25,0.25,0.25,0.25,0,0,0,0],
[0,0,0.5,0.5,0,0,0,0],
[0,0,0,1,0,0,0,0],
[0.25,0,0,0.25,0.25,0,0,0.25],
[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
[0,0,0.25,0.25,0,0,0.25,0.25],
[0,0,0,0.5,0,0,0,0.5]],
[[0.5,0,0,0,0.5,0,0,0],
[0.25,0.25,0,0,0.25,0.25,0,0],
[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
[0.25,0,0,0.25,0.25,0,0,0.25],
[0,0,0,0,1,0,0,0],
[0,0,0,0,0.5,0.5,0,0],
[0,0,0,0,0.25,0.25,0.25,0.25],
[0,0,0,0,0.5,0,0,0.5]],
[[0.25,0.25,0,0,0.25,0.25,0,0],
[0,0.5,0,0,0,0.5,0,0],
[0,0.25,0.25,0,0,0.25,0.25,0],
[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
[0,0,0,0,0.5,0.5,0,0],
[0,0,0,0,0,1,0,0],
[0,0,0,0,0,0.5,0.5,0],
[0,0,0,0,0.25,0.25,0.25,0.25]],
[[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
[0,0.25,0.25,0,0,0.25,0.25,0],
[0,0,0.5,0,0,0,0.5,0],
[0,0,0.25,0.25,0,0,0.25,0.25],
[0,0,0,0,0.25,0.25,0.25,0.25],
[0,0,0,0,0,0.5,0.5,0],
[0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0.5,0.5]],
[[0.25,0,0,0.25,0.25,0,0,0.25],
[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
[0,0,0.25,0.25,0,0,0.25,0.25],
[0,0,0,0.5,0,0,0,0.5],
[0,0,0,0,0.5,0,0,0.5],
[0,0,0,0,0.25,0.25,0.25,0.25],
[0,0,0,0,0,0,0.5,0.5],
[0,0,0,0,0,0,0,1]]])