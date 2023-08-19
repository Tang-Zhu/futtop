import taichi as ti
from datatypeUtilities import *

@ti.func
def vecmul(A : n8m8f64, x : n8m8f64)->vec8f64:
    res = ti.Vector.zero(dt=ti.f64,n=A.n)
    for i in range(A.n):
        for j in range(A.m):
            res[i] = res[i] + A[i,j] * x[j]
    return res
