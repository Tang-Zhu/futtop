import taichi as ti
from datatypeUtilities import *


@ti.func
def isOnBoundary(nx : ti.i64, ny : ti.i64, nz : ti.i64, nodeIndex : index, dofnumber : ti.i64) -> bool:
   return nodeIndex.x == 0 and (nodeIndex.y <(ny / 4) or (nodeIndex.y > ny -1 - (ny / 4)))

@ti.func
def setBCtoInput(input : nd4, v : nd4):
   nx = input.shape[0]
   ny = input.shape[1]
   nz = input.shape[2]
   for idx in ti.grouped(input):
      if isOnBoundary(nx, ny, nz, idx[:3], idx[3]):
         v[idx] = input[idx]

@ti.func
def setBCtoZero(zero, v:nd4):
   nx = v.shape[0]
   ny = v.shape[1]
   nz = v.shape[2]
   for idx in ti.grouped(v):
      if isOnBoundary(nx, ny, nz, idx[:3], idx[3]):
         v[idx]=zero
         