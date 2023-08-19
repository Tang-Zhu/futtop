import taichi as ti
import numpy as np
from boundaryConditions import *
from datatypeUtilities import *

@ti.func
def getNodeIndices(elementIndex : index) -> idx8p:
    x, y, z = elementIndex.x, elementIndex.y, elementIndex.z
    return idx8p([index([x, y + 1, z]), index([x + 1, y + 1, z]),
                index([x + 1, y, z]), index([x, y, z]),
                index([x, y + 1, z + 1]), index([x + 1, y + 1, z + 1]),
                index([x + 1, y, z + 1]), index([x, y, z + 1])])

@ti.func
def NodeIndicesToDofIndices(nodeIndices : idx8p):
    nodes = ti.Matrix([[nodeIndices[i,j] for j in range(3)] for i in range(8) for _ in range(3)])
    dofs = ti.Vector([i % 3 for i in range(24)])
    return (nodes, dofs)

@ti.func
def readLocalState(u : nd4, elementIndex :index) -> vec24f64:
    nodeIndices = getNodeIndices(elementIndex)
    res = vec24f64(0)
    for i in range(8):
        x, y, z = nodeIndices[i, :]
        res[i * 3] = u[x, y, z, 0]
        res[i * 3 + 1] = u[x, y, z, 1]
        res[i * 3 + 2] = u[x, y, z, 2]
    return res  

@ti.func
def getLocalState(u : nd4f64, elementIndex :index) -> vec24f64:
    res = ti.Vector.zero(dt=ti.f64, n=24)
    if(indexIsInside(u.shape[0]-1,u.shape[1]-1,u.shape[2]-1,elementIndex)):
        nodeIndices, dofIndices = NodeIndicesToDofIndices(getNodeIndices(elementIndex))
        nx, ny, nz = u.shape[0], u.shape[1], u.shape[2]
        isOnB = ti.Vector([isOnBoundary(nx,ny,nz,index([nodeIndices[i,0],nodeIndices[i,1],nodeIndices[i,2]]), dofIndices[i]) for i in range(dofIndices.n)])
        ulocal = readLocalState(u, elementIndex)
        for i in range(isOnB.n):
            if not isOnB[i]:
                res[i] = ulocal[i]
    return res

@ti.func
def addIndices(a : index, b : index) -> index:
    return index([a.x + b.x, a.y + b.y, a.z + b.z])

@ti.func
def indexIsInside (nelx : ti.int64, nely : ti.int64, nelz : ti.int64, idx : index) -> bool:
    return (idx.x >= 0 and idx.y >= 0 and idx.z >= 0 and idx.x < nelx and idx.y < nely and idx.z < nelz)

@ti.func
def getLocalNodeIndex(node : index) -> ti.i32:
    res = -1
    if node.x == 0 and node.y == 1 and node.z == 0:
        res = 0
    elif node.x == 1 and node.y == 1 and node.z == 0:
        res = 1
    elif node.x == 1 and node.y == 0 and node.z == 0:
        res = 2
    elif node.x == 0 and node.y == 0 and node.z == 0:
        res = 3              
    elif node.x == 0 and node.y == 1 and node.z == 1:
        res = 4
    elif node.x == 1 and node.y == 1 and node.z == 1:
        res = 5 
    elif node.x == 1 and node.y == 0 and node.z == 1:
        res = 6 
    elif node.x == 0 and node.y == 0 and node.z == 1:
        res = 7  
    return res

@ti.func
def getDensityUnsafe(x: nd3f32, elementIndex: index) -> ti.f64:
    return ti.cast(x[elementIndex.x, elementIndex.y, elementIndex.z], ti.f64)

