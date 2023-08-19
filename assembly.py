import taichi as ti
from datatypeUtilities import *
from indexUtilities import *
from assemblyUtilities import *

@ti.func
def uoffsets():
    X = ti.Vector([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    Y = ti.Vector([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1])
    Z = ti.Vector([-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
    return(X,Y,Z)

@ti.func
def getLocalU(u:nd4f64,nodeIndex:index)->vec81f64:
    nodeOffsetsX,nodeOffsetsY,nodeOffsetsZ = uoffsets()
    nx = u.shape[0]
    ny = u.shape[1]
    nz = u.shape[2]
    ni = nodeIndex
    res = ti.Vector.zero(dt=ti.f64,n=81)
    for i in range(nodeOffsetsX.n):
        if indexIsInside(nx,ny,nz,index([ni.x+nodeOffsetsX[i],ni.y+nodeOffsetsY[i],ni.z+nodeOffsetsZ[i]])):
            res[i*3] = u[ni.x+nodeOffsetsX[i],ni.y+nodeOffsetsY[i],ni.z+nodeOffsetsZ[i],0]
            res[i*3+1] = u[ni.x+nodeOffsetsX[i],ni.y+nodeOffsetsY[i],ni.z+nodeOffsetsZ[i],1]
            res[i*3+2] = u[ni.x+nodeOffsetsX[i],ni.y+nodeOffsetsY[i],ni.z+nodeOffsetsZ[i],2]
    return res

@ti.func
def getCellContribution(l : ti.int32, x : nd3f32, nodeIndex : index, elementOffset : index) -> n3m24f64:
    cellIndex = addIndices(nodeIndex, elementOffset)
    nodeOffset = index(-elementOffset.x, -elementOffset.y, -elementOffset.z)
    weights = getInitialWeights(nodeOffset)
    valX = getCoarseCellContribution(x, 0, l , cellIndex, weights)
    valY = getCoarseCellContribution(x, 1, l , cellIndex, weights)
    valZ = getCoarseCellContribution(x, 2, l , cellIndex, weights)
    return (valX, valY, valZ)

cellValues = ti.field(dtype=ti.f64,shape=(3,192))
@ti.func
def getNodeAssembledRow(l : ti.int32, x : nd3f32, nodeIndex : index) -> n3m81f64:
    for i in range(elementOffsets.n):
        valX, valY, valZ = getCellContribution(l,x,nodeIndex,index([elementOffsets[i,0],elementOffsets[i,1],elementOffsets[i,2]]))
        for j in range(valX.n):
            cellValues[0, j + i * valX.n] = valX[j]
            cellValues[1, j + i * valY.n] = valY[j]
            cellValues[2, j + i * valZ.n] = valZ[j]
    res = ti.Matrix.zero(dt=ti.f64,n=3,m=81)
    for i in range(ti_elementAssembledOffsets.n):
        res[0,ti_elementAssembledOffsets[i]] = res[0,ti_elementAssembledOffsets[i]] + cellValues[0,i]
        res[1,ti_elementAssembledOffsets[i]] = res[1,ti_elementAssembledOffsets[i]] + cellValues[1,i]
        res[2,ti_elementAssembledOffsets[i]] = res[2,ti_elementAssembledOffsets[i]] + cellValues[2,i]
    cellValues.fill(0)
    return res
        
    

@ti.func
def assembleStiffnessMatrix(l : ti.int32, x : nd3f32, res : nd5f64):
    ncell = 2**l
    nx = ti.int32(x.shape[0]/ncell+1)
    ny = ti.int32(x.shape[1]/ncell+1)
    nz = ti.int32(x.shape[2]/ncell+1)
    for i,j,k in ti.ndrange(nx,ny,nz):
        tmp = getNodeAssembledRow(l,x,index([i,j,k]))
        for n,m in ti.ndrange(tmp.n,tmp.m):
            res[i,j,k,n,m]=tmp[n,m]

@ti.func
def extractDiagonal(mat : nd5f64, res : nd4f64):
    for i,j,k in ti.ndrange(mat.shape[0],mat.shape[1],mat.shape[2]):
        res[i,j,k,0] = mat[i,j,k,0,39]
        res[i,j,k,1] = mat[i,j,k,1,40]
        res[i,j,k,2] = mat[i,j,k,2,41]

@ti.func
def extractInverseDiagonal(mat : nd5f64, res : nd4f64):
    extractDiagonal(mat, res)
    for idx in ti.grouped(res):
        res[idx] = 1. / res[idx]

@ti.func
def applyAssembledStiffnessMatrix(mat : nd5f64,u:nd4f64,res:nd4f64):
    for i,j,k in ti.ndrange(mat.shape[0],mat.shape[1],mat.shape[2]):
        uloc = getLocalU(u,index([i,j,k]))
        mloc = n3m81f64([[mat[i,j,k,n,m] for m in range(mat.shape[4])] for n in range(mat.shape[3])])
        for n in range(mat.shape[3]):
            sum = 0.
            for m in range(mat.shape[4]):
                sum = sum+uloc[m]*mloc[n,m]
            res[i,j,k,n] = sum

