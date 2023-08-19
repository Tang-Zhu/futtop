import taichi as ti
from datatypeUtilities import *
from indexUtilities import *
from material import *
from assemblyWeights import *
from utility import *

elementOffsets = ti.Matrix(
    [[-1,0-1],[0,0,-1],
     [0,-1,-1],[-1,-1,-1],
     [-1,0,0],[0,0,0],
     [0,-1,0],[-1,-1,0]],
     ti.int32)

@ti.func
def getFineValue(x:nd3f32,dofNumber:ti.i64,cellIndex :index, w:nodalWeights)->vec24f64:
    nelx = x.shape[0]
    nely = x.shape[1]
    nelz = x.shape[2]
    elementScale = getElementYoungsModule(x, cellIndex)
    w_onBoundary = applyBoundaryConditionsToWeightsInverse(cellIndex, w, nelx, nely, nelz, dofNumber)
    w_inDomain = applyBoundaryConditionsToWeights(cellIndex, w, nelx, nely, nelz, dofNumber)
    l_onBoundary = generateLoad(dofNumber, w_onBoundary)
    l_inDomain = generateLoad(dofNumber, w_inDomain)
    kevalues = keprod(elementScale, l_inDomain)
    return ti.Vector([kevalues[i] + l_onBoundary[i] for i in range(24)])

@ti.func
def generateLoad(o:ti.i64, w:nodalWeights) -> lclst:
    res = ti.Vector.zero(dt=ti.f64,n=24)
    idxs = ti.Vector([0+o,3+o,6+o,9+o,12+o,15+o,18+o,21+o])
    for i in range(idxs.n):
        res[idxs[i]] = w[i]
    return res

@ti.func
def getInitialWeights(nodeOffset : index) -> nodalWeights:
    li = getLocalNodeIndex(nodeOffset)
    res = ti.Vector.zero(dt=ti.f64,n=8)
    res[li] = 1
    return res

@ti.func
def applyBoundaryConditionsToWeights(elementIndex :index, w :nodalWeights, nx :ti.i64, ny :ti.i64, nz :ti.i64, dofNumber :ti.i64)->nodalWeights:
    idxs = getNodeIndices(elementIndex)
    flags = ti.Vector([isOnBoundary(nx,ny,nz,index(idxs[i,0],idxs[i,1],idxs[i,2]),dofNumber)])
    res = ti.Vector.zero(dt=ti.f64,n=8)
    for i in range(flags.n):
        if not flags[i]:
            res[i] = w[i]
    return res

@ti.func
def applyBoundaryConditionsToWeightsInverse(elementIndex :index, w :nodalWeights, nx :ti.i64, ny :ti.i64, nz :ti.i64, dofNumber :ti.i64)->nodalWeights:
    idxs = getNodeIndices(elementIndex)
    flags = ti.Vector([isOnBoundary(nx,ny,nz,index(idxs[i,0],idxs[i,1],idxs[i,2]),dofNumber) and indexIsInside(nx,ny,nz,index(idxs[i,0],idxs[i,1],idxs[i,2])) for i in range(idxs.n)])
    res = ti.Vector.zero(dt=ti.f64,n=8)
    for i in range(flags.n):
        if flags[i]:
            res[i] = w[i] / 8
    return res

ti_prolongationWeights = ti.field(dtype=ti.f64,shape=(8,8,8))
ti_prolongationWeights.from_numpy(prolongationWeights)

@ti.func
def prolongateCellValues(w:nodalWeights)->nodalWeightsDim8:
    res = ti.Matrix.zero(dt=ti.f64,n=8,m=8)
    for i in range(ti_prolongationWeights.shape[0]):
        tmp = vecmul(ti.Matrix([[ti_prolongationWeights[i,j,k] for k in range(ti_prolongationWeights.shape[2])] for j in range(ti_prolongationWeights.shape[1])]), w)
        for j in range(tmp.n):
            res[i,j] = tmp[j]
    return res

@ti.func
def prolongateCellIndices(cellIndex : index)->idx8p:
    return getNodeIndices(index([2*cellIndex.x,2*cellIndex.y,2*cellIndex.z]))

@ti.func
def restrictCell(vals : n8m24f64)->vec24f64:
    indX = ti.Vector([0,3,6, 9,12,15,18,21])
    indY   = ti.Vector([1,4,7,10,13,16,19,22])
    indZ   = ti.Vector([2,5,8,11,14,17,20,23])
    indAll = ti.Vector([0,3,6, 9,12,15,18,21,1,4,7,10,13,16,19,22,2,5,8,11,14,17,20,23])
    valM = ti.Matrix([[vals[i,indX[j]] for j in range(indX.n)] for i in range(vals.n)])
    for i in range(ti_prolongationWeights.shape[0]):
        tmp = vecmul(ti.Matrix([[ti_prolongationWeights[i,j,k] for k in range(ti_prolongationWeights.shape[2])] for j in range(ti_prolongationWeights.shape[1])]).transpose(), ti.Vector([valM[i,j] for j in range(valM.m)]))
        for j in range(tmp.n):
            valM[i,j] = tmp[j]
    valMT = valM.transpose()
    valX = ti.Vector.zero(dt=ti.f64,n=valMT.n)
    for i in range(valMT.n):
        for j in range(valMT.m):
            valX[i] = valX[i] + valMT[i,j]

    valM = ti.Matrix([[vals[i,indY[j]] for j in range(indY.n)] for i in range(vals.n)])
    for i in range(ti_prolongationWeights.shape[0]):
        tmp = vecmul(ti.Matrix([[ti_prolongationWeights[i,j,k] for k in range(ti_prolongationWeights.shape[2])] for j in range(ti_prolongationWeights.shape[1])]).transpose(), ti.Vector([valM[i,j] for j in range(valM.m)]))
        for j in range(tmp.n):
            valM[i,j] = tmp[j]
    valMT = valM.transpose()
    valY = ti.Vector.zero(dt=ti.f64,n=valMT.n)
    for i in range(valMT.n):
        for j in range(valMT.m):
            valY[i] = valY[i] + valMT[i,j]
    
    valM = ti.Matrix([[vals[i,indZ[j]] for j in range(indZ.n)] for i in range(vals.n)])
    for i in range(ti_prolongationWeights.shape[0]):
        tmp = vecmul(ti.Matrix([[ti_prolongationWeights[i,j,k] for k in range(ti_prolongationWeights.shape[2])] for j in range(ti_prolongationWeights.shape[1])]).transpose(), ti.Vector([valM[i,j] for j in range(valM.m)]))
        for j in range(tmp.n):
            valM[i,j] = tmp[j]
    valMT = valM.transpose()
    valZ = ti.Vector.zero(dt=ti.f64,n=valMT.n)
    for i in range(valMT.n):
        for j in range(valMT.m):
            valZ[i] = valZ[i] + valMT[i,j]

    res = ti.Vector.zero(dt=ti.f64,n=24)
    for i in range(valX.n):
        res[indAll[i]] = valX[i]
    for i in range(valY.n):
        res[indAll[i + valX.n]] = valY[i]
    for i in range(valZ.n):
        res[indAll[i + valX.n + valY.n]] = valZ[i]

    return res
    

@ti.func
def getCoarseCellContribution(x:nd3f32,dofNumber:ti.i64,l:ti.int32,cellIndex:index,w:nodalWeights)->vec24f64:
    res = ti.Vector.zero(dt=ti.f64,n=24)
    if l == 0:
        res = getFineValue(x,dofNumber,cellIndex,w)

    elif l == 1:
        fineCellWeights = prolongateCellValues(w)
        fineCellIndices = prolongateCellIndices(cellIndex)
        fineValues = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
        for i in range(fineValues.n):
            tmp = getFineValue(x,dofNumber,fineCellIndices[i],fineCellWeights[i])
            for j in range(fineValues.m):
                fineValues[i,j] = tmp[j]
        res = restrictCell(fineValues)

    elif l == 2:
        fineCellWeights = prolongateCellValues(w)
        fineCellIndices = prolongateCellIndices(cellIndex)
        fineValues = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
        for i in range(fineCellIndices.n):
            fineCellWeightsRow = prolongateCellValues(fineCellWeights[i])
            fineCellIndicesRow = prolongateCellIndices(fineCellIndices[i])
            fineValuesRow = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
            for j in range(fineCellIndicesRow.n):
                tmp = getFineValue(x,dofNumber,fineCellIndicesRow[j],fineCellWeightsRow[j])
                for k in range(fineValuesRow.m):
                    fineValuesRow[j,k] = tmp[k]
            tmp = restrictCell(fineValuesRow)
            for j in range(tmp.n):
                fineValues[i,j]=tmp[j]
        res = restrictCell(fineValues)
    
    elif l == 3:
        fineCellWeights = prolongateCellValues(w)
        fineCellIndices = prolongateCellIndices(cellIndex)
        fineValues = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
        for l1 in range(fineCellIndices.n):
            fineCellWeights1Layer = prolongateCellValues(fineCellWeights[l1])
            fineCellIndices1Layer = prolongateCellIndices(fineCellIndices[l1])
            fineValues1Layer = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
            for l2 in range(fineCellIndices1Layer.n):
                fineCellWeights2Layer = prolongateCellValues(fineCellWeights1Layer[l2])
                fineCellIndices2Layer = prolongateCellIndices(fineCellIndices1Layer[l2])
                fineValues2Layer = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
                for i in range(fineCellIndices2Layer.n):
                    tmp = getFineValue(x,dofNumber,fineCellIndices2Layer[i],fineCellWeights2Layer[i])
                    for j in range(fineValues2Layer.m):
                        fineValues2Layer[i,j] = tmp[j]
                tmp = restrictCell(fineCellIndices2Layer)
                for i in range(tmp.n):
                    fineValues1Layer[l2, i] = tmp[i]
            tmp = restrictCell(fineValues1Layer)
            for i in range(tmp.n):
                fineValues[l1,i]=tmp[i]
        res = restrictCell(fineValues)

    elif l == 4:
        fineCellWeights = prolongateCellValues(w)
        fineCellIndices = prolongateCellIndices(cellIndex)
        fineValues = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
        for l1 in range(fineCellIndices.n):
            fineCellWeights1Layer = prolongateCellValues(fineCellWeights[l1])
            fineCellIndices1Layer = prolongateCellIndices(fineCellIndices[l1])
            fineValues1Layer = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
            for l2 in range(fineCellIndices1Layer.n):
                fineCellWeights2Layer = prolongateCellValues(fineCellWeights1Layer[l2])
                fineCellIndices2Layer = prolongateCellIndices(fineCellIndices1Layer[l2])
                fineValues2Layer = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
                for l3 in range(fineCellIndices2Layer.n):
                    fineCellWeights3Layer = prolongateCellValues(fineCellWeights2Layer[l3])
                    fineCellIndices3Layer = prolongateCellIndices(fineCellIndices2Layer[l3])
                    fineValues3Layer = ti.Matrix.zero(dt=ti.f64,n=8,m=24)
                    for i in range(fineCellIndices3Layer.n):
                        tmp = getFineValue(x,dofNumber,fineCellIndices3Layer[i],fineCellWeights3Layer[i])
                        for j in range(fineValues3Layer.m):
                            fineValues3Layer[i,j] = tmp[j]
                    tmp = restrictCell(fineCellIndices3Layer)
                    for i in range(tmp.n):
                        fineValues2Layer[l3, i] = tmp[i]
                tmp = restrictCell(fineValues2Layer)
                for i in range(tmp.n):
                    fineValues1Layer[l2,i]=tmp[i]
            tmp = restrictCell(fineValues1Layer)
            for i in range(tmp.n):
                fineValues[l1,i]=tmp[i]
        res = restrictCell(fineValues)
    
    return res

# 192
elementAssembledOffsets = np.array(
    [18, 19, 20, 45, 46, 47, 36, 37, 38, 9, 10,
    11, 21, 22, 23, 48, 49, 50, 39, 40, 41, 12,
    13, 14, 45, 46, 47, 72, 73, 74, 63, 64, 65,
    36, 37, 38, 48, 49, 50, 75, 76, 77, 66, 67,
    68, 39, 40, 41, 36, 37, 38, 63, 64, 65, 54,
    55, 56, 27, 28, 29, 39, 40, 41, 66, 67, 68,
    57, 58, 59, 30, 31, 32, 9, 10, 11, 36, 37,
    38, 27, 28, 29, 0, 1, 2, 12, 13, 14, 39,
    40, 41, 30, 31, 32, 3, 4, 5, 21, 22, 23,
    48, 49, 50, 39, 40, 41, 12, 13, 14, 24, 25,
    26, 51, 52, 53, 42, 43, 44, 15, 16, 17, 48,
    49, 50, 75, 76, 77, 66, 67, 68, 39, 40, 41,
    51, 52, 53, 78, 79, 80, 69, 70, 71, 42, 43,
    44, 39, 40, 41, 66, 67, 68, 57, 58, 59, 30,
    31, 32, 42, 43, 44, 69, 70, 71, 60, 61, 62,
    33, 34, 35, 12, 13, 14, 39, 40, 41, 30, 31,
    32, 3, 4, 5, 15, 16, 17, 42, 43, 44, 33,
    34, 35, 6, 7, 8])
ti_elementAssembledOffsets = ti.field(dtype=ti.int32,shape=192)