import taichi as ti
from indexUtilities import *
from material import *
from keConstants import *
from boundaryConditions import *
from projection import *
from datatypeUtilities import *

ti.init(arch = ti.cuda)

ti_keslices = ti.field(dtype = ti.f64, shape = (8,3,24))
ti_keslices.from_numpy(keslices)

@ti.func
def getElementContribution(nodeIndex : index, u : nd4f64, x : nd3f32, elementOffset : index) -> vec3f64:
    elementIndex = addIndices(nodeIndex, elementOffset)
    ulocal = getLocalState(u, elementIndex)
    E = getElementYoungsModule(x, elementIndex)
    recievingNodeOffset = index(-elementOffset)
    li = getLocalNodeIndex(recievingNodeOffset)
    res = vec3f64(0)
    for i in range(3):
        for j in range(24):
            res[i] = res[i] + ti_keslices[li, i, j] * E * ulocal[j]
    return res
            

@ti.func
def applyStencilOnNode(nodeIndex : index, u : nd4f64, x : nd3f32) -> vec3f64:
    stencils = ti.Matrix([
        [0, 0, 0],   [-1, 0, 0],
        [0, -1, 0],  [-1, -1, 0],
        [0, 0, -1],  [-1, 0, -1],
        [0, -1, -1], [-1, -1, -1]
        ], ti.i32)
    res = vec3f64(0)
    for i in range(stencils.n):
        idx = index([stencils[i,0],stencils[i,1],stencils[i,2]])
        res += getElementContribution(nodeIndex, u, x, idx)
    return res
    

@ti.func
def applyStiffnessMatrix(x: nd3f32, u : nd4f64, res : nd4f64):
    for i,j,k in ti.ndrange(u.shape[0], u.shape[1], u.shape[2]):
        tmp = applyStencilOnNode(index([i,j,k]), u, x)
        res[i,j,k, 0] = tmp[0]
        res[i,j,k, 1] = tmp[1]
        res[i,j,k, 2] = tmp[2]
    setBCtoInput(u, res)


@ti.func
def applyStiffnessMatrixSingle(x:nd3f32,u:nd4f64,res:nd4f32):
    applyStiffnessMatrix(x,u,res)



# u must be the finest grid size
@ti.func
def applyCoarseStiffnessMatrix(l : ti.int32, x : nd3f32, u : nd4f64, boundaries : index, res : nd4f64):
    
    for _ in range(l):
        boundaries = projectToFiner(u, boundaries)
    
    applyStiffnessMatrix(x,u,res)

    for _ in range(l):
        boundaries = projectToCoarser(res, boundaries)


    
    
