import taichi as ti
from indexUtilities import *
from datatypeUtilities import *

@ti.func
def getWeight(idx : index) -> ti.f64:
    dist = ti.abs(idx[0]) + ti.abs(idx[1]) + ti.abs(idx[2])
    res = 0.0
    if dist == 0:
        res = 1.0 
    elif dist == 1:
        res = 0.5
    elif dist == 2:
        res = 0.25
    elif dist == 3:
        res = 0.125
    return res

@ti.func
def loadValue(u : nd4f64, boundaries : index, zero : vec3f64, idx : index) -> vec3f64:
    nx = boundaries[0]
    ny = boundaries[1]
    nz = boundaries[2]
    res = ti.Vector([u[idx,0],u[idx,1],u[idx,2]],ti.f64)
    if idx[0] < 0 or idx[1] < 0 or idx[2] < 0 or idx[0] >= nx or idx[1] >= ny or idx[2] >= nz:
        res = zero
    return res

@ti.func
def projectToFiner(u : nd4f64, boundaries : index) -> index:
    nxf = ti.int32((boundaries[0]-1) * 2 + 1)
    nyf = ti.int32((boundaries[1]-1) * 2 + 1)
    nzf = ti.int32((boundaries[2]-1) * 2 + 1)
    for i, j, k in ti.ndrange(nxf, nyf, nzf):
        ic1 =  ti.int32((i)     / 2)
        ic2 =  ti.int32((i + 1) / 2)
        jc1 =  ti.int32((j)     / 2)
        jc2 =  ti.int32((j + 1) / 2)
        kc1 =  ti.int32((k)     / 2)
        kc2 =  ti.int32((k + 1) / 2)
        M = ti.Matrix([[u[ic1,jc1,kc1,0], u[ic1,jc1,kc1,1], u[ic1,jc1,kc1,2]],
                        [u[ic1,jc1,kc2,0], u[ic1,jc1,kc2,1], u[ic1,jc1,kc2,2]],
                        [u[ic1,jc2,kc1,0], u[ic1,jc2,kc1,1], u[ic1,jc2,kc1,2]],
                        [u[ic1,jc2,kc2,0], u[ic1,jc2,kc2,1], u[ic1,jc2,kc2,2]],
                        [u[ic2,jc1,kc1,0], u[ic2,jc1,kc1,1], u[ic2,jc1,kc1,2]],
                        [u[ic2,jc1,kc2,0], u[ic2,jc1,kc2,1], u[ic2,jc1,kc2,2]],
                        [u[ic2,jc2,kc1,0], u[ic2,jc2,kc1,1], u[ic2,jc2,kc1,2]],
                        [u[ic2,jc2,kc2,0], u[ic2,jc2,kc2,1], u[ic2,jc2,kc2,2]]], 
                        ti.f64)
        
        MT = M.transpose()
        V = ti.Vector.zero(ti.f64, MT.n)

        for n in range(MT.n):
            for m in range(MT.m):
                V[n] = V[n] + MT[n,m]

        for n in range(V.n):
            V[n] = V[n] * 0.125
        
        u[i,j,k,0] = V[0]
        u[i,j,k,1] = V[1]
        u[i,j,k,2] = V[2]

    return index([nxf, nyf, nzf])
        

@ti.func
def projectToCoarser(u : nd4f64, boundaries : index) -> index:
    nxc = ti.i32((boundaries[0]-1) / 2 + 1)
    nyc = ti.i32((boundaries[1]-1) / 2 + 1)
    nzc = ti.i32((boundaries[2]-1) / 2 + 1)

    for i,j,k in ti.ndrange(nxc, nyc, nzc):
        icenter = i * 2
        jcenter = j * 2
        kcenter = k * 2
        offsets = ti.Matrix(
        [[-1,-1,-1], [-1,-1, 0], [-1,-1, 1],
        [-1, 0,-1], [-1, 0, 0], [-1, 0, 1],
        [-1, 1,-1], [-1, 1, 0], [-1, 1, 1],
        [ 0,-1,-1], [ 0,-1, 0], [ 0,-1, 1],
        [ 0, 0,-1], [ 0, 0, 0], [ 0, 0, 1],
        [ 0, 1,-1], [ 0, 1, 0], [ 0, 1, 1],
        [ 1,-1,-1], [ 1,-1, 0], [ 1,-1, 1],
        [ 1, 0,-1], [ 1, 0, 0], [ 1, 0, 1],
        [ 1, 1,-1], [ 1, 1, 0], [ 1, 1, 1]], 
        ti.i32)
        
        weights = ti.Vector.zero(ti.f64,offsets.n)
        for n in range(offsets.n):
            weights[n] = getWeight(index(offsets[n,0],offsets[n,1],offsets[n,2]))

        values = ti.Matrix.zero(ti.f64,offsets.n,3)
        for n in range(offsets.n):
            tmp = loadValue(u, boundaries,vec3f64([0,0,0]), index([offsets[n,0]+icenter,offsets[n,1]+jcenter,offsets[n,2]+kcenter]))  
            values[n,0] = tmp[0]
            values[n,1] = tmp[1]
            values[n,2] = tmp[2]
        
        for n in range(values.n):
            weight = weights[n]
            for m in range(values.m):
                values[n,m] = values[n,m] * weight

        valuesT = values.transpose()

        tmp = ti.Vector.zero(ti.f64, valuesT.n)
        for n in range(valuesT.n):
            for m in range(valuesT.m):
                tmp[n] = tmp[n] + valuesT[n,m]
       
        u[i,j,k,0] = tmp[0]
        u[i,j,k,1] = tmp[1]
        u[i,j,k,2] = tmp[2]
    
    return index([nxc, nyc, nzc])


@ti.func
def projectToCoarserSingle(u : nd4f32, boundaries : index) -> index:
    nxc = ti.i32((boundaries[0]-1) / 2 + 1)
    nyc = ti.i32((boundaries[1]-1) / 2 + 1)
    nzc = ti.i32((boundaries[2]-1) / 2 + 1)

    for i,j,k in ti.ndrange(nxc, nyc, nzc):
        icenter = i * 2
        jcenter = j * 2
        kcenter = k * 2
        offsets = ti.Matrix(
        [[-1,-1,-1], [-1,-1, 0], [-1,-1, 1],
        [-1, 0,-1], [-1, 0, 0], [-1, 0, 1],
        [-1, 1,-1], [-1, 1, 0], [-1, 1, 1],
        [ 0,-1,-1], [ 0,-1, 0], [ 0,-1, 1],
        [ 0, 0,-1], [ 0, 0, 0], [ 0, 0, 1],
        [ 0, 1,-1], [ 0, 1, 0], [ 0, 1, 1],
        [ 1,-1,-1], [ 1,-1, 0], [ 1,-1, 1],
        [ 1, 0,-1], [ 1, 0, 0], [ 1, 0, 1],
        [ 1, 1,-1], [ 1, 1, 0], [ 1, 1, 1]], 
        ti.i32)
        
        weights = ti.Vector.zero(ti.f64,offsets.n)
        for n in range(offsets.n):
            weights[n] = getWeight(index(offsets[n,0],offsets[n,1],offsets[n,2]))

        values = ti.Matrix.zero(ti.f64,offsets.n,3)
        for n in range(offsets.n):
            tmp = loadValue(u, boundaries,vec3f64([0,0,0]), index([offsets[n,0]+icenter,offsets[n,1]+jcenter,offsets[n,2]+kcenter]))  
            values[n,0] = tmp[0]
            values[n,1] = tmp[1]
            values[n,2] = tmp[2]
        
        for n in range(values.n):
            weight = weights[n]
            for m in range(values.m):
                values[n,m] = values[n,m] * weight

        valuesT = values.transpose()

        tmp = ti.Vector.zero(ti.f64, valuesT.n)
        for n in range(valuesT.n):
            for m in range(valuesT.m):
                tmp[n] = tmp[n] + valuesT[n,m]
       
        u[i,j,k,0] = ti.f32(tmp[0])
        u[i,j,k,1] = ti.f32(tmp[1])
        u[i,j,k,2] = ti.f32(tmp[2])
    
    return index([nxc, nyc, nzc])