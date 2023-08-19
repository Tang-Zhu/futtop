import taichi as ti
import numpy as np
from datatypeUtilities import *
from indexUtilities import *
from material import *
from keConstants import *
from assembly import *

omega_const :ti.f64 = 0.6

@ti.func
def getSendingNode(elementOffset :index, nodeOffset :index):
    recievingNodeOffset = index([-elementOffset.x,-elementOffset.y,-elementOffset.z])
    sendingNodeOffset = index([nodeOffset.x-elementOffset.x, nodeOffset.y-elementOffset.y, nodeOffset.z-elementOffset.z])
    return (getLocalNodeIndex(recievingNodeOffset), getLocalNodeIndex(sendingNodeOffset))

@ti.func
def getLocalMatrix(elementOffset :index, nodeOffset :index)->lclmt:
    recieve,send = getSendingNode(elementOffset,nodeOffset)
    return getke_l0(recieve,send)

@ti.func
def getInputVector(nodeIndex :index, nodeOffset :index, u : nd4f32)->vec3f64:
    loadIndex = index([nodeIndex.x+nodeOffset.x,nodeIndex.y+nodeOffset.y,nodeIndex.z+nodeOffset.z])
    nx = u.shape[0]
    ny = u.shape[1]
    nz = u.shape[2]
    res = vec3f64([0,0,0])
    for d in range(3):
        if indexIsInside(nx,ny,nz,loadIndex) and not isOnBoundary(nx,ny,nz,loadIndex,d):
            res[d] = u[loadIndex.x,loadIndex.y,loadIndex.z,d]
    return res

@ti.func
def multiplyScaledLocalMatrix(m:lclmt,a:vec3f64,s:ti.f64)->vec3f64:
    return vec3f64([(s*(m.xx*a[0]+m.xy*a[1]+m.xz*a[2])),
                    (s*(m.yx*a[0]+m.yy*a[1]+m.yz*a[2])),
                    (s*(m.zx*a[0]+m.zy*a[1]+m.zz*a[2]))])

@ti.func
def getSContribution(x:nd3f32, u:nd4f32,nodeIndex:index,elementOffset :index, nodeOffset :index)->vec3f64:
    elementIndex = addIndices(nodeIndex,elementOffset)
    elementScale = getElementYoungsModule(x,elementIndex)
    localMatrix = getLocalMatrix(elementOffset,nodeOffset)
    ulocal = vec3f64(getInputVector(nodeIndex,nodeOffset,u))
    return multiplyScaledLocalMatrix(localMatrix,ulocal,elementScale)

@ti.func
def getOwnLocalMatrix(elementOffset :index)->lclmt:
    recievingNodeOffset = index(-elementOffset.x, -elementOffset.y, -elementOffset.z)
    li = getLocalNodeIndex(recievingNodeOffset)
    return getke_l0(li, li)

@ti.func
def scaleLocalMatrix(m:lclmt,s:ti.f64)->lclmt:
    return lclmt(xx=s*m.xx,xy=s*m.xy,xz=s*m.xz,
                yx=s*m.yx,yy=s*m.yy,yz=s*m.yz,
                zx=s*m.zx,zy=s*m.zy,zz=s*m.zz)

@ti.func
def getMContribution(x:nd3f32,nodeIndex :index, elementOffset :index)->lclmt:
    elementIndex = addIndices(nodeIndex,elementOffset)
    elementScale = getElementYoungsModule(x,elementIndex)
    localMatrix = getOwnLocalMatrix(elementOffset)
    return scaleLocalMatrix(localMatrix,elementScale)
    

@ti.func
def build_S(x:nd3f32, u:nd4f32,nodeIndex:index)->vec3f64:
    d1:ti.f64 = 0.
    d2:ti.f64 = 0.
    d3:ti.f64 = 0.
    for i in range(ti_allOffsetPairs.shape[0]):
        con = getSContribution(x,u,nodeIndex,index([ti_allOffsetPairs[i,0,0],ti_allOffsetPairs[i,0,1],ti_allOffsetPairs[i,0,2]]),index([ti_allOffsetPairs[i,1,0],ti_allOffsetPairs[i,1,1],ti_allOffsetPairs[i,1,2]]))
        d1= d1 + con[0]
        d2= d2 + con[1]
        d3= d3 + con[2]
    return vec3f64([d1,d2,d3])

@ti.func
def addLocalMatrix(a:lclmt,b:lclmt)->lclmt:
    return lclmt(xx=a.xx+b.xx,xy=a.xy+b.xy,xz=a.xz+b.xz,
                yx=a.yx+b.yx,yy=a.yy+b.yy,yz=a.yz+b.yz,
                zx=a.zx+b.zx,zy=a.zy+b.zy,zz=a.zz+b.zz)

@ti.func
def build_M(x:nd3f32,nodeIndex:index):
    elementOffsets =  ti.Matrix([[0, 0, 0], [-1, 0, 0],
                        [0,-1, 0], [-1,-1, 0],
                        [0, 0,-1], [-1, 0,-1],
                        [0,-1,-1], [-1,-1,-1]])
    res = lclmt(0,0,0,0,0,0,0,0,0)
    for i in range(elementOffsets.n):
        tmp = getMContribution(x,nodeIndex,elementOffsets[i,:])
        res = res + tmp
    return res

@ti.func
def sorLocal(f:vec3f64,S:vec3f64,M:lclmt,u:vec3f64)->vec3f64:
  rx     = M.xy*u[1] + M.xz*u[2]
  ux_new = (1/M.xx) * (f[0]-S[0]-rx)
  ry     = M.yx*ux_new + M.yz*u[2]
  uy_new = (1/M.yy) * (f[1]-S[1]-ry)
  rz     = M.zx*ux_new + M.zy*uy_new
  uz_new = (1/M.zz) * (f[2]-S[2]-rz)
  return vec3f64([ux_new, uy_new, uz_new])

@ti.func
def sorNodeForward(x:nd3f32, u:nd4f32, f:vec3f32, nodeIndex:index)->vec3f32:
    ni = nodeIndex
    ux_old = ti.f64(u[ni.x,ni.y,ni.z,0])
    uy_old = ti.f64(u[ni.x,ni.y,ni.z,1])
    uz_old = ti.f64(u[ni.x,ni.y,ni.z,2])
    uold = ti.Vector([ux_old,uy_old,uz_old])

    S = build_S(x,u,ni)
    M = build_M(x,ni)

    ff = ti.Vector([f[0],f[1],f[2]],dt=ti.f64)

    utmp = sorLocal(ff,S,M,uold)
    usmoothed = vec3f64([0,0,0])
    for i in range(usmoothed.n):
        usmoothed[i] = omega_const*utmp[i] + (1-omega_const)*uold[i]
    return usmoothed

@ti.func
def sorLocalBack(f:vec3f64,S:vec3f64,M:lclmt,u:vec3f64)->vec3f64:
    rz     = M.zx*u[0] + M.zy*u[1]
    uz_new = (1/M.zz) * (f[2]-S[2]-rz)
    ry     = M.yx*u[0] + M.yz*uz_new
    uy_new = (1/M.yy) * (f[1]-S[1]-ry)
    rx     = M.xy*uy_new + M.xz*uz_new
    ux_new = (1/M.xx) * (f[0]-S[0]-rx)
    return vec3f64([ux_new, uy_new, uz_new])

@ti.func
def sorNodeBackward(x:nd3f32,u:nd4f32,f:vec3f32,nodeIndex:index)->vec3f32:
    ni = nodeIndex

    ux_old = ti.f64(u[ni.x,ni.y,ni.z,0])
    uy_old = ti.f64(u[ni.x,ni.y,ni.z,1])
    uz_old = ti.f64(u[ni.x,ni.y,ni.z,2])
    uold = ti.Vector([ux_old, uy_old, uz_old])

    S = build_S(x,u,ni)
    M = build_M(x,ni)

    ff = ti.Vector([f[0],f[1],f[2]],dt=ti.f64)

    utmp = sorLocalBack(ff,S,M,uold)

    usmoothed = vec3f64([0,0,0])
    for i in range(usmoothed.n):
        usmoothed[i] = omega_const*utmp[i] + (1-omega_const)*uold[i]
    return usmoothed


@ti.func
def ssorSweep(x:nd3f32, f:nd4f32, u:nd4f32, uhalf:nd4f32):
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]

    for i,j,k in ti.ndrange(nx,ny,nz):
        tmp = sorNodeForward(x,u,vec3f64([f[i,j,k,0],f[i,j,k,1],f[i,j,k,2]]),index([i,j,k]))
        uhalf[i,j,k,0]=tmp[0]
        uhalf[i,j,k,1]=tmp[1]
        uhalf[i,j,k,2]=tmp[2]
        
    setBCtoZero(0,uhalf)

    for i,j,k in ti.ndrange(nx,ny,nz):
        tmp = sorNodeBackward(x,uhalf,vec3f64([f[i,j,k,0],f[i,j,k,1],f[i,j,k,2]]),index([i,j,k]))
        u[i,j,k,0]=tmp[0]
        u[i,j,k,1]=tmp[1]
        u[i,j,k,2]=tmp[2]
    
    setBCtoZero(0,u)

    
@ti.func
def sorMatrixFree(x:nd3f32, f:nd4f32, u:nd4f32, uhalf:nd4f32):
    number_sweeps : ti.i32= 1
    for _ in range(number_sweeps):
        ssorSweep(x,f,u,uhalf)

@ti.func
def sorLocalMatrix(omega :ti.f64, f : vec3f64, S :vec3f64, M :n3m3f64, u :vec3f64)->vec3f64:
    rx     = M[0,1]*u[1] + M[0,2]*u[2]
    ux_new = (1/M[0,0]) * (f[0]-S[0]-rx)
    ry     = M[1,0]*ux_new + M[1,2]*u[2]
    uy_new = (1/M[1,1]) * (f[1]-S[1]-ry)
    rz     = M[2,0]*ux_new + M[2,1]*uy_new
    uz_new = (1/M[2,2]) * (f[2]-S[2]-rz)
    unew = vec3f64([ux_new, uy_new, uz_new])
    return vec3f64([omega*unew[i] + (1-omega)*u[i] for i in range(3)])

@ti.func
def sorLocalMatrixBack(omega :ti.f64, f : vec3f64, S :vec3f64, M :n3m3f64, u :vec3f64)->vec3f64:
    rz     = M[2,0]*u[0] + M[2,1]*u[1]
    uz_new = (1/M[2,2]) * (f[2]-S[2]-rz)
    ry     = M[1,0]*u[0] + M[1,2]*uz_new
    uy_new = (1/M[1,1]) * (f[1]-S[1]-ry)
    rx     = M[0,1]*uy_new + M[0,2]*uz_new
    ux_new = (1/M[0,0]) * (f[0]-S[0]-rx)
    unew = vec3f64([ux_new, uy_new, uz_new])
    return vec3f64([omega*unew[i] + (1-omega)*u[i] for i in range(3)])   

@ti.func
def sorForwardAssembled(omega :ti.f64,mat :n3m81f64, uStencil : vec81f64, f :vec3f64)->vec3f64:
    uold = vec3f64([uStencil[39],uStencil[40],uStencil[41]])
    uStencil[39],uStencil[40],uStencil[41]=0,0,0
    S = vec3f64([0,0,0])
    for i in range(mat.n):
        sum = 0.
        for j in range(mat.m):
            sum = sum + uStencil[j] * mat[i,j]
        S[i] = sum

    M = ti.Matrix.zero(ti.f64,3,3)
    for i in range(3):
        M[i,0] = mat[i,39]
        M[i,1] = mat[i,40]
        M[i,2] = mat[i,41]

    return sorLocalMatrix(omega,f,S,M,uold)


@ti.func
def sorBackwardAssembled(omega :ti.f64,mat :n3m81f64, uStencil : vec81f64, f :vec3f64)->vec3f64:
    uold = vec3f64([uStencil[39],uStencil[40],uStencil[41]])
    uStencil[39],uStencil[40],uStencil[41]=0,0,0
    S = vec3f64([0,0,0])
    for i in range(mat.n):
        sum = 0.
        for j in range(mat.m):
            sum = sum + uStencil[j] * mat[i,j]
        S[i] = sum

    M = ti.Matrix.zero(ti.f64,3,3)
    for i in range(3):
        M[i,0] = mat[i,39]
        M[i,1] = mat[i,40]
        M[i,2] = mat[i,41]

    return sorLocalMatrixBack(omega,f,S,M,uold)


@ti.func
def ssorSweepAssembled(omega :ti.f64,mat:nd5f64,f:nd4f64,u:nd4f64,uhalf:nd4f64):
    nx = u.shape[0]
    ny = u.shape[1]
    nz = u.shape[2]
    for i,j,k in ti.ndrange(nx,ny,nz):
        uloc = getLocalU(u,index([i,j,k]))
        mloc = n3m81f64([[mat[i,j,k,n,m] for m in range(mat.shape[4])] for n in range(mat.shape[3])])
        floc = vec3f64([f[i,j,k,n] for n in range(f.shape[3])])
        tmp = sorForwardAssembled(omega,mloc,uloc,floc)
        uhalf[i,j,k,0]=tmp[0]
        uhalf[i,j,k,1]=tmp[1]
        uhalf[i,j,k,2]=tmp[2]

    for i,j,k in ti.ndrange(nx,ny,nz):
        uloc = getLocalU(uhalf,index([i,j,k]))
        mloc = n3m81f64([[mat[i,j,k,n,m] for m in range(mat.shape[4])] for n in range(mat.shape[3])])
        floc = vec3f64([f[i,j,k,n] for n in range(f.shape[3])])
        tmp = sorBackwardAssembled(omega,mloc,uloc,floc)
        u[i,j,k,0]=tmp[0]
        u[i,j,k,1]=tmp[1]
        u[i,j,k,2]=tmp[2]


@ti.func
def sorAssembled(mat:nd5f64,f:nd4f64,u:nd4f64,uhalf:nd4f64):
    number_sweeps : ti.i32 = 2
    for _ in range(number_sweeps):
        ssorSweepAssembled(omega_const,mat,f,u,uhalf)


allOffsetPairs = np.array([
[[0, 0, 0], [0, 1, 0]],
[[0, 0, 0], [1, 1, 0]],
[[0, 0, 0], [1, 0, 0]],
[[0, 0, 0], [0, 1, 1]],
[[0, 0, 0], [1, 1, 1]],
[[0, 0, 0], [1, 0, 1]],
[[0, 0, 0], [0, 0, 1]],
[[-1, 0, 0],[-1, 1, 0]],
[[-1, 0, 0], [0, 1, 0]],
[[-1, 0, 0],[-1, 0, 0]],
[[-1, 0, 0], [-1, 1, 1]],
[[-1, 0, 0], [0, 1, 1]],
[[-1, 0, 0],[0, 0, 1]],
[[-1, 0, 0], [-1, 0, 1]],
[[0, -1, 0],[1, 0, 0]],
[[0, -1, 0], [1, -1, 0]],
[[0, -1, 0],[0, -1, 0]],
[[0, -1, 0], [0, 0, 1]],
[[0, -1, 0],[1, 0, 1]],
[[0, -1, 0], [1, -1, 1]],
[[0, -1, 0],[0, -1, 1]],
[[-1, -1, 0], [-1, 0, 0]],
[[-1, -1, 0], [0, -1, 0]],
[[-1, -1, 0], [-1, -1, 0]],
[[-1, -1, 0], [-1, 0, 1]],
[[-1, -1, 0], [0, 0, 1]],
[[-1, -1, 0], [0, -1, 1]],
[[-1, -1, 0], [-1, -1, 1]],
[[0, 0, -1], [0, 1, -1]],
[[0, 0, -1],[1, 1, -1]],
[[0, 0, -1], [1, 0, -1]],
[[0, 0, -1],[0, 0, -1]],
[[0, 0, -1], [0, 1, 0]],
[[0, 0, -1],[1, 1, 0]],
[[0, 0, -1], [1, 0, 0]],
[[-1, 0, -1],[-1, 1, -1]],
[[-1, 0, -1], [0, 1, -1]],
[[-1, 0, -1], [0, 0, -1]],
[[-1, 0, -1], [-1, 0, -1]],
[[-1, 0, -1], [-1, 1, 0]],
[[-1, 0, -1], [0, 1, 0]],
[[-1, 0, -1], [-1, 0, 0]],
[[0, -1, -1], [0, 0, -1]],
[[0, -1, -1], [1, 0, -1]],
[[0, -1, -1], [1, -1, -1]],
[[0, -1, -1], [0, -1, -1]],
[[0, -1, -1], [1, 0, 0]],
[[0, -1, -1],[1, -1, 0]],
[[0, -1, -1], [0, -1, 0]],
[[-1, -1, -1], [-1, 0, -1]],
[[-1, -1, -1], [0, 0, -1]],
[[-1, -1, -1], [0, -1, -1]],
[[-1, -1, -1], [-1, -1, -1]],
[[-1, -1, -1], [-1, 0, 0]],
[[-1, -1, -1], [0, -1, 0]],
[[-1, -1, -1], [-1, -1, 0]]])

ti_allOffsetPairs = ti.field(dtype=ti.i32,shape=(56,2,3))
ti_allOffsetPairs.from_numpy(allOffsetPairs)