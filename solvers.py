import taichi as ti
from datatypeUtilities import *
from applyStiffnessMatrix import *
from multigrid import*

@ti.func
def norm(a:nd4f64)->ti.f32:
    res = 0.
    for idx in ti.grouped(a):
        res = res + a[idx] * a[idx]
    return ti.math.sqrt(res)

# m0 : mgL0Data, m1 : mgL1Data, m2 : mgL2Data, m3n4 : mgL3DataDim4, m3n5 : mgL3DataDim5, x :nd3f32, f: nd4f32, tmp10:nd4f64,  boundaries : index, tmp12:nd4f64, z10: nd4f32,  v:nd4f32,z12: nd4f64,r:nd4f64,p:nd4f64,q:nd4f64
@ti.kernel
def cgSolveMG(m0 : mgL0Data, m1 : mgL1Data, m2 : mgL2Data, m3n4 : mgL3DataDim4, m3n5 : mgL3DataDim5,x:nd4f64,b:nd4f64,u:nd4f64,f:nd4f64, z_10:nd4f64,  boundaries : index, res: nd4f64,  v:nd4f64, z: nd4f64,u_10:nd4f64,r:nd4f64,p:nd4f64,q:nd4f64,pold:nd4f64):
    tol = 1.e-5
    maxIt = 200
    bnorm = norm(b)

    for idx in ti.grouped(pold):
        pold[idx]=0

    applyStiffnessMatrix(x,u,f)

    for idx in ti.grouped(f):
        f[idx] = b[idx] - f[idx]

    rhoold = 1
    relres = 1
    for its in maxIt:
        if relres <= tol:
            break
        vcycle_l0(m0,m1,m2,m3n4,m3n5,x,f,z_10,v,res,boundaries,z,u_10,r,p,q)
        rho = innerProduct(f,res)
        beta = rho / rhoold
        for idx in ti.grouped(pold):
            pold[idx]=beta*pold[idx]+res[idx]
        applyStiffnessMatrix(x,pold,q)
        alpha = rho/innerProduct(pold,q)
        for idx in ti.grouped(u):
            u[idx]=u[idx]+alpha*pold[idx]
        for idx in ti.grouped(f):
            f[idx]=f[idx]-alpha*q[idx]
        rhoold = rho
        relres = norm(f) / bnorm
    return (relres, its)