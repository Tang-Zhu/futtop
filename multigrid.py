import taichi as ti
from datatypeUtilities import *
from assembly import *
from sor import *
from applyStiffnessMatrix import *

@ti.func
def innerProduct(a:nd4f64,b:nd4f64)->ti.f64:
    sum0 = ti.Vector.zero(ti.f64,a.shape[0]) 
    for i in range(a.shape[0]):
        sum1 = ti.Vector.zero(ti.f64,a.shape[1])
        for j in range(a.shape[1]):
            sum2 = ti.Vector.zero(ti.f64,a.shape[2])
            for k in range(a.shape[2]): 
                for n in range(a.shape[3]):
                    sum2[k] = sum2[k] + a[i,j,k,n] * b[i,j,k,n]
            for k in range(a.shape[2]):
                sum1[j] = sum1[j] + sum2[k]
        for j in range(a.shape[1]):
            sum0[i] = sum0[i] + sum1[j]
    sum=0.
    for i in range(a.shape[0]):
        sum+=sum0[i]
    return sum
     

@ti.func
def cgSolveJacSubspace(m0 : mgL0Data, m1 : mgL1Data, m2 : mgL2Data, m3n4 : mgL3DataDim4, m3n5 : mgL3DataDim5, b :nd4f64, z:nd4f64,u:nd4f64,r:nd4f64,p:nd4f64,q:nd4f64):
    maxIt = 800
    rho = 1.
    for idx in ti.grouped(p):
        r[idx] = b[idx]
        z[idx],u[idx],p[idx],q[idx] = 0
        

    for _ in range(maxIt):
        for idx in ti.grouped(z):
            z[idx] = m3n4[idx] * r[idx]
        tmp = rho
        rho = innerProduct(r,z)
        beta = rho / tmp
        for idx in ti.grouped(p):
            p[idx] = p[idx] * beta + z[idx]
        applyAssembledStiffnessMatrix(m3n5,p,q)
        alpha  = rho / (innerProduct(p, q))
        for idx in ti.grouped(u):
            u[idx] = u[idx] + alpha * p[idx]
        for idx in ti.grouped(r):
            r[idx] = r[idx] - alpha * p[idx]


# res size（nx,ny,nz） depend on x size and l:
# ncell = 2**l
# nx = ti.int32(x.shape[0]/ncell+1)
# ny = ti.int32(x.shape[1]/ncell+1)
# nz = ti.int32(x.shape[2]/ncell+1)
@ti.kernel
def generateMultigridData(x : nd3f32, m0 : mgL0Data, m1 : mgL1Data, m2 : mgL2Data, m3n4 : mgL3DataDim4, m3n5 : mgL3DataDim5):
    assembleStiffnessMatrix(2,x,m2)
    assembleStiffnessMatrix(3,x,m3n5)
    extractInverseDiagonal(m3n5, m3n4)


@ti.func
def vcycle_l2(m0 : mgL0Data, m1 : mgL1Data, m2 : mgL2Data, m3n4 : mgL3DataDim4, m3n5 : mgL3DataDim5, f: nd4f64, v:nd4f64, res:nd4f64, boundaries:index, z:nd4f64,u:nd4f64,r:nd4f64,p:nd4f64,q:nd4f64):
    # z = res
    sorAssembled(m2,f,v,res)
    applyAssembledStiffnessMatrix(m2,v,res)
    for idx in ti.grouped(f):
        res[idx] = f[idx] - res[idx] 
    boundaries = projectToCoarser(res,boundaries)
    cgSolveJacSubspace(m0, m1, m2, m3n4, m3n5, res, z,u,r,p,q)
    boundaries = projectToCoarser(u,boundaries)
    for idx in ti.grouped(u):
        res[idx] = res[idx]+u[idx]
    sorAssembled(m2,f,res,v)

@ti.func
def vcycle_l0(m0 : mgL0Data, m1 : mgL1Data, m2 : mgL2Data, m3n4 : mgL3DataDim4, m3n5 : mgL3DataDim5, x :nd4f64, f: nd4f64,  z_10: nd4f64,v:nd4f64, res:nd4f64, boundaries : index,z: nd4f64,u:nd4f64,r:nd4f64,p:nd4f64,q:nd4f64):
    sorMatrixFree(x,f,z_10,v)
    for idx in ti.grouped(z_10):
        v[idx]=z_10[idx]
    applyStiffnessMatrix(x,v,res)
    for idx in ti.grouped(res):
        f[idx] = f[idx] - res[idx]
    boundaries = projectToCoarser(f, boundaries)
    boundaries = projectToCoarser(f, boundaries)
    
    vcycle_l2(m0, m1, m2, m3n4, m3n5, f, v, res, boundaries, z,u,r,p,q)

    boundaries = projectToFiner(res, boundaries)
    boundaries = projectToFiner(res, boundaries)

    for idx in ti.grouped(z_10):
        res[idx] = res[idx]+z_10[idx]
    sorMatrixFree(x,f,res,z_10)
