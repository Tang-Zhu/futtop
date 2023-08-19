import taichi as ti
import numpy as np
from indexUtilities import *
from datatypeUtilities import *
from keConstants import *

E = np.float64(1)
Emin = np.float64(1.e-6)

E_f32 =  np.float32(E)
Emin_f32 =  np.float32(Emin)

@ti.func
def getYoungsModule(x :ti.f64) -> ti.f64:
     return (Emin + (E - Emin) * x * x * x)

@ti.func
def getElementYoungsModule(x: nd3f32, elementIndex: index) -> ti.f64:
    res = ti.f64(0.)
    if indexIsInside(x.shape[0], x.shape[1], x.shape[2], elementIndex):
        res = getYoungsModule(getDensityUnsafe(x, elementIndex))
    return res


ti_keconst = ti.field(dtype=ti.f64,shape=(24,24))
ti_keconst.from_numpy(keconst)

@ti.func
def keprod(s:ti.f64,f:vec24f64)->vec24f64:
    res = ti.Vector.zero(dt=ti.f64,n=24)
    for i in range(ti_keconst.shape[0]):
        for j in range(ti_keconst.shape[1]):
            res[i] = res[i] + f[j] * ti_keconst[i,j] * s
    return res