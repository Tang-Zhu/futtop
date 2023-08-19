import "utility"
import "indexUtilities"
import "assembly"
import "boundaryConditions"
import "material"
import "keConstants"
import "applyStiffnessMatrix"

def omega_const         :f64 = 0.6

-- 对局部矩阵进行前向SOR求解
def sorLocalMatrix (omega :f64) (f :[3]f64) (S :[3]f64) (M :[3][3]f64) (u :[3]f64) =
  -- 计算每个维度上的残差
  let rx     = #[unsafe] M[0,1]*u[1] + M[0,2]*u[2]
  let ux_new = #[unsafe] (1/M[0,0]) * (f[0]-S[0]-rx)
  let ry     = #[unsafe] M[1,0]*ux_new + M[1,2]*u[2]
  let uy_new = #[unsafe] (1/M[1,1]) * (f[1]-S[1]-ry)
  let rz     = #[unsafe] M[2,0]*ux_new + M[2,1]*uy_new
  let uz_new = #[unsafe] (1/M[2,2]) * (f[2]-S[2]-rz)
  let unew = [ux_new, uy_new, uz_new]
  in #[sequential] map2 (\un uo -> omega*un + (1-omega)*uo) unew u

-- 对三维局部矩阵进行后向SOR求解
def sorLocalMatrixBack (omega :f64) (f :[3]f64) (S :[3]f64) (M :[3][3]f64) (u :[3]f64) =
  let rz     = #[unsafe] M[2,0]*u[0] + M[2,1]*u[1]
  let uz_new = #[unsafe] (1/M[2,2]) * (f[2]-S[2]-rz)
  let ry     = #[unsafe] M[1,0]*u[0] + M[1,2]*uz_new
  let uy_new = #[unsafe] (1/M[1,1]) * (f[1]-S[1]-ry)
  let rx     = #[unsafe] M[0,1]*uy_new + M[0,2]*uz_new
  let ux_new = #[unsafe] (1/M[0,0]) * (f[0]-S[0]-rx)
  let unew = [ux_new, uy_new, uz_new]
  in #[sequential] map2 (\un uo -> omega*un + (1-omega)*uo) unew u

-- 前向SOR迭代
def sorForwardAssembled (omega :f64) (mat :[3][81]f64) (uStencil :*[81]f64) (f :[3]f64) :*[3]f64 =
  let uold = #[unsafe] [uStencil[39], uStencil[40], uStencil[41]]
  -- 将中心节点置为零，以计算非对角元素与位移向量的乘积
  let uStencil = uStencil with [39] = 0
  let uStencil = uStencil with [40] = 0
  let uStencil = uStencil with [41] = 0
  let S = map (\row -> (f64.sum (map2 (*) uStencil row))) mat
  let M = mat[:3,39:42] :> [3][3]f64
  -- 局部矩阵的SOR求解
  in sorLocalMatrix omega f S M uold

-- 后向SOR迭代
def sorBackwardAssembled (omega :f64) (mat :[3][81]f64) (uStencil :*[81]f64) (f :[3]f64) :*[3]f64 =
  let uold = #[unsafe] [uStencil[39], uStencil[40], uStencil[41]]
  let uStencil = uStencil with [39] = 0
  let uStencil = uStencil with [40] = 0
  let uStencil = uStencil with [41] = 0
  let S = map (\row -> (f64.sum (map2 (*) uStencil row))) mat
  let M = mat[:3,39:42] :> [3][3]f64
  in sorLocalMatrixBack omega f S M uold

def ssorSweepAssembled [nx][ny][nz] (omega :f64) (mat :[nx][ny][nz][3][81]f64) (f :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) =
  -- 并行前向迭代
  let uhalf = tabulate_3d nx ny nz (\i j k ->
    let uloc = getLocalU u {x=i,y=j,z=k}
    in #[unsafe] sorForwardAssembled omega mat[i,j,k] uloc f[i,j,k])
  -- 并行前后迭代  
  let unew = tabulate_3d nx ny nz (\i j k ->
    let uloc = getLocalU uhalf {x=i,y=j,z=k}
    in #[unsafe] sorBackwardAssembled omega mat[i,j,k] uloc f[i,j,k])
  in unew


-- 使用超松弛迭代法SOR逐点法对给定的刚度矩阵进行迭代求解
entry sorAssembled [nx][ny][nz] (mat :[nx][ny][nz][3][81]f64) (f :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) =
  -- 迭代次数
  let number_sweeps :i32 = 2
  in (iterate number_sweeps (ssorSweepAssembled omega_const mat f)) u

entry symGS [nx][ny][nz] (mat :[nx][ny][nz][3][81]f64) (f :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) =
  let number_sweeps :i32 = 1
  in (iterate number_sweeps (ssorSweepAssembled 1.0 mat f)) u


-- let generateNodeOffsetWithoutCenter (eo :index) =
--   let no = getNodeIndices eo
--   let no = filter (\i -> i.x != 0 || i.y != 0 || i.z != 0) no
--   let no = no :> [7]index
--   in zip (replicate 7 eo) no
--
-- let allOffsetPairs =
--   [{x=( 0),y=( 0),z=( 0)}, {x=(-1),y=( 0),z=( 0)},
--    {x=( 0),y=(-1),z=( 0)}, {x=(-1),y=(-1),z=( 0)},
--    {x=( 0),y=( 0),z=(-1)}, {x=(-1),y=( 0),z=(-1)},
--   {x=( 0),y=(-1),z=(-1)}, {x=(-1),y=(-1),z=(-1)}] |> map generateNodeOffsetWithoutCenter |> flatten_to (7*8)

def allOffsetPairs = [
({x = 0i64, y = 0i64, z = 0i64}, {x = 0i64, y = 1i64, z = 0i64}),
({x = 0i64, y = 0i64, z = 0i64}, {x = 1i64, y = 1i64, z = 0i64}),
({x = 0i64, y = 0i64, z = 0i64}, {x = 1i64, y = 0i64, z = 0i64}),
({x = 0i64, y = 0i64, z = 0i64}, {x = 0i64, y = 1i64, z = 1i64}),
({x = 0i64, y = 0i64, z = 0i64}, {x = 1i64, y = 1i64, z = 1i64}),
({x = 0i64, y = 0i64, z = 0i64}, {x = 1i64, y = 0i64, z = 1i64}),
({x = 0i64, y = 0i64, z = 0i64}, {x = 0i64, y = 0i64, z = 1i64}),
({x = -1i64, y = 0i64, z = 0i64},{x = -1i64, y = 1i64, z = 0i64}),
({x = -1i64, y = 0i64, z = 0i64}, {x = 0i64, y = 1i64, z = 0i64}),
({x = -1i64, y = 0i64, z = 0i64},{x = -1i64, y = 0i64, z = 0i64}),
({x = -1i64, y = 0i64, z = 0i64}, {x = -1i64, y = 1i64, z = 1i64}),
({x = -1i64, y = 0i64, z = 0i64}, {x = 0i64, y = 1i64, z = 1i64}),
({x = -1i64, y = 0i64, z = 0i64},{x = 0i64, y = 0i64, z = 1i64}),
({x = -1i64, y = 0i64, z = 0i64}, {x = -1i64, y = 0i64, z = 1i64}),
({x = 0i64, y = -1i64, z = 0i64},{x = 1i64, y = 0i64, z = 0i64}),
({x = 0i64, y = -1i64, z = 0i64}, {x = 1i64, y = -1i64, z = 0i64}),
({x = 0i64, y = -1i64, z = 0i64},{x = 0i64, y = -1i64, z = 0i64}),
({x = 0i64, y = -1i64, z = 0i64}, {x = 0i64, y = 0i64, z = 1i64}),
({x = 0i64, y = -1i64, z = 0i64},{x = 1i64, y = 0i64, z = 1i64}),
({x = 0i64, y = -1i64, z = 0i64}, {x = 1i64, y = -1i64, z = 1i64}),
({x = 0i64, y = -1i64, z = 0i64},{x = 0i64, y = -1i64, z = 1i64}),
({x = -1i64, y = -1i64, z = 0i64}, {x = -1i64, y = 0i64, z = 0i64}),
({x = -1i64, y = -1i64, z = 0i64}, {x = 0i64, y = -1i64, z = 0i64}),
({x = -1i64, y = -1i64, z = 0i64}, {x = -1i64, y = -1i64, z = 0i64}),
({x = -1i64, y = -1i64, z = 0i64}, {x = -1i64, y = 0i64, z = 1i64}),
({x = -1i64, y = -1i64, z = 0i64}, {x = 0i64, y = 0i64, z = 1i64}),
({x = -1i64, y = -1i64, z = 0i64}, {x = 0i64, y = -1i64, z = 1i64}),
({x = -1i64, y = -1i64, z = 0i64}, {x = -1i64, y = -1i64, z = 1i64}),
({x = 0i64, y = 0i64, z = -1i64}, {x = 0i64, y = 1i64, z = -1i64}),
({x = 0i64, y = 0i64, z = -1i64},{x = 1i64, y = 1i64, z = -1i64}),
({x = 0i64, y = 0i64, z = -1i64}, {x = 1i64, y = 0i64, z = -1i64}),
({x = 0i64, y = 0i64, z = -1i64},{x = 0i64, y = 0i64, z = -1i64}),
({x = 0i64, y = 0i64, z = -1i64}, {x = 0i64, y = 1i64, z = 0i64}),
({x = 0i64, y = 0i64, z = -1i64},{x = 1i64, y = 1i64, z = 0i64}),
({x = 0i64, y = 0i64, z = -1i64}, {x = 1i64, y = 0i64, z = 0i64}),
({x = -1i64, y = 0i64, z = -1i64},{x = -1i64, y = 1i64, z = -1i64}),
({x = -1i64, y = 0i64, z = -1i64}, {x = 0i64, y = 1i64, z = -1i64}),
({x = -1i64, y = 0i64, z = -1i64}, {x = 0i64, y = 0i64, z = -1i64}),
({x = -1i64, y = 0i64, z = -1i64}, {x = -1i64, y = 0i64, z = -1i64}),
({x = -1i64, y = 0i64, z = -1i64}, {x = -1i64, y = 1i64, z = 0i64}),
({x = -1i64, y = 0i64, z = -1i64}, {x = 0i64, y = 1i64, z = 0i64}),
({x = -1i64, y = 0i64, z = -1i64}, {x = -1i64, y = 0i64, z = 0i64}),
({x = 0i64, y = -1i64, z = -1i64}, {x = 0i64, y = 0i64, z = -1i64}),
({x = 0i64, y = -1i64, z = -1i64}, {x = 1i64, y = 0i64, z = -1i64}),
({x = 0i64, y = -1i64, z = -1i64}, {x = 1i64, y = -1i64, z = -1i64}),
({x = 0i64, y = -1i64, z = -1i64}, {x = 0i64, y = -1i64, z = -1i64}),
({x = 0i64, y = -1i64, z = -1i64}, {x = 1i64, y = 0i64, z = 0i64}),
({x = 0i64, y = -1i64, z = -1i64},{x = 1i64, y = -1i64, z = 0i64}),
({x = 0i64, y = -1i64, z = -1i64}, {x = 0i64, y = -1i64, z = 0i64}),
({x = -1i64, y = -1i64, z = -1i64}, {x = -1i64, y = 0i64, z = -1i64}),
({x = -1i64, y = -1i64, z = -1i64}, {x = 0i64, y = 0i64, z = -1i64}),
({x = -1i64, y = -1i64, z = -1i64}, {x = 0i64, y = -1i64, z = -1i64}),
({x = -1i64, y = -1i64, z = -1i64}, {x = -1i64, y = -1i64, z = -1i64}),
({x = -1i64, y = -1i64, z = -1i64}, {x = -1i64, y = 0i64, z = 0i64}),
({x = -1i64, y = -1i64, z = -1i64}, {x = 0i64, y = -1i64, z = 0i64}),
({x = -1i64, y = -1i64, z = -1i64}, {x = -1i64, y = -1i64, z = 0i64})]

-- 计算接收和发送节点的索引
def getSendingNode (elementOffset :index, nodeOffset :index) :(i32, i32) =
  -- 计算接收节点的偏移量
  let recievingNodeOffset :index = {x=(-elementOffset.x), y=(-elementOffset.y), z=(-elementOffset.z)}
  -- 计算发送节点的偏移量
  let sendingNodeOffset :index = {x=nodeOffset.x-elementOffset.x, y=nodeOffset.y-elementOffset.y, z=nodeOffset.z-elementOffset.z}
  in (getLocalNodeIndex(recievingNodeOffset), getLocalNodeIndex(sendingNodeOffset))

-- 根据节点索引和偏移获取位移量
def getInputVector [nx][ny][nz] (nodeIndex :index, nodeOffset :index, u :[nx][ny][nz][3]f64) :[3]f64 =
  let loadIndex :index = {x=nodeIndex.x+nodeOffset.x,y=nodeIndex.y+nodeOffset.y,z=nodeIndex.z+nodeOffset.z} in
  --根据边界条件并行获取位移量
  tabulate 3 (\d ->
    if ((indexIsInside (nx,ny,nz) loadIndex) && !(isOnBoundary (nx,ny,nz) loadIndex d)) then
      #[unsafe] (u[loadIndex.x,loadIndex.y,loadIndex.z,d])
    else
      0)

def getInputVectorSingle [nx][ny][nz] (nodeIndex :index, nodeOffset :index, u :[nx][ny][nz][3]f32) :[3]f32 =
  let loadIndex :index = {x=nodeIndex.x+nodeOffset.x,y=nodeIndex.y+nodeOffset.y,z=nodeIndex.z+nodeOffset.z} in
  tabulate 3 (\d ->
    if ((indexIsInside (nx,ny,nz) loadIndex) && !(isOnBoundary (nx,ny,nz) loadIndex d)) then
      #[unsafe] (u[loadIndex.x,loadIndex.y,loadIndex.z,d])
    else
      0)

-- 使用局部矩阵、数组'a'和标量's'执行缩放乘法
def multiplyScaledLocalMatrix(m :localMatrix, a :[3]f64, s :f64) :[3]f64 =
  [(s*(m.xx*a[0]+m.xy*a[1]+m.xz*a[2])),
   (s*(m.yx*a[0]+m.yy*a[1]+m.yz*a[2])),
   (s*(m.zx*a[0]+m.zy*a[1]+m.zz*a[2]))]

-- 对局部矩阵进行缩放
def scaleLocalMatrix(m :localMatrix) (s :f64) :localMatrix =
 {xx=s*m.xx,xy=s*m.xy,xz=s*m.xz,
  yx=s*m.yx,yy=s*m.yy,yz=s*m.yz,
  zx=s*m.zx,zy=s*m.zy,zz=s*m.zz}

-- 两个局部矩阵相加
def addLocalMatrix(a :localMatrix) (b :localMatrix) :localMatrix =
 {xx=a.xx+b.xx,xy=a.xy+b.xy,xz=a.xz+b.xz,
  yx=a.yx+b.yx,yy=a.yy+b.yy,yz=a.yz+b.yz,
  zx=a.zx+b.zx,zy=a.zy+b.zy,zz=a.zz+b.zz}

-- 根据给定的偏移量获取局部矩阵
def getLocalMatrix (elementOffset :index, nodeOffset :index) :localMatrix =
  -- 获取接收和发送节点
  let (recieve,send) = getSendingNode(elementOffset, nodeOffset)
  -- 返回局部矩阵
  in getke_l0 (recieve,send)

-- 获取对矩阵S的贡献
def getSContribution [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (u :[nx][ny][nz][3]f32) (nodeIndex :index) (elementOffset :index, nodeOffset :index) :[3]f64 =
  let elementIndex = addIndices nodeIndex elementOffset -- 计算元素索引
  let elementScale = getElementYoungsModule x elementIndex -- 获取元素的杨氏模量
  let localMatrix  = getLocalMatrix(elementOffset,nodeOffset) -- 获取局部矩阵
  let ulocal       = getInputVectorSingle(nodeIndex,nodeOffset,u) -- 获取位移
  let ulocal       = #[sequential] map f64.f32 ulocal
  in multiplyScaledLocalMatrix(localMatrix, ulocal, elementScale)

-- 构建矩阵S
def build_S [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (u :[nx][ny][nz][3]f32) (nodeIndex :index) :[3]f64 =
  -- 复制所有的偏移对并计算每对贡献
  copy allOffsetPairs
  |> (#[sequential] map (getSContribution x u nodeIndex))
  |> transpose
  |> (#[sequential] map f64.sum)

-- 获取当前元素局部矩阵
def getOwnLocalMatrix (elementOffset :index) =
  let recievingNodeOffset :index = {x=(-elementOffset.x), y=(-elementOffset.y), z=(-elementOffset.z)}
  let li = getLocalNodeIndex(recievingNodeOffset)
  in getke_l0 (li,li)

-- 获取对矩阵M的贡献
def getMContribution [nelx][nely][nelz] (x :[nelx][nely][nelz]f32) (nodeIndex :index) (elementOffset :index) =
  let elementIndex = addIndices nodeIndex elementOffset -- 计算元素索引
  let elementScale = getElementYoungsModule x elementIndex -- 获取元素的杨氏模量
  let localMatrix  = getOwnLocalMatrix elementOffset -- 获取当前元素局部矩阵
  in scaleLocalMatrix localMatrix elementScale

-- 构建矩阵M
def build_M [nelx][nely][nelz] (x :[nelx][nely][nelz]f32) (nodeIndex :index) =
  [{x=( 0),y=( 0),z=( 0)}, {x=(-1),y=( 0),z=( 0)},
   {x=( 0),y=(-1),z=( 0)}, {x=(-1),y=(-1),z=( 0)},
   {x=( 0),y=( 0),z=(-1)}, {x=(-1),y=( 0),z=(-1)},
   {x=( 0),y=(-1),z=(-1)}, {x=(-1),y=(-1),z=(-1)}]
   |> (#[sequential] map (getMContribution x nodeIndex)) -- 获取对矩阵M的贡献
   |> (#[sequential] reduce addLocalMatrix {xx=0,xy=0,xz=0,yx=0,yy=0,yz=0,zx=0,zy=0,zz=0})

-- 使用SOR局部方法计算新的u值
def sorLocal (f :[3]f64) (S :[3]f64) (M :localMatrix) (u :[3]f64) =
  let rx     = #[unsafe] M.xy*u[1] + M.xz*u[2]
  let ux_new = #[unsafe] (1/M.xx) * (f[0]-S[0]-rx)
  let ry     = #[unsafe] M.yx*ux_new + M.yz*u[2]
  let uy_new = #[unsafe] (1/M.yy) * (f[1]-S[1]-ry)
  let rz     = #[unsafe] M.zx*ux_new + M.zy*uy_new
  let uz_new = #[unsafe] (1/M.zz) * (f[2]-S[2]-rz)
  in [ux_new, uy_new, uz_new]

-- 使用SOR局部方法的后向扫描计算新的u值
def sorLocalBack (f :[3]f64) (S :[3]f64) (M :localMatrix) (u :[3]f64) =
  let rz     = #[unsafe] M.zx*u[0] + M.zy*u[1]
  let uz_new = #[unsafe] (1/M.zz) * (f[2]-S[2]-rz)
  let ry     = #[unsafe] M.yx*u[0] + M.yz*uz_new
  let uy_new = #[unsafe] (1/M.yy) * (f[1]-S[1]-ry)
  let rx     = #[unsafe] M.xy*uy_new + M.xz*uz_new
  let ux_new = #[unsafe] (1/M.xx) * (f[0]-S[0]-rx)
  in [ux_new, uy_new, uz_new]

-- 执行前向SOR扫描的节点级运算
def sorNodeForward  [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (u :[nx][ny][nz][3]f32) (f :[3]f32) (nodeIndex :index) :[3]f32 =
-- extract value of own node, and zero
 let ni = nodeIndex
-- 从当前节点提取旧值
 let ux_old = #[unsafe] f64.f32 u[ni.x,ni.y,ni.z,0]
 let uy_old = #[unsafe] f64.f32 u[ni.x,ni.y,ni.z,1]
 let uz_old = #[unsafe] f64.f32 u[ni.x,ni.y,ni.z,2]
 let uold = [ux_old, uy_old, uz_old]
-- 构建S和M矩阵
 let f = #[sequential] map f64.f32 f
 let S = build_S x u ni
 let M = build_M x ni
-- 使用SOR局部方法计算新的节点值
 let utmp = sorLocal f S M uold
 -- 平滑化计算的值
 let usmoothed = #[sequential] map2 (\un uo -> omega_const*un + (1-omega_const)*uo) utmp uold
 in #[sequential] map f32.f64 usmoothed

-- 执行后向SOR扫描的节点级运算（同上）
def sorNodeBackward [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (u :[nx][ny][nz][3]f32) (f :[3]f32) (nodeIndex :index) :[3]f32 =
 -- extract value of own node, and zero
 let ni = nodeIndex

 let ux_old = #[unsafe] f64.f32 u[ni.x,ni.y,ni.z,0]
 let uy_old = #[unsafe] f64.f32 u[ni.x,ni.y,ni.z,1]
 let uz_old = #[unsafe] f64.f32 u[ni.x,ni.y,ni.z,2]
 let uold = [ux_old, uy_old, uz_old]

 let f = #[sequential] map f64.f32 f
 let S = build_S x u ni
 let M = build_M x ni

 let utmp = sorLocalBack f S M uold
 let usmoothed = #[sequential] map2 (\un uo -> omega_const*un + (1-omega_const)*uo) utmp uold
 in #[sequential] map f32.f64 usmoothed

-- 单次SSOR扫描
def ssorSweep [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (f :[nx][ny][nz][3]f32) (u :[nx][ny][nz][3]f32) =
  -- 向前扫描
  let uhalf = tabulate_3d nx ny nz (\i j k -> sorNodeForward x u f[i,j,k] {x=i,y=j,z=k})
  -- 设置边界条件为0
  let uhalf = setBCtoZero 0 uhalf
  -- 向后扫描
  let unew = tabulate_3d nx ny nz (\i j k -> sorNodeBackward x uhalf f[i,j,k] {x=i,y=j,z=k})
  -- 设置边界条件为0
  in setBCtoZero 0 unew

-- 使用SOR方法在给定的x上进行迭代求解
entry sorMatrixFree [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (f :[nx][ny][nz][3]f32) (u :[nx][ny][nz][3]f32) =
  -- 设置扫描次数
  let number_sweeps :i32 = 1
  -- 使用SSOR扫描进行迭代
  in (iterate number_sweeps (ssorSweep x f)) u

-- entry jacobiReference [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (b :[nx][ny][nz][3]f32) (u :[nx][ny][nz][3]f32) =
--  let invD = assembleInverseDiagonal 0 x |> map_4d f32.f64 :> [nx][ny][nz][3]f32
--  in jacobiSmootherSingle x b invD u

-- ==
-- entry: sorMatrixFree
-- nobench input @../testData/sor1.txt auto output
-- nobench input @../testData/sor1.txt output @../testData/sor1Out.txt
-- nobench input @../testData/sor2.txt output @../testData/sor2Out.txt
-- compiled random input { [64][64][64]f32 [65][65][65][3]f32 [65][65][65][3]f32 }
-- compiled random input { [128][128][128]f32 [129][129][129][3]f32 [129][129][129][3]f32 }
-- compiled random input { [256][256][256]f32 [257][257][257][3]f32 [257][257][257][3]f32 }

def sorCoarseForward [nelx][nely][nelz][nx][ny][nz] (l :u8) (x :[nelx][nely][nelz]f32) (bdiag :[nx][ny][nz][3][3]f64) (f :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) = 
  let Ku = applyCoarseStiffnessMatrix l x u
  let Kd = map2_3d vecmul_f64 bdiag u
  let S  = map2_4d (-) Ku Kd
  let unew = map4_3d (sorLocalMatrix omega_const) f S bdiag u
  in setBCtoZero 0 unew

def sorCoarseBackward [nelx][nely][nelz][nx][ny][nz] (l :u8) (x :[nelx][nely][nelz]f32) (bdiag :[nx][ny][nz][3][3]f64) (f :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) = 
  let Ku = applyCoarseStiffnessMatrix l x u
  let Kd = map2_3d vecmul_f64 bdiag u
  let S  = map2_4d (-) Ku Kd
  let unew = map4_3d (sorLocalMatrixBack omega_const) f S bdiag u
  in setBCtoZero 0 unew

def sorCoarseSweep [nelx][nely][nelz][nx][ny][nz] (l :u8) (x :[nelx][nely][nelz]f32) (bdiag :[nx][ny][nz][3][3]f64) (f :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) = 
  let uhalf = sorCoarseForward l x bdiag f u
  in sorCoarseBackward l x bdiag f uhalf

def sorCoarse [nelx][nely][nelz][nx][ny][nz] (l :u8) (x :[nelx][nely][nelz]f32) (bdiag :[nx][ny][nz][3][3]f64) (f :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) =
  let sweep v = sorCoarseSweep l x bdiag f v
  let number_sweeps :i32 = 1
  in (iterate number_sweeps sweep) u


