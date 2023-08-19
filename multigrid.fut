import "applyStiffnessMatrix"
import "assembly"
import "projection"
import "utility"
import "sor"

-- multigrid网格数据结构（4层）
-- 逆对角线元素和一个3x81的刚度矩阵
type~ mgL3Data = ([][][][3]f64, [][][][3][81]f64)
-- 3x81的刚度矩阵
type~ mgL2Data = [][][][3][81]f64
type~ mgL1Data = {}
type~ mgL0Data = {}
type~ multigridData = (mgL0Data, mgL1Data, mgL2Data, mgL3Data)

-- 生成多重网格
def generateMultigridData [nelx][nely][nelz] (x :[nelx][nely][nelz]f32) :multigridData =
  -- 对第一层生成对角线数据
  -- let d1 = assembleBlockDiagonal 1 x
  -- 对第二层生成刚度矩阵
  let m2 = assembleStiffnessMatrix 2 x
  -- 对第三层生成刚度矩阵
  let m3 = assembleStiffnessMatrix 3 x
  -- 从第三层的刚度矩阵中提取逆对角线数据
  let d3 = extractInverseDiagonal m3
  in ({}, {}, m2, (d3,m3))

def jacobiSmoother [nelx][nely][nelz][nx][ny][nz] (l :u8) (x :[nelx][nely][nelz]f32) (invD :[nx][ny][nz][3]f64) (b :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) :[nx][ny][nz][3]f64 =
  let nsweeps = 4
  let omega   = 0.6
  let smooth u = applyCoarseStiffnessMatrix l x u
    |> map4_4d (\uu bb dd tt -> uu - omega * dd * (tt - bb)) u b invD
  in (iterate nsweeps smooth) u

def jacobiSmootherFine [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (invD :[nx][ny][nz][3]f64) (b :[nx][ny][nz][3]f32) (u :[nx][ny][nz][3]f32) :[nx][ny][nz][3]f32 =
  let nsweeps = 4
  let omega   = 0.6
  let b = map_4d f64.f32 b
  let u = map_4d f64.f32 u
  let smooth u = applyStiffnessMatrix x u
    |> map4_4d (\uu bb dd tt -> uu - omega * dd * (tt - bb)) u b invD
  in (iterate nsweeps smooth) u |> map_4d f32.f64

def jacobiBlockSmoother [nelx][nely][nelz][nx][ny][nz] (l :u8) (x :[nelx][nely][nelz]f32) (bdiag :[nx][ny][nz][3][3]f64) (b :[nx][ny][nz][3]f64) (u :[nx][ny][nz][3]f64) :[nx][ny][nz][3]f64 =
  let nsweeps = 6
  let omega   = 0.6
  let smooth u = applyCoarseStiffnessMatrix l x u
    |> map2_4d (\bb rr -> rr - bb) b
    |> applyBlockDiagonal bdiag
    |> map2_4d (\uold unew -> uold - omega * unew) u
  in (iterate nsweeps smooth) u

-- 计算两个四维矩阵的内积
def innerProduct [n][m][l][k] (a :[n][m][l][k]f64) (b :[n][m][l][k]f64) :f64 =
  map2_4d (*) a b
  |> map_3d f64.sum
  |> map_2d f64.sum
  |> map    f64.sum
  |>        f64.sum

def norm [n][m][l][k] (a :[n][m][l][k]f64) :f64 =
  map_4d (\x -> x*x) a
  |> map_3d f64.sum
  |> map_2d f64.sum
  |> map    f64.sum
  |>        f64.sum
  |> f64.sqrt

-- 使用Jacobi共轭梯度法求解
def cgSolveJacSubspace [nx][ny][nz] (data :multigridData) (b :[nx][ny][nz][3]f64) :[nx][ny][nz][3]f64 =
  let maxIt   = 800 -- 最大迭代次数
  -- 从数据中提取逆对角线矩阵和刚度矩阵
  let invD    = data.3.0 :> [nx][ny][nz][3]f64
  let matrix  = data.3.1 :> [nx][ny][nz][3][81]f64
  -- 创建四维零矩阵
  let zero_4d = replicate nx (replicate ny (replicate nz [0f64,0f64,0f64]))
  -- 定义共轭梯度法的内部迭代函数
  let inner_iteration (uold, rold, pold, rhoold) = 
    let z      = map2_4d (*) invD rold -- jacobi preconditioner
    -- 计算残差
    let rho    = innerProduct rold z
    let beta   = rho / rhoold
    -- 更新方向向量p
    let p      = map2_4d (\pp zz -> beta * pp + zz) pold z
    -- 通过刚度矩阵应用p来获得q
    let q      = applyAssembledStiffnessMatrix matrix p
    -- 计算步长
    let alpha  = rho / (innerProduct p q)
    -- 更新解向量u
    let u      = map2_4d (\uu pp -> uu + alpha * pp) uold p
    -- 更新解向量r
    let r      = map2_4d (\rr qq -> rr - alpha * qq) rold q
    in (u, r, p, rho)

  -- 共轭梯度法迭代
  let  (u, _, _, _) = (iterate maxIt inner_iteration) (zero_4d, b, zero_4d, 1f64)
  in u

-- vcycle_12（L2层计算）
def vcycle_l2 [nx][ny][nz] (data :multigridData) (f :[nx][ny][nz][3]f64) :[nx][ny][nz][3]f64 =
  -- 从L2层获取刚度矩阵
  let matrix = data.2 :> [nx][ny][nz][3][81]f64
  -- 创建四维零矩阵
  let zero_4d = replicate nx (replicate ny (replicate nz [0f64,0f64,0f64]))
  -- 使用SOR方法进行初步近似
  let z = sorAssembled matrix f zero_4d
  -- 使用已有近似值计算误差
  let v = z
    |> applyAssembledStiffnessMatrix matrix -- 应用刚度矩阵到近似值
    |> map2_4d (\ff dd -> ff - dd) f -- 计算残差
    |> projectToCoarser  -- 将残差投影到更粗的网格
    |> cgSolveJacSubspace data -- 在粗网格上求解残差方程
    |> projectToFiner -- 将解决方案投影回细网格
  -- 更新近似值
  let z = map2_4d (+) z (v :> [nx][ny][nz][3]f64)
  -- 再次使用SOR方法进行修正
  in sorAssembled matrix f z

-- def vcycle_l1 [nelx][nely][nelz][nx][ny][nz] (data :multigridData) (x :[nelx][nely][nelz]f32) (f :[nx][ny][nz][3]f64) :[nx][ny][nz][3]f64 =
--   let diag = data.1 :> [nx][ny][nz][3][3]f64
--   let zero_4d = replicate nx (replicate ny (replicate nz [0f64,0f64,0f64]))
--   let z = sorCoarse 1 x diag f zero_4d
--   let v = z
--     |> applyCoarseStiffnessMatrix 1 x
--     |> map2_4d (\ff dd -> ff - dd) f
--     |> projectToCoarser
--     |> vcycle_l2 data
--     |> projectToFiner 
--   let z = map2_4d (+) z (v :> [nx][ny][nz][3]f64)
--   in sorCoarse 1 x diag f z

-- vcycle_10（L0层计算）
def vcycle_l0 [nelx][nely][nelz][nx][ny][nz] (data :multigridData) (x :[nelx][nely][nelz]f32) (f :[nx][ny][nz][3]f32) :[nx][ny][nz][3]f32 =
  -- 创建四维零矩阵
  let zero_4d = replicate nx (replicate ny (replicate nz [0,0,0f32]))
  -- 使用SOR方法在L0层上求解线性系统，并得到近似解z
  let z = sorMatrixFree x f zero_4d
  -- 计算残差v
  let v = z
    |> applyStiffnessMatrixSingle x -- 应用刚度矩阵
    |> map2_4d (\ff dd -> ff - dd) f - 计算r
    |> projectToCoarserSingle -- 将r投影到更粗的网格上
    |> projectToCoarserSingle
    |> map_4d f64.f32
    |> vcycle_l2 data -- 在L2层上计算v
    |> map_4d f32.f64
    |> projectToFinerSingle -- 将v投影回到更细的网格上
    |> projectToFinerSingle

  -- 更新z的值
  let z = map2_4d (+) z (v :> [nx][ny][nz][3]f32)

  -- 使用更新后的z再次使用SOR方法在L0层上求解线性系统
  in sorMatrixFree x f z

