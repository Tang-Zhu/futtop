import "keConstants"
import "indexUtilities"
import "boundaryConditions"
import "projection"
import "utility"
import "material"

-- 计算所有模板结点的贡献值
def getElementContribution [nx][ny][nz][nelx][nely][nelz] (nodeIndex :index, u :[nx][ny][nz][3]f64, x :[nelx][nely][nelz]f32) (elementOffset :index) :[3]f64 =
  let elementIndex = addIndices nodeIndex elementOffset -- 计算参考结点索引
  let ulocal = getLocalState u elementIndex -- 获取结点当前u状态
  let E = getElementYoungsModule x elementIndex -- 获取杨氏模量
  let recievingNodeOffset :index = {x=(-elementOffset.x), y=(-elementOffset.y), z=(-elementOffset.z)} -- 坐标翻转得到刚度矩阵索引
  let li = i64.i32 (getLocalNodeIndex recievingNodeOffset) -- 获取刚度系数
  in map (\ke_row -> (f64.sum (map2(\ke u -> ke*E*u) ke_row ulocal))) keslices[li] -- 计算ke*E*u

-- 应用检测模板，对象为当前节点及其负方向七个顶点
def applyStencilOnNode [nelx][nely][nelz][nx][ny][nz] (nodeIndex :index, u :[nx][ny][nz][3]f64, x :[nelx][nely][nelz]f32) :[3]f64 =
   [{x=( 0),y=( 0),z=( 0)}, {x=(-1),y=( 0),z=( 0)},
    {x=( 0),y=(-1),z=( 0)}, {x=(-1),y=(-1),z=( 0)},
    {x=( 0),y=( 0),z=(-1)}, {x=(-1),y=( 0),z=(-1)},
    {x=( 0),y=(-1),z=(-1)}, {x=(-1),y=(-1),z=(-1)}]
    |> map (\eo -> getElementContribution (nodeIndex, u, x) eo) -- 并行计算贡献值
    |> transpose
    |> map (\x -> f64.sum x) -- 并行计算K(X)*u

-- returns the matrix-vector product K(x)*u 计算应变
#[noinline]
entry applyStiffnessMatrix [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (u :[nx][ny][nz][3]f64) :[nx][ny][nz][3]f64 =
  let f = tabulate_3d nx ny nz (\i j k -> applyStencilOnNode({x=i,y=j,z=k}, u, x)) -- 三维并行计算应变
  in setBCtoInput u f -- 绑定应变与dof坐标

-- returns the matrix-vector product K(x)*u for f32 计算应变（单精度）
def applyStiffnessMatrixSingle [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (u :[nx][ny][nz][3]f32) :[nx][ny][nz][3]f32 =
  map_4d f64.f32 u
  |> applyStiffnessMatrix x
  |> map_4d f32.f64

-- returns the matrix-vector product (P^T*K(x)*P)*u for any subspace 计算粗网格应变
def applyCoarseStiffnessMatrix [nelx][nely][nelz][nxc][nyc][nzc] (l :u8) (x :[nelx][nely][nelz]f32) (u :[nxc][nyc][nzc][3]f64) :[nxc][nyc][nzc][3]f64 =
  let ufine =
    loop u for i < (i16.u8 l) do
      projectToFiner u  --将位移投影到指定细网格

  let rfine = applyStiffnessMatrix x ufine --计算应力

  let rcoarse =
    loop r = rfine for i < (i16.u8 l) do
      projectToCoarser r --将应力映射至粗网格

  in (rcoarse :> [nxc][nyc][nzc][3]f64)

-- ==
-- entry: applyStiffnessMatrix
-- nobench input @../testData/applyStateOperator1.txt output @../testData/applyStateOperator1Output.txt
-- compiled random input { [64][64][64]f32 [65][65][65][3]f64 }
-- compiled random input { [128][128][128]f32 [129][129][129][3]f64 }
-- compiled random input { [256][256][256]f32 [257][257][257][3]f64 }

entry applyCoarseStiffnessMatrix_test0 [nelx][nely][nelz][nxc][nyc][nzc] (x :[nelx][nely][nelz]f32) (u :[nxc][nyc][nzc][3]f64) =
  applyCoarseStiffnessMatrix 0 x u

entry applyCoarseStiffnessMatrix_test1 [nelx][nely][nelz][nxc][nyc][nzc] (x :[nelx][nely][nelz]f32) (u :[nxc][nyc][nzc][3]f64) =
  applyCoarseStiffnessMatrix 1 x u

entry applyCoarseStiffnessMatrix_test2 [nelx][nely][nelz][nxc][nyc][nzc] (x :[nelx][nely][nelz]f32) (u :[nxc][nyc][nzc][3]f64) =
  applyCoarseStiffnessMatrix 2 x u

entry applyCoarseStiffnessMatrix_test3 [nelx][nely][nelz][nxc][nyc][nzc] (x :[nelx][nely][nelz]f32) (u :[nxc][nyc][nzc][3]f64) =
  applyCoarseStiffnessMatrix 3 x u

-- ==
-- entry: applyCoarseStiffnessMatrix_test0
-- compiled random input { [128][128][128]f32 [129][129][129][3]f64 }

-- ==
-- entry: applyCoarseStiffnessMatrix_test1
-- compiled random input { [128][128][128]f32 [65][65][65][3]f64 }

-- ==
-- entry: applyCoarseStiffnessMatrix_test2
-- compiled random input { [128][128][128]f32 [33][33][33][3]f64 }

-- ==
-- entry: applyCoarseStiffnessMatrix_test3
-- compiled random input { [128][128][128]f32 [17][17][17][3]f64 }
