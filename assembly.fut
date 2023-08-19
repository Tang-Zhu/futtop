import "indexUtilities"
-- import "keConstants"
import "boundaryConditions"
import "utility"
import "assemblyUtilities"
import "material"

-- 根据给定的节点权重、单元格索引和自由度编号，计算该单元格精细的贡献值
-- takes a elementindex-weght values pair, and generates the resulting row
-- contribution of the matrix for a given dofNumber.
def getFineValue [nelx][nely][nelz] (x :[nelx][nely][nelz]f32) (dofNumber :i64) (cellIndex :index) (w :nodalWeights) :[24]f64 =
-- 根据输入的模型密度x获取元素的杨氏模量
  let elementScale = getElementYoungsModule x cellIndex
-- 应用边界条件，得到边界上的权重（并乘以1/8），以及域内的权
  let w_onBoundary = applyBoundaryConditionsToWeightsInverse cellIndex w ((nelx+1),(nely+1),(nelz+1)) dofNumber
  let w_inDomain   = applyBoundaryConditionsToWeights        cellIndex w ((nelx+1),(nely+1),(nelz+1)) dofNumber
-- 根据上述权重生成载荷
  let l_onBoundary = generateLoad dofNumber w_onBoundary
  let l_inDomain   = generateLoad dofNumber w_inDomain
-- 使用元素的杨氏模量和域内的载荷来计算刚度矩阵KE的值
  let kevalues = keprod elementScale l_inDomain
-- 将KE的值和边界上的载荷相加，得到最终的精细值
  in #[sequential] map2 (+) kevalues l_onBoundary

def getDiagonalCellContribution [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) (nodeIndex :index) (elementOffset :index) :[3]f64 =
  let cellIndex = addIndices nodeIndex elementOffset
  let nodeOffset = {x=(-elementOffset.x),y=(-elementOffset.y),z=(-elementOffset.z)}
  let li = i64.i32 (getLocalNodeIndex nodeOffset)
  let weights = getInitialWeights nodeOffset
  let valsX = getCoarseCellContribution (getFineValue x 0) l cellIndex weights
  let valsY = getCoarseCellContribution (getFineValue x 1) l cellIndex weights
  let valsZ = getCoarseCellContribution (getFineValue x 2) l cellIndex weights
  in [valsX[3*li+0],valsY[3*li+1],valsZ[3*li+2]]

def getNodeDiagonalValues [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) (nodeIndex :index) :[3]f64 =
   map (getDiagonalCellContribution l x nodeIndex) elementOffsets
   |> transpose
   |> map (reduce (+) 0)

def assembleDiagonal [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) :[][][][3]f64 =
  let ncell = 2**(i64.u8 l)
  let nx = (nelx/ncell)+1
  let ny = (nely/ncell)+1
  let nz = (nelz/ncell)+1
  in #[incremental_flattening(only_inner)]
    tabulate_3d nx ny nz (\i j k -> getNodeDiagonalValues l x {x=i,y=j,z=k})

def assembleInverseDiagonal [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) :[][][][3]f64 =
  assembleDiagonal l x |> map_4d (\x -> 1/x)

-- 提取刚度矩阵的对角线元素
def extractDiagonal [nx][ny][nz] (mat :[nx][ny][nz][3][81]f64) :[nx][ny][nz][3]f64 =
  map_3d (\nodevals ->
  -- 提取对角线上的特定位置的值
    let x = nodevals[0,39]
    let y = nodevals[1,40]
    let z = nodevals[2,41]
    in [x,y,z]
    ) mat

-- 提取对角线元素，计算每个元素的逆
def extractInverseDiagonal [nx][ny][nz] (mat :[nx][ny][nz][3][81]f64) :[nx][ny][nz][3]f64 =
  extractDiagonal mat |> map_4d (\x -> 1/x)

-- 获取给定元素对特定节点的贡献
def getCellContribution [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) (nodeIndex :index) (elementOffset :index) :[3][24]f64 =
-- 根据给定的节点和元素偏移量计算单元格的索引  
  let cellIndex = addIndices nodeIndex elementOffset
  -- 计算节点相对于元素的偏移量
  let nodeOffset = {x=(-elementOffset.x),y=(-elementOffset.y),z=(-elementOffset.z)}
  -- 获取初始权重
  let weights = getInitialWeights nodeOffset
  -- 计算x、y、z方向上的贡献
  let valsX = getCoarseCellContribution (getFineValue x 0) l cellIndex weights
  let valsY = getCoarseCellContribution (getFineValue x 1) l cellIndex weights
  let valsZ = getCoarseCellContribution (getFineValue x 2) l cellIndex weights
  in [valsX,valsY,valsZ]

-- 获取指定节点需要组装的行元素
def getNodeAssembledRow [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) (nodeIndex :index) :[3][81]f64 =
  -- 计算所有元素对给定节点的贡献
  let cellValues  = map (getCellContribution l x nodeIndex) elementOffsets
    |> transpose
    |> map (flatten_to 192)
  -- 根据偏移量，将所有元素贡献值合并为81个行元素
  in map (\v -> #[sequential] reduce_by_index (replicate 81 0f64) (+) 0 elementAssembledOffsets v) cellValues

-- 组装刚度矩阵
def assembleStiffnessMatrix [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) :[][][][3][81]f64 =
  -- 根据给定的层级l计算细分的单元数量
  let ncell = 2**(i64.u8 l)
  -- 计算每个维度上的节点数量
  let nx = (nelx/ncell)+1
  let ny = (nely/ncell)+1
  let nz = (nelz/ncell)+1
  -- 对于每个节点，计算并组装刚度矩阵的行
  in #[incremental_flattening(only_inner)]
    tabulate_3d nx ny nz (\i j k ->etNodeAssembledRow l x {x=i,y=j,z=k})

-- let uoffsets: ([27]i64,[27]i64,[27]i64) =
--   let nodeOffsetsX = [replicate 9 (-1i64), replicate 9 0i64, replicate 9 1i64]
--     |> flatten_to 27
--   let nodeOffsetsY = replicate 3 ([replicate 3 (-1i64), replicate 3 0i64, replicate 3 1i64])
--     |> flatten
--     |> flatten_to 27
--   let nodeOffsetsZ = replicate 3 ( greplicate 3 ([-1,0,1]))
--     |> flatten
--     |> flatten_to 27
--   in (nodeOffsetsX,nodeOffsetsY,nodeOffsetsZ)

def uoffsets: ([27]i64,[27]i64,[27]i64) =
  ([-1i64, -1i64, -1i64, -1i64, -1i64, -1i64, -1i64, -1i64, -1i64, 0i64, 0i64,
  0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64,
  1i64, 1i64, 1i64], [-1i64, -1i64, -1i64, 0i64, 0i64, 0i64, 1i64, 1i64, 1i64,
  -1i64, -1i64, -1i64, 0i64, 0i64, 0i64, 1i64, 1i64, 1i64, -1i64, -1i64, -1i64,
  0i64, 0i64, 0i64, 1i64, 1i64, 1i64], [-1i64, 0i64, 1i64, -1i64, 0i64, 1i64,
  -1i64, 0i64, 1i64, -1i64, 0i64, 1i64, -1i64, 0i64, 1i64, -1i64, 0i64, 1i64,
  -1i64, 0i64, 1i64, -1i64, 0i64, 1i64, -1i64, 0i64, 1i64])

-- 获取当前节点及其相邻节点的位移数据
def getLocalU [nx][ny][nz] (u :[nx][ny][nz][3]f64) (nodeIndex :index) :*[81]f64 =
  let (nodeOffsetsX,nodeOffsetsY,nodeOffsetsZ) = uoffsets
  let ni = nodeIndex
  -- 并行遍历相邻节点，是否位于内部，并获取相应的位移数据
  in (map3 (\i j k ->
    if (indexIsInside (nx,ny,nz) {x=ni.x+i,y=ni.y+j,z=ni.z+k}) then
      #[unsafe] (u[ni.x+i,ni.y+j,ni.z+k])
    else
      [0,0,0]) nodeOffsetsX nodeOffsetsY nodeOffsetsZ)
  -- 展平结果
    |> flatten_to 81

-- 计算刚度矩阵和位移向量的点积
def applyAssembledStiffnessMatrix [nx][ny][nz] (mat :[nx][ny][nz][3][81]f64) (u :[nx][ny][nz][3]f64) =
  -- 并行遍历每个节点
  tabulate_3d nx ny nz (\i j k ->
  -- 获取当前节点及其相邻节点的位移
    let uloc = getLocalU u {x=i,y=j,z=k}
    -- 点积计算
    in map (\row -> (reduce (+) 0 (map2 (\a b -> a*b) uloc row))) (mat[i,j,k])
    )




entry assembleInverseDiagonal_test [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) :[][][][3]f64 =
  assembleInverseDiagonal l x

entry applyAssembledStiffnessMatrix_test [nx][ny][nz] (mat :[nx][ny][nz][3][81]f64) (u :[nx][ny][nz][3]f64) =
  applyAssembledStiffnessMatrix mat u

entry assembleStiffnessMatrix_test [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) :[][][][3][81]f64 =
  assembleStiffnessMatrix l x

entry assembleStiffnessMatrix_fixed1 [nelx][nely][nelz] (x :[nelx][nely][nelz]f32) :[][][][3][81]f64 =
  assembleStiffnessMatrix 1 x

entry assembleStiffnessMatrix_fixed2 [nelx][nely][nelz] (x :[nelx][nely][nelz]f32) :[][][][3][81]f64 =
  assembleStiffnessMatrix 2 x

-- ==
-- entry: assembleInverseDiagonal_test
-- nobench input @../testData/assembleInverseDiagonal1.txt output @../testData/assembleInverseDiagonal1Out.txt
-- nobench input @../testData/assembleInverseDiagonal2.txt output @../testData/assembleInverseDiagonal2Out.txt
-- nobench input @../testData/assembleInverseDiagonal3.txt output @../testData/assembleInverseDiagonal3Out.txt
-- compiled random input { 0u8 [128][128][128]f32 }
-- compiled random input { 1u8 [128][128][128]f32 }
-- compiled random input { 2u8 [128][128][128]f32 }

-- ==
-- entry: applyAssembledStiffnessMatrix_test
-- compiled random input { [129][129][129][3][81]f64 [129][129][129][3]f64 }
-- compiled random input { [65][65][65][3][81]f64 [65][65][65][3]f64 }
-- compiled random input { [33][33][33][3][81]f64 [33][33][33][3]f64 }
-- compiled random input { [17][17][17][3][81]f64 [17][17][17][3]f64 }

-- ==
-- entry: assembleStiffnessMatrix_test
-- compiled random input { 1u8 [128][128][128]f32 }
-- compiled random input { 2u8 [128][128][128]f32 }
-- compiled random input { 3u8 [128][128][128]f32 }
-- compiled random input { 4u8 [128][128][128]f32 }

-- ==
-- entry: assembleStiffnessMatrix_fixed1
-- compiled random input { [128][128][128]f32 }

-- ==
-- entry: assembleStiffnessMatrix_fixed2
-- compiled random input { [128][128][128]f32 }

-- 计算块对角线上单个元素的贡献
def getBlockDiagonalCellContribution [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) (nodeIndex :index) (elementOffset :index) :[3][3]f64 =
  -- 计算单元格索引
  let cellIndex = addIndices nodeIndex elementOffset 
  -- 计算节点偏移
  let nodeOffset = {x=(-elementOffset.x),y=(-elementOffset.y),z=(-elementOffset.z)}
  -- 获取局部节点索引
  let li = i64.i32 (getLocalNodeIndex nodeOffset)
  -- 生成初始权重
  let weights = getInitialWeights nodeOffset
  -- 根据粗网格单元格计算X、Y、Z的贡献
  let valsX = getCoarseCellContribution (getFineValue x 0) l cellIndex weights
  let valsY = getCoarseCellContribution (getFineValue x 1) l cellIndex weights
  let valsZ = getCoarseCellContribution (getFineValue x 2) l cellIndex weights
  -- 提取对应的行数据
  let xrow = valsX[3*li:3*(li+1)] :> [3]f64
  let yrow = valsY[3*li:3*(li+1)] :> [3]f64
  let zrow = valsZ[3*li:3*(li+1)] :> [3]f64
  -- 返回三个方向的行数据
  in [xrow,yrow,zrow]

-- 获取节点的块对角值 对节点每个元素偏移映射获取对角共享并进行规约操作
def getNodeBlockDiagonalValues [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) (nodeIndex :index) :[3][3]f64 =
   map (getBlockDiagonalCellContribution l x nodeIndex) elementOffsets
   |> transpose
   |> map transpose
   |> map (map (reduce (+) 0))

-- 装配所有节点对角线贡献值
def assembleBlockDiagonal [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) :[][][][3][3]f64 =
-- 计算所有节点数
  let ncell = 2**(i64.u8 l)
  let nx = (nelx/ncell)+1
  let ny = (nely/ncell)+1
  let nz = (nelz/ncell)+1
  in #[incremental_flattening(only_inner)]
-- 并行创建每个节点的块对角贡献值
    tabulate_3d nx ny nz (\i j k -> getNodeBlockDiagonalValues l x {x=i,y=j,z=k})

def inv33_f64 (m :[3][3]f64) :[3][3]f64 =
  let a = #[unsafe] m[0,0]
  let b = #[unsafe] m[0,1]
  let c = #[unsafe] m[0,2]
  let d = #[unsafe] m[1,0]
  let e = #[unsafe] m[1,1]
  let f = #[unsafe] m[1,2]
  let g = #[unsafe] m[2,0]
  let h = #[unsafe] m[2,1]
  let i = #[unsafe] m[2,2]
  let det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
  let invdet = 1/det
  in [[e*i-f*h, c*h-b*i, b*f-c*e],
      [f*g-d*i, a*i-c*g, c*d-a*f],
      [d*h-e*g, b*g-a*h, a*e-b*d]]
      |> map_2d (*invdet)

def assembleInverseBlockDiagonal [nelx][nely][nelz] (l :u8) (x :[nelx][nely][nelz]f32) :[][][][3][3]f64 =
  assembleBlockDiagonal l x |> map_3d inv33_f64

def applyBlockDiagonal [nx][ny][nz] (bdiag :[nx][ny][nz][3][3]f64) (u :[nx][ny][nz][3]f64) =
  map2_3d vecmul_f64 bdiag u