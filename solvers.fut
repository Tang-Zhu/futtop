import "applyStiffnessMatrix"
import "multigrid"
import "utility"

-- added kahan summation from the futhark examples
-- kahan求和算法
type kahan = (f32, f32)

-- 使用Kahan算法进行求和，以减少浮点数的累计误差
def kahan_add ((s1, c1) : kahan) ((s2, c2) : kahan) : kahan =
  let s = s1 + s2
  let d  = if f32.abs s1 >= f32.abs s2 then (s1 - s) + s2
           else (s2 - s) + s1
  let c = (c1 + c2) + d
  in (s, c)

-- 使用Kahan算法对数组进行求和 
def kahan_sum (xs: []f32) : f32 =
  let (s,c) = reduce kahan_add (0,0) (map (\x -> (x,0)) xs)
  in s + c

-- 计算两个四维数组之间的内积
def innerProduct [n][m][l][k] (a :[n][m][l][k]f32) (b :[n][m][l][k]f32) :f32 =
  map2_4d (*) a b
  |> flatten_4d
  |> kahan_sum

-- 计算四维数组的范数
def norm [n][m][l][k] (a :[n][m][l][k]f32) :f32 =
  map_4d (\x -> x*x) a
  |> flatten_4d
  |> kahan_sum
  |> f32.sqrt

-- 使用共轭梯度法和多网格预处理来求解线性方程组
def cgSolveMG [nelx][nely][nelz][nx][ny][nz] (x :[nelx][nely][nelz]f32) (mgData :multigridData) (b :[nx][ny][nz][3]f32) (u :[nx][ny][nz][3]f32) =
  
  -- 设定收敛的阈值和最大迭代次数
  let tol     = 1e-5
  let maxIt   = 200

  -- 计算b的范数
  let bnorm   = norm b

  -- 创建四维零数组
  let zero_4d = replicate_4d nx ny nz 3 0f32

  -- 计算残差r
  let r = applyStiffnessMatrixSingle x u
    |> map2_4d (\bb rr -> bb - rr) b

  -- 使用循环实现共轭梯度法，直到满足收敛条件或超过最大迭代次数
  let  (u, _, _, relres, it, _) =
    loop (uold, rold, pold, res, its, rhoold) = (u, r, zero_4d, 1f32, 0i32, 1f32) while (res > tol && its < maxIt) do
      let z      = vcycle_l0 mgData x rold -- vcycle层求解
      let rho    = innerProduct rold z
      let beta   = rho / rhoold
      let p      = map2_4d (\pp zz -> beta * pp + zz) pold z
      let q      = applyStiffnessMatrixSingle x p -- 应用刚度矩阵
      let alpha  = rho / (innerProduct p q)
      let u      = map2_4d (\uu pp -> uu + alpha * pp) uold p
      let r      = map2_4d (\rr qq -> rr - alpha * qq) rold q
      let relres = (norm r) / bnorm
    in (u, r, p, relres, its+1, rho)
  in (u, relres, it)
