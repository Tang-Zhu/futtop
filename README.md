# futtop
 futtop-code

此库为futtop程序主要代码

futtop.c 为主程序入口
io.h io.c 为输入输出
libmultigrid.fut 定义了C与futhark之间的程序接口

其余fut文件为fuhark版本的主代码，py文件为taichi版本的对应代码，除部分函数为区分CPU与GPU采用了ti_作为前缀命名，其余函数命名相同。
