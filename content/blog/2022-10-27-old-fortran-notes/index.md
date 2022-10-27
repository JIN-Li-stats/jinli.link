---
title: 陈旧的 Fortran 笔记
author: ''
date: '2022-10-27'
slug: old-fortran-notes
toc: true
comments: true
categories:
  - 日志
tags:
  - Simulation
---
> 很久之前，用过一阵子Fortran（大约几周吧），然后火速转向了R语言。那时候记录了一些 ISML 函数库里面的一些函数的用法，反而基本语法倒是没怎么做笔记。好久没有更新，翻旧资料看到这个 Mrakdown 文件，于是便拿出来凑个数。内容应该是来自于 ISML 函数库，可能还有网络上搜集到的东西，已经记得不是很清楚了。
> 
> 以后可以恬不知耻地说，哥们儿也是用过 Fortran 这种上古编程语言的！

## 随机数生成(ISML)

### 一元分布随机数（离散型）

#### 均匀分布 $U(0, 1)$

```fortran
CALL RNUN (NR,R)
```

NR：生成个数
R：输出向量

#### 均匀分布 $U(a, b)$

```fortran
CALL RNUN (NR, R)
CALL SSCAL (NR, B-A, R, 1) !乘以尺度B-A
CALL SADD (NR, A, R, 1)    !加上初始的A
```

#### 二项分布 $b(n, p)$：

```fortran
CALL RNBIN (NR, N, P, IR)
```

NR：生成个数
N：二项分布参数n
P：二项分布参数p
IR：输出向量

#### 几何分布 $Ge(p)$

```fortran
CALL RNGEO (NR, P, IR)
```

NR：生成个数
P：几何分布参数p
R：输出向量

#### 超几何分布分布 $h(n, N, M)$

```fortran
CALL RNHYP (NR, n, M, N, IR)
```

NR：生成个数
n：超几何分布参数n
N：超几何分布参数N
M：超几何分布参数M
R：输出向量

N：超几何分布参数N

M：超几何分布参数M

R：输出向量

#### 泊松分布 $P(r)$

```fortran
CALL RNPOI (NR, r, IR)
```

NR：生成个数
r：泊松分布参数r
R：输出向量

r：泊松分布参数r

R：输出向量

### 一元分布随机数（连续型）

#### 标准正态分布 $N(0, 1)$

```fortran
CALL RNNOR (NR, R)
```

NR：生成个数
R：输出向量

#### 正态分布 $N(M, SD)$

```fortran
CALL RNNOR (NR, R)
CALL SSCAL (NR, SD, R, 1) !乘以标准差SD
CALL SADD (NR, M, R, 1)    !加上均值M
```

#### 卡方分布

```fortran
CALL RNCHI (NR, DF, R)
```

NR：生成个数
DF：自由度
R：输出向量

### 多元分布随机数

#### 多元正态分布

```fortran
CALL RNMVN (NR, K, RSIG, LDRSIG, R, LDR)
```

NR：生成随机向量的个数
K：随机向量的维度
RSIG：协方差矩阵的Cholesky分解中的 K ​ K 阶上三角矩阵
LDRSIG：方阵RSIG的行数(K)
R：NR ​ K 阶输出矩阵
LDR：矩阵R的行数(NR)

K：随机向量的维度

RSIG：协方差矩阵的Cholesky分解中的 K $\times$ K 阶上三角矩阵

LDRSIG：方阵RSIG的行数(K)

R：NR $\times$ K 阶输出矩阵

LDR：矩阵R的行数(NR)

**注：** 生成的随机向量均值为 $\vec{0}$，若要生成其他均值的随机向量，需要在输出矩阵的每行加上均值向量。

> Cholesky 分解是把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解。它要求矩阵的所有特征值必须大于零，故分解的下三角的对角元也是大于零的。Cholesky分解法又称平方根法，是当A为实对称正定矩阵时，LU三角分解法的变形。

```fortran
CALL CHFAC (K, COV, LDA, TOL, IRANK, RSIG, LDRSIG)
```

K：协方差矩阵的阶数
COV：协方差矩阵
LDA：矩阵COV的行数(K)
TOL：用于确定线性相关，取0.00001
IRANK：COV的秩（输出量）
RSIG：协方差矩阵的Cholesky分解中的 K ​ K 阶上三角矩阵（输出量）
LDRSIG：矩阵RSIG的行数(K)

COV：协方差矩阵

LDA：矩阵COV的行数(K)

TOL：用于确定线性相关，取0.00001

IRANK：COV的秩（输出量）

RSIG：协方差矩阵的Cholesky分解中的 K $\times$ K 阶上三角矩阵（输出量）

LDRSIG：矩阵RSIG的行数(K)

```fortran
!生成多元正态随机数
INTEGER I, IRANK, ISEED, K, LDR, LDRSIG, NR
REAL COV(2,2), R(5,2), RSIG(2,2)
EXTERNAL CHFAC, RNMVN, RNSET

NR = 5
K = 2
LDRSIG = 2
LDR = 5
COV(1,1) = 0.5
COV(1,2) = 0.375
COV(2,1) = 0.375
COV(2,2) = 0.5

! Obtain the Cholesky factorization.
CALL CHFAC (K, COV, 2, 0.00001, IRANK, RSIG, LDRSIG)

! Initialize seed of random number generator.
ISEED = 123457
CALL RNSET (ISEED)
CALL RNMVN (NR, K, RSIG, LDRSIG, R, LDR)

END
```

## 矩阵运算

### 矩阵的转置

```fortran
matrix2 = transpose(matrix1)
```

### 矩阵的乘法

```fortran
matrix = matmul(matrix_a, matrix_b)
```

### 矩阵-向量乘法

利用向量的点乘运算：`dot_product`

```fortran
do i = 1,n
    C(i) = dot_product(A(i,:),B)
end do
```

### 矩阵的逆：LINRG/DLINRG (单精度/双精度)

```fortran
CALL LINRG (N, A, LDA, AINV, LDAINV)
```

N：(输入)矩阵的阶数
A：(输入)矩阵
LDA：(输入)矩阵A的主维度，取N即可
AINV：(输出)A的逆，可以不用增加新的变量而直接赋值给A
LDAINV：AINV的主维度，取N即可

### 输出矩阵子程序

```fortran
subroutine PrintArray(array)
    real array(:,:)
    integer length,i
    length=size(array,1) 
    do i=1,length
    print *,array(i,:)
    end do
     print *,''
end subroutine
```
