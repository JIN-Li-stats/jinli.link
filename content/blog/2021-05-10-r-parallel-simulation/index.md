---
title: 在 R 中通过并行计算提高统计模拟的计算速度
author: ''
date: '2021-05-10'
slug: r-parallel-simulation
comments: true
toc: true
categories: [笔记]
tags:
  - Simulation
  - R
---
## 0. 前言

R 语言是不擅长做循环的，因此在编程时我们需要用各种办法来避免或减少循环的出现，例如使用 `apply` 函数族等等。但是在统计模拟中，一些循环甚至是大量的循环可能是必要的。一方面，我们应该尽量优化代码，例如减少类似于自加自减的运算、提前设置变量长度等，*Advanced R* $^{[2]}$ 一书的 $\S$ 5.3.1、$\S$ 24.6 等章节有细致的说明，阅读并实践此书可以写出更高效规范的 R 代码 。另一方面，使用一些更强大的工具来克服 R 的先天不足，例如本文要探讨的并行计算，以及在 R 中使用 C++、 Fortran 等编译型语言等等。

什么是并行计算？简单来说，并行计算是将多个计算指令分配给多个处理器并发执行，即将一个问题分解然后交给不同的计算机或者计算机的多个「核」同时运算。更准确、专业的描述需要更多的计算机知识，这里不深入探讨（嗐，咱也不会）。现在我们常用的计算机大都是 $n$ 核 2 $n$ 线程的，如 4 核 8 线程等。对于「核」与「线程」做个简单解释， 一个核就是 CPU 集成的一个实在的物理的运算核心，这个运算核心可以模拟当成两个核心来工作，即曰「2 线程」。R 语言默认是单核单线程计算的，故而不能发挥计算机的全部性能，我们可以通过并行计算提高 CPU 的利用率从而达到高效计算的目的。需要注意的是对于比较简单的计算，并行计算可能不但不能加快运行速度，可能还会拖慢程序，因此我建议写程序时不要从一开始就写成并行的，可以等到最后再优化运行速度。

接下来需要思考一下我们希望解决的问题。在统计模拟中，我们需要对一个过程进行几万次的重复运行，然后得到模拟的结果。而每一个过程是相对独立的，它不依赖于其他过程输出的结果。那么，我们可以把这几万次的重复运行分配给不同的运算核心（这里即以下皆指线程），将每次运行后得到的我们所关心的输出值联合起来加以处理以便分析模拟结果。这就是我们希望达到的目的。下面简单介绍利用 `foreach` 实现简单的并行计算，关于 R 语言的并行计算可以阅读《数据科学中的 R 语言》 $^{[1]}$ 一书的 $\S$ 16.4 节。

## 1. 用到的 R 包
###  parallel
parallel 包是 R 自带的用于并行的包。

### foreach
foreach 包提供了 `foreach` 循环。其基本用法为：
```r
foreach(i = 1:100, .combine = 'c') %do% {
  statement
  b
}
```
其中 `i= 1:100` 是该循环所要遍历的对象，注意这里和 `for` 循环的写法是不一样的。`statement` 是循环内的语句。 `b` 表示每次循环希望输出的运算结果， `.combine` 决定了运算结果的整合方式，默认为列表，取 `c` 时运算结果整合成一个向量。可以将计算结果赋值给一个变量以便之后的操作。以下为一个小小的例子。
```r
library(foreach)
set.seed(1)
result <- foreach(i = 1:5, .combine = 'c') %do% {
  a <- sqrt(i)
  b <- rnorm(1) + a
}
print(result)
mean(result)
```

关于 foreach 包的详细说明可见 https://CRAN.R-project.org/package=foreach 。

### doParallel

doParallel 包 提供了 foreach 循环的并行后端。下文会介绍其使用方法。

关于 doParallel 包的详细说明可见 https://CRAN.R-project.org/package=doParallel 。

**注意：** doParallel 包依赖于 iterators 包，在使用前要安装并加载 iterators 包，关于 iterators 包的详细说明可见  https://CRAN.R-project.org/package=iterators 。

## 2. 操作
第一步当然时加载 R 包了。
```r
library(parallel)
library(foreach)
library(iterators)
library(doParallel)
```

使用以下命令可以查看当前计算机有多少个核（线程）。
```r
detectCores()
```
接下来的代码建立了一个并行集群的对象并注册这个集群对象使之为 `foreach` 所用。听起来有点不像人话，我的理解就是取了计算机的若干个核心来做并行计算，我们可以把任务分配给注册的集群中的核心。
```r
cl <- makeCluster(8)
registerDoParallel(cl)

```
注意 `makeCluster` 中的数字不可以超过 `detectCores` 的返回值。

接下来我们把 `foreach` 循环「改造」成并行的版本。非常简单，只需要把 `%do%` 改成 `%dopar%` 即可：
```r
result <- foreach(i = 1:100, .combine = 'c', .packages = c("P1", "P1") ,.export = c("fun1", "fun2")) %dopar% {
  statement
  b
}
```
其他均不变。注意到这里增加了两个参数，`.packages` 和 `.export` 分别表明了循环中需要用到的 R 包 和自定义函数，这在非并行时是不需要的。如此便可以将这许多个任务分配给多个计算机核心同时进行计算，我们可以将结果赋值给一个变量如 `result` 以便做其他的处理。

最后记得加一句 `stopCluster(cl)` 以停止集群，好借好还，再借不难。

## 3. 例子
如下一个例子来自 [3]， 是一个控制图的模拟计算。并行计算一定是用于解决一个相对复杂的统计模拟，简单的例子似乎没有什么意义。而每个人有不同的专业方向，我实在想不到一个好的且通用的例子，因而只放了一个我所熟悉的。

```r
rm(list = ls())
library(parallel)
library(foreach)
library(iterators)
library(doParallel)

# Random number -----------------------------------------------------------
rgbe <- function(theta1,theta2,delta){
  unif  <-  runif(1,0,1)
  psi <- rbinom(1,1,delta)
  e <- rexp(1,1) + psi*rexp(1,1)
  x1 <- theta1*(unif^delta)*e
  x2 <- theta2*((1-unif)^delta)*e
  return(c(x1,x2))
}

# Function for ARL Computation  -------------------------------------------
arl <- function(tau, controlLimit){
  loop1 <- 10000
  loop2 <- 5000
  k <- 0.05
  theta1 <- 1
  theta2 <- 1
  delta <- 0.5
  mu0 <- c(theta1, theta2)
  rho <- 2*(gamma(delta+1)^2)/gamma(2*delta+1) - 1
  Sig <- matrix(data = c(theta1^2, rho*theta1*theta2,
                         rho*theta1*theta2, theta2^2), nrow=2, ncol=2, byrow=TRUE)
  inverse.Sig <- solve(Sig)
  # detectCores()
  cl <- makeCluster(8)
  registerDoParallel(cl)
  rls <- foreach(i = 1:loop1, .combine = 'c', .export = "rgbe") %dopar% {
    st <- c(0,0)
    for (j in 1:loop2) {
      xt <- rgbe(theta1,theta2,delta)*tau
      ct <- ((st + xt - mu0)%*%inverse.Sig%*%(st +xt -mu0))^(0.5)
      if(ct > k){
        st <- (st + xt - mu0)*rep((1 - k/ct),2)
      }else{
        st <- c(0,0)
      }
      qt <- (st%*%inverse.Sig%*%st)^(0.5)
     if(qt > controlLimit | j == loop2){
       rl <- j
       break
     }
    }
    rl
  }
  stopCluster(cl)
  arl <- mean(rls)
  sdrl <- sd(rls)
  return(c(arl,sdrl))
}
```


## 4. 注意事项
没什么要注意的，bug 从来都是在我不想让它出现的时候出现。

## 参考文献

1. 李舰, 肖凯. 数据科学中的 R 语言 [M]. 西安交通大学出版社, 2015.
2. Wickham, Hadley. Advanced R [M]. CRC Press, 2019. ([online version](https://adv-r.hadley.nz/))
3. Xie F P ,  Sun J S ,  Castagliola P , et al. A multivariate CUSUM control chart for monitoring Gumbel's bivariate exponential data [J]. Quality and Reliability Engineering International.