---
title: Julia 深度学习（二）
author: ''
date: '2022-11-30'
slug: julia-dl-2
comments: true
toc: true
categories:
  - 笔记
tags:
  - Julia
  - Deep learning
---
> 本来想起名叫「Julia 深度学习（一.X）」，因为这篇博客内容太少了，但是「（一.X）」看起来实在是有点难看，就叫「（二）」吧，下次再聊聊隐藏层的一些基本情况。

我们来写一个自定义函数，比如就叫做 `my_nn`，输入是一些超参数， 输出是一个神经网络。
首先的首先，加载一下 Flux。
```julia
using Flux
```

## 1. 没有标题（或者叫准备环节吧）
首先需要解决的一个问题是（也是本文的关键），怎样建立一个由输入参数决定的 $n$ 个相同的隐藏层汇集到 `Chain` 里呢？注意这里的 $n$ 是由函数的输入得到的！请参考  [这里](https://discourse.julialang.org/t/function-for-creating-a-neural-network-with-n-hidden-layers-in-flux/75589) 。这里我们总最简单的开始，把 3 个相同的隐藏层叠起来：

```julia
n = 3
layers = [Dense(3 => 3,relu) for _ in 1:3]
## 3-element Vector{Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}}:
##  Dense(3 => 3, relu)  # 12 parameters
##  Dense(3 => 3, relu)  # 12 parameters
##  Dense(3 => 3, relu)  # 12 parameters

model = Chain(layers...)
## Chain(
##   Dense(3 => 3, relu),                  # 12 parameters
##   Dense(3 => 3, relu),                  # 12 parameters
##   Dense(3 => 3, relu),                  # 12 parameters
## )                   # Total: 6 arrays, 36 parameters, 528 bytes.
```

这里的一个关键点在于 `layers...` 这个地方的`...`，[上一篇](https://jinli.link/blog/julia-dl-1/)博客里提到了这个 "splat" 符号，可以去看一下。如果不加上这个小玩意儿，就会出现大问题：
```julia
model_error = Chain(layers)
## Chain([
##   Dense(3 => 3, relu),                  # 12 parameters
##   Dense(3 => 3, relu),                  # 12 parameters
##   Dense(3 => 3, relu),                  # 12 parameters
])                  # Total: 6 arrays, 36 parameters, 528 bytes.
```
对比一下，是不是就能看出来这个 `model_error` 不大行。

## 2. 写个小函数
```julia
using Flux

function my_sim_nn(n_input, n_out, n_inter_layers, n_nodes, act_fun)
    # n_inter_layers 是除了第一层和最后一层之外的层数
    
    first_layer = Dense(n_input => n_nodes, act_fun)
    last_layer = Dense(n_nodes => n_out, identity)

    intermediate_layers = [Dense(n_nodes => n_nodes, act_fun) for _ in 1:n_inter_layers]

    model = Chain(
    first_layer,
    intermediate_layers...,
    last_layer
    )
    return model
end
## my_sim_nn (generic function with 1 method)
```

然后测试一下，直接看例子：
```julia
nn1 = my_sim_nn(4, 2, 1, 4, NNlib.selu)
## Chain(
##   Dense(4 => 4, selu),                  # 20 parameters
##   Dense(4 => 4, selu),                  # 20 parameters
##   Dense(4 => 2),                        # 10 parameters
## )                   # Total: 6 arrays, 50 parameters, 584 bytes.
```
## 3. 写个复杂点的函数（加上丢弃层）
再次首先，丢弃（dropout）层的语法是 `Dropout(p)`，其中 `p` 是丢弃率，会把前一层的节点以概率  `p` 扔掉（应该是这个意思吧）。关键问题在于怎样把普通的隐藏层和 `Dropout(p)` 一个接着一个的排起来。方法应该有很多，我是这么写的：

```julia
function my_nn(n_input, n_out, n_inter_layers, n_nodes, act_fun, drop_rate)
    
    first_layer = Dense(n_input => n_nodes, act_fun)
    last_layer = Dense(n_nodes => n_out, identity)

    hide_layer = [Dense(n_nodes => n_nodes, act_fun) for _ in 1:n_inter_layers]
    dropout_layer = [Dropout(drop_rate) for _ in 1:n_inter_layers]

    mid_layers = hcat(hide_layer, dropout_layer)

    intermediate_layers = mid_layers[1, :]
    if n_inter_layers > 1
        for i in 2 : n_inter_layers
            intermediate_layers = vcat(intermediate_layers, mid_layers[i, :])
        end
    end
        
    model = Chain(
    first_layer, Dropout(drop_rate),
    intermediate_layers...,
    last_layer
    )
    return model
end
## my_nn (generic function with 1 method)
```
通常情况下，我的方法总是很笨，总是要方便我蠢蠢的脑袋瓜理解，并且一不小心就走弯路！还是测试一下吧：

```julia
nn2 = my_nn(4, 2, 2, 4, NNlib.relu, 0.1)
## Chain(
##   Dense(4 => 4, relu),                  # 20 parameters
##   Dropout(0.1),
##   Dense(4 => 4, relu),                  # 20 parameters
##   Dropout(0.1),
##   Dense(4 => 4, relu),                  # 20 parameters
##   Dropout(0.1),
##   Dense(4 => 2),                        # 10 parameters
## )
```
## 4. 总结
不想写了！

