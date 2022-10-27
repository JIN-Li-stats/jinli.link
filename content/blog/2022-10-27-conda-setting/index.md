---
title: Conda 常用命令及设置
author: ''
date: '2022-10-27'
slug: conda-setting
comments: true
categories:
  - 笔记
tags:
  - python
  - conda
---
> 好久没有更新内容了。距离上一次更新已经快1年了，这么一算竟还颇有感慨。

本文是我在使用 Miniconda 时用到的一些命令以及设置，因为本人记忆力实在是不大行，故整理下来方便下次使用时复制。内容当然还是比较少的，而且。我会在以后的学习中继续整理、更新！

## 基础设置
- 查看 conda 信息
  
  ```
  conda info
  ```
- 查看有哪些虚拟环境的信息
  ```
  conda info -e 或 conda env list
  ```
- 清理缓存
  ```
  conda clean -all
  ```
### `.condarc`中的设置
- `.condarc`文件的位置：`conda info` 中 `user config file` 所指示的位置。
- 更改虚拟环境位置：在`.condarc`文件中添加
  ```
  envs_dirs:
    - C:\Miniconda3\envs
  ```
- 切换下载源：在`.condarc`文件中添加（或修改）
  ```
  channels:
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    - defaults
  show_channel_urls: true
  ```
## 虚拟环境
- 创建虚拟环境
  ```
  conda create -n 环境名称 python=3.8
  ```
- 激活虚拟环境
  ```
  conda activate 环境名称
  ```
- 退出虚拟环境（回到 base 环境）
  ```
  conda deactivate 环境名称
  ```
- 删除虚拟环境
  ```
  conda remove -n 环境名称 --all
  ```
- 复制一个环境，如从 py310 复制一个名为 clone_py310 的环境
  ```
  conda create -n clone_py310 --clone py310
  ```
## 包管理
- 查看所有安装的包
  ```
  conda list
  ```
- 安装包
  ```
  conda install 包名(=版本号)
  ```
- 删除包
  ```
  conda uninstall 包名(=版本号)
  ```

## 其他
### 在 jupyter 中管理环境

1. 在 base 环境和该环境下都安装 `ipykernel` package：
  ```
  conda install ipykernel
  ```
2. 将环境写入 jupyter 的 kernel 中：
  ```
  python -m ipykernel install --user --name 环境名称 --display-name "显示的名称"
  ```

3. 删除kernel环境（不要进入此环境）：
  ```
  jupyter kernelspec remove 环境名称
  ```