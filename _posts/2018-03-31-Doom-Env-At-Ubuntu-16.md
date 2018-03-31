---
layout: post
title: VizDoom在Ubuntu 16中的相关配置
date: 2018-03-31 14:00:00
categories: blog
tags: [VizDoom,Reinforcement Learning,Ubuntu,Gym]
description: 文章金句。
---


# 
# 安装流程
- [官方安装链接](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md)
# 简要过程
1. 下载[Anacanda](https://www.anaconda.com/download/)，安装Python 3 

   ```
   $./Anaconda3-5.1.0-Linux-x86_64.sh
   ```

2. 安装基本库
   ```
   sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev unzip
   sudo apt-get install libboost-all-dev

   ```
3. 安装VizDoom

   使用 
      ``` 
      pip install vizdoom
      ``` 
   出现各种错误，编译源码来完成完成
   此步使用如下进行安装
      ```
      $./cmake_all.sh
      $make
      > To manually install Python package copy vizdoom_root_dir/bin/pythonX.X/pip_package contents to python_root_dir/lib/pythonX.X/site-packages/site-packages/vizdoom.
      $cp -r /home/xxx/Documents/SourceFiles/ViZDoom/bin/python3.6/pip_package/* ./vizdoom/
      $cp ./vizdoom pythonPath/site-packages/site-packages/vizdoom
      ```
4. 安装Doom-py库

   使用下面命令即可，不需要编译源码【本来打算的:(】
      ```
      pip install doom-py
      ```
5. 安装gym、gym-pull[可选]

   使用gym-pull 注意，导入出现下面的错误:
      ```
      /anaconda3/bin/python "/Users/001GymUsage.py" 
       Traceback (most recent call last):
       File "/Users/001GymUsage.py", line 4, in <module>
       import gym_pull
       File "/anaconda3/lib/python3.6/site-packages/gym_pull/__init__.py", line 6, in <module>
       from gym.configuration import logger_setup, undo_logger_setup
       ModuleNotFoundError: No module named 'gym.configuration'
      ```
    进行过相关搜索，未找到解决方法，可能的缘由：新版本的gym不支持直接配置类的操作了
    
# 安装出现的问题
- 修改源码中camke_all.sh 的内容(通过log发现有些库未指定，如果是Python3的话，DBUILD_PYTHON3=ON)
   ```
   #cmake -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYTHON3=ON -DBUILD_JAVA=OFF
   cmake -DCMAKE_BUILD_TYPE=Release \
   -DBUILD_PYTHON3=ON \
   -DPYTHON_EXECUTABLE=/home/xxx/anaconda3/bin/python \
   -DPYTHON_INCLUDE_DIR=/home/xxx/anaconda3/include/python3.6m \
   -DNUMPY_ROOT_DIR=/home/xxx/anaconda3/lib/python3.6/site-packages/numpy\
   -DNUMPY_LIBRARIES=/home/xxx/anaconda3/lib/python3.6/site-packages/numpy/lib\
   -DNUMPY_INCLUDES=/home/xxx/anaconda3/lib/python3.6/site-packages/numpy/core/include
   ```
- 出现以下类似的错误（Anaconda路径下的libharfbuzz未定义）[网络出现的类似问题](https://github.com/jaagr/polybar/issues/310),使用[harfbuzz库](https://harfbuzz.github.io/install-harfbuzz.html#download)解决此问题 **缘由貌似是Anaconda中的库太老，==通过更新解决==**
   ```
   libharfbuzz: Undefined reference to 'FT_Get_Var_Blend_Coordinates'
   ``` 

   ```
   # 解决方法：
   # 1 Download the source
   # 2 配置
   $./configure --prefix=/usr --with-gobject  --enable-introspection
   # 3 安装
   $ make
   $ sudo make install
   # 4 拷贝到Anaconda的目录中
   $ cp /usr/lib/libharfbuzz.so.0.10706.0 /home/hatim/anaconda3/lib/
   # 指向新的库
   $ ln -f /home/xxx/anaconda3/lib/libharfbuzz.so.0.10706.0 /home/xxx/anaconda3/lib/libharfbuzz.so.0 
   ```

