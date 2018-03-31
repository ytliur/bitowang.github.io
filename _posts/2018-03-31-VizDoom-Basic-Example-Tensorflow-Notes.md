---
layout: post
title: VizDoom Tensorflow实例代码解析
date: 2018-03-31 15:00:00
categories: blog
tags: [Tensorflow,VizDoom]
description: 文章金句。
---


# 概述
1. 程序从初始化开始并设置一些参数,然后生成所有可能的动作（3个按钮，2 ^ 3动作，以便网络有8个输出），创建网络和重播内存
2. 初始化后，开始训练。它涉及多个训练时期。每个时期都包含许多学习步骤（perform_learning_step）。在整个学习过程中，ε是线性减少的，这意味着在早期，代理将完成随机行为（探索），并开始在事后合理地开展工作。每个训练时期以多个测试集结束，在此期间epsilon为0，因此不允许进行探索。代理的平均，最大和最小结果被打印，网络的权重被保存
3. 训练后，DoomGame被重新初始化为不同的模式，并显示游戏窗口，以便您可以观察代理人执行其魔法

# 主函数 main

    ```
    if __name__ == '__main__':
        # Create Doom instance # 创建Doom实例
        game = initialize_vizdoom(config_file_path)

        # Action = which buttons are pressed
        # 获取动作的个数 此处为3
        n = game.get_available_buttons_size()
        # 以 0 和 1 组合长度为3的list列表的list
        # actions = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        actions = [list(a) for a in it.product([0, 1], repeat=n)]

        # Create replay memory which will store the transitions
        # 创建存储转化的重播内存
        memory = ReplayMemory(capacity=replay_memory_size)
        # 开启会话
        session = tf.Session()
        # 得到网络结构
        learn, get_q_values, get_best_action = create_network(session, len(actions))
        # 训练保存句柄
        saver = tf.train.Saver()
        # 是否加载模型
        if load_model:
            # 加载模型
            print("Loading model from: ", model_savefile)
            saver.restore(session, model_savefile)
        else:
            # 初始化
            init = tf.global_variables_initializer()
            session.run(init)
        print("Starting the training!")

        time_start = time()
        if not skip_learning: # 判断是否跳过学习，此处为否
            for epoch in range(epochs):
                print("\nEpoch %d\n-------" % (epoch + 1))
                train_episodes_finished = 0
                train_scores = []

                print("Training...")
                game.new_episode()
                # trange tqdm的模块 进度条
                for learning_step in trange(learning_steps_per_epoch, leave=False):
                    perform_learning_step(epoch)
                    if game.is_episode_finished():
                        score = game.get_total_reward()
                        train_scores.append(score)
                        game.new_episode()
                        train_episodes_finished += 1

                print("%d training episodes played." % train_episodes_finished)

                train_scores = np.array(train_scores)

                print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                      "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

                print("\nTesting...")
                test_episode = []
                test_scores = []
                for test_episode in trange(test_episodes_per_epoch, leave=False):
                    game.new_episode()
                    while not game.is_episode_finished():
                        state = preprocess(game.get_state().screen_buffer)
                        best_action_index = get_best_action(state)

                        game.make_action(actions[best_action_index], frame_repeat)
                    r = game.get_total_reward()
                    test_scores.append(r)

                test_scores = np.array(test_scores)
                print("Results: mean: %.1f±%.1f," % (
                    test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                      "max: %.1f" % test_scores.max())

                print("Saving the network weigths to:", model_savefile)
                saver.save(session, model_savefile)

                print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

        game.close()
        print("======================================")
        print("Training finished. It's time to watch!")

        # Reinitialize the game with window visible
        # 将游戏窗口显示为可见
        game.set_window_visible(True)
        # 设置模式为异步play模式
        game.set_mode(Mode.ASYNC_PLAYER)
        # 游戏初始化
        game.init()

        for _ in range(episodes_to_watch):
            game.new_episode()
            while not game.is_episode_finished():
                # 从屏幕换从中得到状态
                state = preprocess(game.get_state().screen_buffer)
                # 获得状态的最佳操作
                best_action_index = get_best_action(state)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                game.set_action(actions[best_action_index])
                for _ in range(frame_repeat):
                    game.advance_action()

            # Sleep between episodes
            sleep(1.0)
            score = game.get_total_reward()
            print("Total score: ", score)

    ```

# 导包及相关函数

- 包
    ```
    from __future__ import division
    from __future__ import print_function
    from vizdoom import *
    import itertools as it
    from random import sample, randint, random
    from time import time, sleep
    import numpy as np
    import skimage.color, skimage.transform
    import tensorflow as tf
    from tqdm import trange
    ```

- 模型参数

    ```
    # Q-learning settings 
    learning_rate = 0.00025
    # learning_rate = 0.0001
    discount_factor = 0.99
    epochs = 20
    learning_steps_per_epoch = 2000
    replay_memory_size = 10000

    # NN learning settings
    batch_size = 64

    # Training regime
    test_episodes_per_epoch = 100

    # Other parameters
    frame_repeat = 12
    resolution = (30, 45) # 分辨率
    episodes_to_watch = 10

    # 模型保存参数
    model_savefile = "/tmp/model.ckpt"
    save_model = True # 是否保存模型
    load_model = False # 是否加载模型
    skip_learning = False # 是否跳过训练过程
    # Configuration file path
    config_file_path = "../../scenarios/simpler_basic.cfg"

    ```

- 预处理 转化及降采样输入图像

    ``` 
    # Converts and down-samples the input image
    def preprocess(img):
        img = skimage.transform.resize(img, resolution)
        img = img.astype(np.float32)
        return img
    ```

- ReplayMemory - 存储代理所经历的存储转换的类 每次转换包括状态前和转换后的两张图片、执行操作的索引、奖励和一个代表第二章台是否结束的布尔值

    ```
    class ReplayMemory:
        def __init__(self, capacity):
            channels = 1
            state_shape = (capacity, resolution[0], resolution[1], channels)
            self.s1 = np.zeros(state_shape, dtype=np.float32)
            self.s2 = np.zeros(state_shape, dtype=np.float32)
            self.a = np.zeros(capacity, dtype=np.int32)
            self.r = np.zeros(capacity, dtype=np.float32)
            self.isterminal = np.zeros(capacity, dtype=np.float32)

            self.capacity = capacity
            self.size = 0
            self.pos = 0

        def add_transition(self, s1, action, s2, isterminal, reward):
            self.s1[self.pos, :, :, 0] = s1
            self.a[self.pos] = action
            if not isterminal:
                self.s2[self.pos, :, :, 0] = s2
            self.isterminal[self.pos] = isterminal
            self.r[self.pos] = reward

            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        def get_sample(self, sample_size):
            i = sample(range(0, self.size), sample_size)
            return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]
    ```

- initialize_vizdoom - 初始化VizDoom


    ```
    # Creates and initializes ViZDoom environment.
    def initialize_vizdoom(config_file_path):
        print("Initializing doom...")
        game = DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(False)
        game.set_mode(Mode.PLAYER)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.init()
        print("Doom initialized.")
        return game
    ```

# 网络

- 网络将状态作为输入并为每个可用操作返回一个Q值
- 有一个尺寸为30x45和8个动作的单个灰度图像（3个按钮提供了8个组合的开启和关闭），我们到达1350个输入和8个输出
> A different architecture could be proposed that consists of 1350 + 3 (one additional for each button) inputs and 1 output. This representation seems conceptually better but requires multiple forward passes to compute all Q-values for a single state so it's more computationally demanding.

## 创建网络

- create_network - 函数，创建网络。返回网络对象，函数可使用其进行学习和输出Q值

    ```
    def create_network(session, available_actions_count):
        # Create the input variables
        s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
        best_a = tf.argmax(q, 1)

        loss = tf.losses.mean_squared_error(q, target_q_)

        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)
        # 以学习内容作为输入并返回平均损失，同时执行反向传播
        def function_learn(s1, target_q):
            feed_dict = {s1_: s1, target_q_: target_q}
            l, _ = session.run([loss, train_step], feed_dict=feed_dict)
            return l
        # 无需任何学习即可获取状态并返回q值
        def function_get_q_values(state):
            return session.run(q, feed_dict={s1_: state})
        # 采取单一状态并返回最佳动作的索引。这不是必要的，但很方便
        def function_get_best_action(state):
            return session.run(best_a, feed_dict={s1_: state})

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]

        return function_learn, function_get_q_values, function_simple_get_best_action
    ```

- learn from memory

    ```
    def learn_from_memory():
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if memory.size > batch_size:
            s1, a, s2, isterminal, r = memory.get_sample(batch_size)

            q2 = np.max(get_q_values(s2), axis=1)
            target_q = get_q_values(s1)
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
            learn(s1, target_q)
    ```

- perform_learning_step - 首先根据epsilon-greedy策略执行操作并将转换存储在重播内存中。Epsilon-greedy策略以概率epsilon选择随机动作且最好的行动（至少根据目前的估计）以概率1-epsilon。每个步骤结束时使用从网络上随机采样（从内存）小批量并运行rmsprop的单步骤

    ```
    def perform_learning_step(epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(epoch):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * epochs  # 10% of learning time
            eps_decay_epochs = 0.6 * epochs  # 60% of learning time

            if epoch < const_eps_epochs:
                return start_eps
            elif epoch < eps_decay_epochs:
                # Linear decay
                return start_eps - (epoch - const_eps_epochs) / \
                                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps

        s1 = preprocess(game.get_state().screen_buffer)

        # With probability eps make a random action.
        eps = exploration_rate(epoch)
        if random() <= eps:
            a = randint(0, len(actions) - 1)
        else:
            # Choose the best action according to the network.
            a = get_best_action(s1)
        reward = game.make_action(actions[a], frame_repeat)

        isterminal = game.is_episode_finished()
        s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        memory.add_transition(s1, a, s2, isterminal, reward)

        learn_from_memory()

    ```
