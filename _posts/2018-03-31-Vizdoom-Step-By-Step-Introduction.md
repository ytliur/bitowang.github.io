---
layout: post
title: VizDoom入门介绍
date: 2018-03-31 16:00:00
categories: blog
tags: [VizDoom, Reinforcement Learning]
description: 文章金句。
---


# 实例及介绍
## 配置

    ```
    from vizdoom import DoomGame
    from vizdoom import Button
    from vizdoom import GameVariable
    from vizdoom import ScreenFormat
    from vizdoom import ScreenResolution
    ```
- DoomGame - 代表游戏，是游戏引擎和代理的引擎
- Button - 可以由代理按下的代表动作的枚举，如攻击等
- GameVariable - 辅助状态信息（健康、弹药）的枚举
- ScreenFormat and ScreenResolution - 屏幕分辨率和格式的枚举


    ```
    game = DoomGame() # 创建DoomGame对象
    #配置外部所需要的文件
    game.set_doom_scenario_path("../../scenarios/basic.wad")
    game.set_doom_map("map01")
    ```
- Doom scenario - *.wad文件定义如何玩和看起来像，简单例程可访问[此链接](https://github.com/Marqt/ViZDoom/tree/master/scenarios)；可以由[DoomBuilder](http://www.doombuilder.com/index.php?p=downloads)或[Slade](http://slade.mancubus.net/)自定义
- Doom map - 单个wad文件（场景）可能包含多个可供选择的地图

    ```
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    ```
- 可以设置渲染模式，包括屏幕分辨率、屏幕缓冲区格式及是否渲染特定的视觉元素，例如十字准线或HUD等

    ```
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)
    ```
- 确定agent的那些操作
- 注：如果跳过这步，没有可用的操作，所有的将徒劳

    ```
    game.add_available_game_variable(GameVariable.AMMO2)
    ```
- 确定哪些游戏变量（健康，弹药，武器的可用性等）将包含在每次获得的状态中
- 任何游戏变量都可以在游戏过程中随时获得，但让它们处于状态可能会更方便。在这里，我们只包括AMMO2，这是手枪弹药

    ```
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    # 改为Fasle，可通过Terminal进行运行，否则不可
    game.set_window_visible(True) 
    ```
- 定义一些其他配置，如窗口的可见性，情节时间（在tics /frame中）或开始时间（初始时间由环境忽略，但在内部，引擎仍运行它们）

    ```
    game.set_living_reward(-1)
    ```
- 无论发生什么代理移动将获得-1的奖励

    ```
    game.init()
    ```
- 初始化，Doom窗口出现

## 游戏运行
- 单个Doom游戏叫episode(情节)，
- 情节是独立的，并在玩家的死亡，超时或某些自定义条件满足时结束，由scenario定义
- 在下面的例子中，情节在300tics或当怪兽被杀掉时结束
- 奖励说明：100杀死怪兽；-6射击并未射中；-1其他

    ```
    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()

        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()

            # Which consists of:
            n           = state.number
            vars        = state.game_variables
            screen_buf  = state.screen_buffer
            depth_buf   = state.depth_buffer
            labels_buf  = state.labels_buffer
            automap_buf = state.automap_buffer
            labels      = state.labels

            # Makes a random action and get remember reward.
            r = game.make_action(choice(actions))

            # Makes a "prolonged" action and skip frames:
            # skiprate = 4
            # r = game.make_action(choice(actions), skiprate)

            # The same could be achieved with:
            # game.set_action(choice(actions))
            # game.advance_action(skiprate)
            # r = game.get_last_reward()

            # Prints state's game variables and reward.
            print("State #" + str(n))
            print("Game variables:", vars)
            print("Reward:", r)
            print("=====================")

            if sleep_time > 0:
                sleep(sleep_time)

        # Check how the episode went.
        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")
    ```
- 在Doom的世界里，包括获取状态（get_state）、做出动作（make_action）和获得奖励（）。
- get_state 返回一个 GameState 对象
- make_action 将一个操作（Action）作为输入并返回一个奖励，该操作（Action）应该是一个长度等于配置中指定的可用按钮数量的整数列表，每个列表位置映射到制定的Button：[0,0,1] 意味 "not left, not right, attack!"，如果太短，0 将在遗失的位置进行填充！


    ```
    game.close()
    ```
- 终止游戏

# 配置文件
不用在代码配置实验，可以从配置文件加载它。每个文件都按顺序读取，因此具有相同(key)密钥的多个条目将相互覆盖
## 格式
配置文件的入口包括一个键值对（分隔符为=），遵循下面规则:

- 每行一个条目（列表参数除外）
- 不区分大小写
- 以＃•开头的行将被忽略
- 键中的下划线_ 被忽略（episode_timeout与episodetimeout）相同，
- 字符串值不应该被撇号或引号包围

违反任何这些规则将导致只忽略错误的行

### 列表参数【list parameters】
- available_buttons and available_game_variables 特殊参数，它们使用多个值，而不是单个值，它们期望由空格分隔并包围在大括号内的值列表 ("{" and "}")
- 只要所有值之间用空格分隔，该列表就可以随意扩展到所有行
- KEY = { VALUES } 将清空先前定义此key的值
- KEY += { VALUES } 可添加

    ```
    #doom_game_path = ../../scenarios/doom2.wad
    doom_scenario_path = ../../scenarios/basic.wad
    doom_map = map01

    # Rewards
    living_reward = -1

    # Rendering options
    screen_resolution = RES_320X240
    screen_format = CRCGCB
    render_hud = True
    render_crosshair = false
    render_weapon = true
    render_decals = false
    render_particles = false
    window_visible = true

    # make episodes start after 20 tics (after unholstering the gun)
    episode_start_time = 14

    # make episodes finish after 300 actions (tics)
    episode_timeout = 300

    # Available buttons 
    available_buttons = 
        { 
            MOVE_LEFT 
            MOVE_RIGHT 
        }
    #    
    available_buttons += { ATTACK }

    # Game variables that will be in the state
    available_game_variables = { AMMO2}

    mode = PLAYER
    doom_skill = 5

    #auto_new_episode = false
    #new_episode_on_timeout = false
    #new_episode_on_player_death = false
    #new_episode_on_map_end = false
    ```

## 执行操作（perfom action）
- 通常使用make_action方法，该方法获取按钮状态列表并返回奖励，可指定第二个参数:tic,将会告诉环境在tic个frame执行同样的操作，DOOM称之为“frame skipping”，可以提升游戏的性能（无需渲染）
### advance_action and set_action
- make_action方法不会让你干涉在跳过的帧期间发生的任何事情
- 为了实现更多的多功能性，您可以使用更精细的控制
    ```
    ...
    game.set_action(my_action)
    tics = 5
    update_state = True # determines whether the state and reward will be updated
    render_only = True  # if update_state==False, it determines whether a new frame (image only) will be rendered (can be retrieved using get_game_screen())

    # action lasts 5 tics
    game.advance_action(tics) 

    # doesn't update the state but renders the screen buffer
    game.advance_action(1, not update_state, render_only)

    # skips one frame and updates the state
    game.advance_action(2, update_state, render_only)

    .
    ```

## 模块
四种操作模式

### PLAYER（游戏）
- 让agent感知状态并做选择
- 完全同步的，游戏引擎等待make_action or advance_action
### SPECTATOR（旁观者）
- 主要用于学徒学习,允许您（人类）在代理读取游戏状态和您的操作时使用键盘和鼠标来玩游戏（get_last_action）
- 同步的，因此引擎的处理将等待代理人授予他权限继续（advance_action）
- 运行35fps，因此代理方的快速计算不会导致整体加速，另一方面未能匹配〜35fps的速度将导致游戏速度缓慢或/和波涛汹涌
    ```
    game = DoomGame()

    #CONFIGURATION
    ...

    game.set_mode(Mode.SPECTATOR)
    game.init()

    episodes = 10
    for i in range(episodes):
        print("Episode #" +str(i+1))

        game.new_episode()
        while not game.is_episode_finished():
            s = game.get_state()
            game.advance_action()
            a = game.get_last_action()
            r = game.get_last_reward()
            ...
     ...    
     ```
### 异步Player & SPECTATOR

## 自定义场景
> To create a custom scenario (iwad file), you need to use a dedicated editor. Doom Builder and Slade are the software tools we recommend for this task.

> Scenarios (iwad files) contain maps and ACS scripts. For starters, it is a good idea to analyze the sample scenarios, which come with ViZDoom (remember that these are binary files).

需创建场景文件，需包含map和脚本
### 注意
> ACS and software for creating wads is quite simple and relatively user friendly but sometimes they act unexpectedly without notifying you so here are some thoughts that can potentailly help you and save hours of wondering:

> 1.0 and 1 is not the same, the first one is the fixed point number stored in int and the second one is an ordinary int. Watch out what is expected by the functions you use cause using the wrong format can result in rubbish.

> Use UDMF format for maps and ZDBPS which is node (whatever that is) builder for Zdoom.

### 奖励(reward)
- 为了使用奖励机制，您需要使用全局变量0
    ```
    global int 0:reward;
    ...
    script 1(void)
    {
        ...
        reward += 100.0;
    }
    ...
    ```
### 用户变量(User Variables)
- GameVariable 表示非视觉游戏数据，例如弹药，健康点或武器准备状态，可以存在于状态中或可随时提取。
- 可以访问对应于ACS脚本的全局变量1-32的用户变量（USER1，USER2 ... USER32)
    ```
    global int 0:reward;
    global int 1:shaping_reward;
    global int 2:some_int_value;
    ...
    script 1(void)
    {
        ...
        reward += 100.0;
        ...
        shaping_reward += 10.0;
        ...
        some_int_value += 1;
    }
    ...
    ```
> By default, the USER variables are treated as ordinary integers, so using fixed point numbers inside the script will result in a rubbish output. However, you can turn the rubbish into meaningful data using doom_fixed_to_double function.
- 使用 doom_fixed_to_double 将垃圾转化为有用的数据
    ```
    ...
    rubbish = game.get_game_variable(GameVariable.USER1)
    legitimate_integer = game.get_game_variable(GameVariable.USER2)
    meaningful_data_as_double = doom_fixed_to_double(rubbish)
    ...
    ```
### 可用按钮(Available Buttons)
> ViZDoom uses "buttons" as constituents of actions like ATTACK of MOVE_LEFT. Most of buttons support only binary values and they act like keyboard keys. More specifically they use boolean values or integers (which are intepreted as bools: 0 is False and everything else is True).
- 可使用布尔
> There are, however, 5 special buttons which accept negative and positive values. Names of these buttons end with "DELTA" and they emulate a mouse device. Value range of delta buttons can be limited by set_button_max_value() method
- 后缀为delta的仿真鼠标设备，可由set_button_max_value进行范围限制

    ```
    LOOK_UP_DOWN_DELTA
    TURN_LEFT_RIGHT_DELTA 
    MOVE_FORWARD_BACKWARD_DELTA
    MOVE_LEFT_RIGHT_DELTA
    MOVE_UP_DOWN_DELTA
    ```
### 自定义键绑定(Custom Key Bindings)
- 当你的程序调用init（）方法时，vizdoom.ini将在程序目录中创建,它将包含默认的ZDoom引擎设置
- 找到 [Doom.Bindings] 并根据需要更改它们
> Snippet showing how to set WSAD keys as moving buttons and enable mouse
    ```
    [Doom.Bindings]
    ...
    w=+forward
    a=+moveleft
    d=+moveright
    s=+back
    freelook=true
    ...
    ```





## Q-Learning with Tensorflow
### Q-Learning
使用Q-Learning进行学习
- 论文 [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236.pdf)
- 视频 [Lecture 10: Reinforcement Learning](https://www.youtube.com/watch?v=w33Lplx49_A)

### 函数
- preprocess - 降采样并将像素从字节转化为[0,1]的32byte浮点数
- ReplayMemory - 存储代理所经历的存储转换的类 每次转换包括状态前和转换后的两张图片、执行操作的索引、奖励和一个代表第二章台是否结束的布尔值
> class for storing transitions (the most recent ones) experienced by the agent. Each transition consists of 2 images (states before and after the transition), index of the performed action, reward and a boolean value saying whether the second state was terminal or not.
- create_network - 函数，创建网络。返回网络对象，函数可使用其进行学习和输出Q值
> function that creates the network in Theano. It returns the network object and functions using it to learn and output Q-values.
- perform_learning_step - 首先根据epsilon-greedy策略执行操作并将转换存储在重播内存中。Epsilon-greedy策略以概率epsilon选择随机动作且最好的行动（至少根据目前的估计）以概率1-epsilon。每个步骤结束时使用从网络上随机采样（从内存）小批量并运行rmsprop的单步骤
> function that first performs an action according to epsilon-greedy policy and stores the transition in the replay memory. Epsilon-greedy policy, with probability epsilon chooses a random action and the best action (at least according to current estimates) with probability 1-epsilon. Each step ends with using randomly sampled (from the memory) mini-batch and running single step of rmsprop on the network.
