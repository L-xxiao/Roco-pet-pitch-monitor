# Roco-pet-pitch-monitor
🎵 洛克王国精灵叫声音调监测器
Windows 平台音调对比工具，可置顶半透明显示在游戏界面上，录制并对比多段音频的音调差异。

功能
A/B/C 三段录制：分别录制三段音频，自动裁剪首尾静音，检测综合音调、最低音调、最高音调
音调对比：自动计算各段之间的音分差（cents），显示高低关系
波形显示：三段音频波形统一时间比例、有效区域居中对齐
导出素材：将裁剪后的有效音频导出为 16bit PCM WAV（无损格式，适合进一步分析）
置顶半透明：窗口始终置顶，88% 透明度，可拖动，双击切换置顶
使用方法
启动程序，自动连接系统音频 Loopback 设备
在游戏中关闭背景音乐和音效，只保留宠物叫声
点击 录制A段 → 游戏播放基底音频 → 点击停止
点击 录制B段 → 游戏播放变调音频 → 点击停止
可选录制 C段 进行更多对比
自动显示对比结果：各段音调及音分差
点击 导出 按钮保存有效音频为 WAV 文件
操作提示
拖动窗口：按住窗口任意位置拖动
切换置顶：双击窗口
最小化：点击标题栏 - 按钮
关闭：点击标题栏 x 按钮
音调算法
检测方法：自相关法 + FFT 辅助验证，抛物线插值精化
综合音调：RMS 加权平均（高能量帧权重更大，排除平坦段干扰）
最低/最高音调：P5/P95 百分位（去除极端离群值）
静音裁剪：以峰值 RMS 为基准，低于 -40dB 的帧标记为静音，自动裁剪首尾
运行环境
Windows 10/11（需要 WASAPI Loopback 支持）
Python 3.10+
从源码运行
pip install numpy pyaudiowpatch -i https://pypi.tuna.tsinghua.edu.cn/simple
python pitch_monitor.py
打包为 EXE
pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m PyInstaller --noconfirm --onefile --windowed --name "音调对比工具" pitch_monitor.py
生成的 EXE 位于 dist/音调对比工具.exe。

依赖
numpy — 音频数据处理
pyaudiowpatch — WASAPI Loopback 音频捕获
tkinter — GUI（Python 内置）
