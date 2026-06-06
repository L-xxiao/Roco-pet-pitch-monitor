"""
音调对比工具 - 置顶半透明窗口
功能：
1. 录制A段：录制第一段音频，检测音调
2. 录制B段：录制第二段音频，检测音调
3. 自动对比A段与B段的音调差异（音分）

使用方法：
- 拖动标题栏移动窗口
- 双击标题栏切换置顶
- 点击关闭按钮退出
"""

import numpy as np
from collections import deque
import wave
import struct

import pyaudiowpatch as pyaudio

from tkinter import (
    Tk, Frame, Button, Label, StringVar,
    Toplevel, Listbox, Canvas
)
from tkinter.font import Font

# ============ 音调检测核心 ============

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def freq_to_note(freq):
    """将频率转换为音名和偏移量(音分)"""
    if freq <= 0:
        return "--", 0
    semitone = 12 * np.log2(freq / 440.0)
    midi = int(round(semitone)) + 69
    note_idx = midi % 12
    octave = (midi // 12) - 1
    cents = (semitone - round(semitone)) * 100
    return f"{NOTE_NAMES[note_idx]}{octave}", cents


def detect_pitch_autocorrelation(audio_data, sample_rate, min_freq=50, max_freq=2000):
    """使用自相关法检测音高"""
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    data = audio_data.astype(np.float64)

    rms = np.sqrt(np.mean(data ** 2))
    if rms < 0.001:
        return 0.0, 0.0

    data = data / (np.max(np.abs(data)) + 1e-10)

    n = len(data)
    min_lag = max(1, int(sample_rate / max_freq))
    max_lag = min(n - 1, int(sample_rate / min_freq))

    if max_lag <= min_lag:
        return 0.0, 0.0

    corr = np.correlate(data, data, mode='full')
    corr = corr[n - 1:]

    corr_segment = corr[min_lag:max_lag + 1]
    if len(corr_segment) == 0:
        return 0.0, 0.0

    peak_idx = np.argmax(corr_segment)
    lag = peak_idx + min_lag

    # 抛物线插值
    if 1 <= peak_idx < len(corr_segment) - 1:
        alpha = corr_segment[peak_idx - 1]
        beta = corr_segment[peak_idx]
        gamma = corr_segment[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            p = 0.5 * (alpha - gamma) / denom
            lag = lag + p

    if lag <= 0:
        return 0.0, 0.0

    freq = sample_rate / lag
    confidence = corr[peak_idx + min_lag] / (corr[0] + 1e-10)

    return freq, min(confidence, 1.0)


def detect_pitch_fft(audio_data, sample_rate, min_freq=50, max_freq=2000):
    """使用FFT检测音高（辅助验证）"""
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    data = audio_data.astype(np.float64)
    rms = np.sqrt(np.mean(data ** 2))
    if rms < 0.001:
        return 0.0, 0.0

    n = len(data)
    window = np.hanning(n)
    data = data * window

    fft_result = np.fft.rfft(data)
    magnitudes = np.abs(fft_result)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return 0.0, 0.0

    masked_mags = magnitudes[mask]
    masked_freqs = freqs[mask]

    peak_idx = np.argmax(masked_mags)
    freq = masked_freqs[peak_idx]
    confidence = masked_mags[peak_idx] / (np.max(masked_mags) + 1e-10)

    return freq, confidence


def detect_pitch(audio_data, sample_rate):
    """综合音高检测：自相关为主，FFT辅助验证"""
    freq_ac, conf_ac = detect_pitch_autocorrelation(audio_data, sample_rate)
    freq_fft, conf_fft = detect_pitch_fft(audio_data, sample_rate)

    if conf_ac < 0.3 and conf_fft > 0.5:
        return freq_fft, conf_fft

    return freq_ac, conf_ac


def trim_silence(audio_data, sample_rate, threshold_db=-40, min_silence_ms=80):
    """自动裁剪首尾静音段，返回有效音频区域 (start_sample, end_sample)"""
    if len(audio_data.shape) > 1:
        mono = audio_data.mean(axis=1)
    else:
        mono = audio_data

    mono = mono.astype(np.float64)

    frame_size = int(sample_rate * 0.01)  # 10ms每帧
    n_frames = len(mono) // frame_size

    if n_frames < 2:
        return 0, len(mono)

    rms_values = np.zeros(n_frames)
    for i in range(n_frames):
        frame = mono[i * frame_size:(i + 1) * frame_size]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))

    peak_rms = np.max(rms_values)
    if peak_rms < 1e-6:
        return 0, 0

    threshold = peak_rms * (10 ** (threshold_db / 20.0))
    is_loud = rms_values > threshold

    min_silence_frames = max(1, int(min_silence_ms / 10))

    # 找第一个有声帧
    start_frame = 0
    for i in range(n_frames):
        if i + min_silence_frames <= n_frames:
            if np.all(is_loud[i:i + min_silence_frames]):
                start_frame = i
                break
    else:
        loud_indices = np.where(is_loud)[0]
        if len(loud_indices) > 0:
            start_frame = loud_indices[0]
        else:
            return 0, 0

    # 找最后一个有声帧
    end_frame = n_frames - 1
    for i in range(n_frames - 1, -1, -1):
        if i - min_silence_frames >= 0:
            if np.all(is_loud[i - min_silence_frames + 1:i + 1]):
                end_frame = i
                break
    else:
        loud_indices = np.where(is_loud)[0]
        if len(loud_indices) > 0:
            end_frame = loud_indices[-1]
        else:
            return 0, 0

    start_sample = max(0, start_frame * frame_size)
    end_sample = min(len(mono), (end_frame + 1) * frame_size)

    return start_sample, end_sample


def analyze_audio_pitch(audio_data, sample_rate):
    """分析整段音频的音高，自动裁剪首尾静音
    返回(中位频率, 音高列表, 置信度列表, 裁剪信息, 最低频率, 最高频率)
    """
    if len(audio_data.shape) > 1:
        mono = audio_data.mean(axis=1)
    else:
        mono = audio_data.copy()

    start, end = trim_silence(mono, sample_rate)
    trimmed = mono[start:end]

    if len(trimmed) < 1024:
        return 0.0, [], [], (start, end, len(mono)), 0.0, 0.0

    trimmed_duration = len(trimmed) / sample_rate
    original_duration = len(mono) / sample_rate
    trim_info = (start, end, len(mono), trimmed_duration, original_duration)

    chunk_size = int(sample_rate * 0.05)
    if chunk_size < 1024:
        chunk_size = 1024

    pitches = []
    confidences = []
    rms_values = []
    for i in range(0, len(trimmed) - chunk_size, chunk_size):
        chunk = trimmed[i:i + chunk_size]
        freq, conf = detect_pitch(chunk, sample_rate)
        chunk_rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        if freq > 0 and conf > 0.2:
            pitches.append(freq)
            confidences.append(conf)
            rms_values.append(chunk_rms)

    if pitches:
        pitches = np.array(pitches)
        rms_values = np.array(rms_values)

        # 按RMS过滤：只保留能量高于中位数50%的帧，排除平坦段干扰
        median_rms = np.median(rms_values)
        rms_threshold = median_rms * 0.5
        mask = rms_values >= rms_threshold

        filtered_pitches = pitches[mask]
        filtered_rms = rms_values[mask]

        if len(filtered_pitches) == 0:
            filtered_pitches = pitches
            filtered_rms = rms_values

        # RMS加权平均作为综合音调
        weights = filtered_rms ** 2
        weighted_freq = float(np.average(filtered_pitches, weights=weights))

        # 最低/最高用过滤后的P5/P95
        sorted_fp = sorted(filtered_pitches)
        p5 = sorted_fp[max(0, int(len(sorted_fp) * 0.05))]
        p95 = sorted_fp[min(len(sorted_fp) - 1, int(len(sorted_fp) * 0.95))]

        return weighted_freq, list(pitches), confidences, trim_info, float(p5), float(p95)
    return 0.0, [], [], trim_info, 0.0, 0.0


# ============ WASAPI Loopback 音频捕获 ============

class AudioCapture:
    """WASAPI Loopback 捕获系统音频"""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.is_capturing = False
        self.sample_rate = 48000
        self.channels = 2
        self.chunk_size = 512
        self._buffer = deque(maxlen=200)
        self._recorded_data = []
        self._lock = __import__('threading').Lock()
        self._device_name = ""

    def get_loopback_devices(self):
        try:
            return list(self.pa.get_loopback_device_info_generator())
        except Exception:
            return []

    def start_loopback_capture(self, device=None):
        if device is None:
            try:
                default_lb = self.pa.get_default_wasapi_loopback()
                return self._open_device(default_lb)
            except Exception:
                pass

        if device is not None:
            try:
                return self._open_device(device)
            except Exception:
                pass

        devices = self.get_loopback_devices()
        for dev in devices:
            try:
                if self._open_device(dev):
                    return True
            except Exception:
                continue
        return False

    def _open_device(self, device_info):
        self.sample_rate = int(device_info['defaultSampleRate'])
        self.channels = device_info['maxInputChannels']
        if self.channels <= 0:
            self.channels = 2
        self._device_name = device_info.get('name', 'Unknown')

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_info['index'],
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )
        self.is_capturing = True
        self.stream.start_stream()
        return True

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            audio_array = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self.channels)
            with self._lock:
                self._buffer.append(audio_array.copy())
                if self._recorded_data is not None:
                    self._recorded_data.append(audio_array.copy())
        except Exception:
            pass
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        with self._lock:
            self._recorded_data = []

    def stop_recording(self):
        with self._lock:
            if self._recorded_data:
                data = np.concatenate(self._recorded_data, axis=0)
            else:
                data = np.zeros((1, self.channels), dtype=np.int16)
            self._recorded_data = None
        return data, self.sample_rate

    def stop(self):
        self.is_capturing = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        self.pa.terminate()


# ============ 设备选择对话框 ============

class DeviceSelectDialog:
    def __init__(self, parent, devices):
        self.result = None
        self.dialog = Toplevel(parent)
        self.dialog.title("选择音频设备")
        self.dialog.geometry("450x350")
        self.dialog.configure(bg='#1a1a2e')
        self.dialog.attributes('-topmost', True)
        self.dialog.grab_set()
        self.dialog.transient(parent)

        Label(self.dialog, text="选择WASAPI Loopback设备", bg='#1a1a2e', fg='#e0e0e0',
              font=Font(family='Microsoft YaHei', size=11, weight='bold')).pack(pady=10)

        self.listbox = Listbox(self.dialog, bg='#0f3460', fg='#e0e0e0',
                               selectbackground='#e94560', selectforeground='white',
                               font=Font(family='Consolas', size=10),
                               relief='flat', bd=0)
        self.listbox.pack(fill='both', expand=True, padx=15, pady=5)

        for d in devices:
            sr = int(d['defaultSampleRate'])
            ch = d['maxInputChannels']
            self.listbox.insert('end', f"  {d['name']}  [{sr}Hz / {ch}ch]")

        if devices:
            self.listbox.selection_set(0)

        btn_frame = Frame(self.dialog, bg='#1a1a2e')
        btn_frame.pack(fill='x', padx=15, pady=10)

        Button(btn_frame, text="确定", bg='#0f3460', fg='white',
               font=Font(family='Microsoft YaHei', size=10),
               relief='flat', padx=20, pady=4,
               command=lambda: self._select(devices)).pack(side='right', padx=5)
        Button(btn_frame, text="取消", bg='#333333', fg='white',
               font=Font(family='Microsoft YaHei', size=10),
               relief='flat', padx=20, pady=4,
               command=self._cancel).pack(side='right', padx=5)

        self.listbox.bind('<Double-Button-1>', lambda e: self._select(devices))

    def _select(self, devices):
        sel = self.listbox.curselection()
        if sel:
            self.result = devices[sel[0]]
        self.dialog.destroy()

    def _cancel(self):
        self.result = None
        self.dialog.destroy()


# ============ GUI 主界面 ============

class PitchMonitorApp:
    """音调对比主窗口"""

    def __init__(self):
        self.root = Tk()
        self.root.title("音调对比")
        self.root.geometry("440x620+100+100")
        self.root.configure(bg='#1a1a2e')

        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.88)
        self.root.overrideredirect(True)

        self._drag_x = 0
        self._drag_y = 0
        self.root.bind('<Button-1>', self._start_drag)
        self.root.bind('<B1-Motion>', self._do_drag)
        self.root.bind('<Double-Button-1>', self._toggle_topmost)

        # 音频捕获
        self.capture = AudioCapture()
        self.capture_started = False

        # A段和B段数据
        self.seg_a_pitch = 0.0
        self.seg_a_note = "--"
        self.seg_a_cents = 0
        self.seg_a_duration = 0.0
        self.seg_a_min_pitch = 0.0
        self.seg_a_max_pitch = 0.0

        self.seg_b_pitch = 0.0
        self.seg_b_note = "--"
        self.seg_b_cents = 0
        self.seg_b_duration = 0.0
        self.seg_b_min_pitch = 0.0
        self.seg_b_max_pitch = 0.0

        # C段数据
        self.seg_c_pitch = 0.0
        self.seg_c_note = "--"
        self.seg_c_cents = 0
        self.seg_c_duration = 0.0
        self.seg_c_min_pitch = 0.0
        self.seg_c_max_pitch = 0.0

        # 存储原始音频数据用于绘制波形
        self.seg_a_audio = None
        self.seg_a_sr = 0
        self.seg_a_trim = (0, 0)
        self.seg_b_audio = None
        self.seg_b_sr = 0
        self.seg_b_trim = (0, 0)
        self.seg_c_audio = None
        self.seg_c_sr = 0
        self.seg_c_trim = (0, 0)

        # 录制状态: None / 'a' / 'b' / 'c'
        self.recording_target = None
        self.record_start_time = 0

        self._build_ui()
        self.root.after(100, self._select_device_and_start)

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _do_drag(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _toggle_topmost(self, event):
        current = self.root.attributes('-topmost')
        self.root.attributes('-topmost', not current)

    def _select_device_and_start(self):
        try:
            default_lb = self.capture.pa.get_default_wasapi_loopback()
            self._start_with_device(default_lb)
            return
        except Exception:
            pass

        devices = self.capture.get_loopback_devices()
        if not devices:
            self.status_var.set("未找到loopback设备！")
            return

        if len(devices) == 1:
            self._start_with_device(devices[0])
            return

        dialog = DeviceSelectDialog(self.root, devices)
        self.root.wait_window(dialog.dialog)
        if dialog.result:
            self._start_with_device(dialog.result)
        else:
            self._start_with_device(devices[0])

    def _start_with_device(self, device_info):
        try:
            success = self.capture.start_loopback_capture(device_info)
            if success:
                self.capture_started = True
                short_name = self.capture._device_name.split('[')[0].strip()
                self.status_var.set(f"已连接: {short_name} | 双击切换置顶")
            else:
                self.status_var.set("音频捕获启动失败！")
        except Exception as e:
            self.status_var.set(f"音频捕获错误: {e}")

    def _build_ui(self):
        main_frame = Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill='both', expand=True, padx=8, pady=8)

        # === 标题栏 ===
        title_frame = Frame(main_frame, bg='#16213e')
        title_frame.pack(fill='x', pady=(0, 5))

        Label(title_frame, text="音调对比工具", bg='#16213e', fg='#e0e0e0',
              font=Font(family='Microsoft YaHei', size=9)).pack(side='left', padx=8, pady=4)

        min_btn = Label(title_frame, text="-", bg='#16213e', fg='#ffdd57',
                        font=Font(size=10, weight='bold'), cursor='hand2')
        min_btn.pack(side='right', padx=4, pady=4)
        min_btn.bind('<Button-1>', lambda e: self.root.iconify())

        close_btn = Label(title_frame, text="x", bg='#16213e', fg='#ff6b6b',
                          font=Font(size=10, weight='bold'), cursor='hand2')
        close_btn.pack(side='right', padx=4, pady=4)
        close_btn.bind('<Button-1>', lambda e: self._on_close())

        # === A段信息 ===
        seg_a_frame = Frame(main_frame, bg='#1a1a2e', highlightbackground='#e94560',
                            highlightthickness=1)
        seg_a_frame.pack(fill='x', pady=2)

        self.seg_a_var = StringVar(value="A段: 未录制")
        Label(seg_a_frame, textvariable=self.seg_a_var, bg='#1a1a2e', fg='#ff8a80',
              font=Font(family='Microsoft YaHei', size=9, weight='bold'),
              anchor='w').pack(fill='x', padx=8, pady=(4, 0))

        self.seg_a_range_var = StringVar(value="")
        Label(seg_a_frame, textvariable=self.seg_a_range_var, bg='#1a1a2e', fg='#cc8888',
              font=Font(family='Consolas', size=8),
              anchor='w').pack(fill='x', padx=8, pady=(0, 4))

        # === B段信息 ===
        seg_b_frame = Frame(main_frame, bg='#1a1a2e', highlightbackground='#4fc3f7',
                            highlightthickness=1)
        seg_b_frame.pack(fill='x', pady=2)

        self.seg_b_var = StringVar(value="B段: 未录制")
        Label(seg_b_frame, textvariable=self.seg_b_var, bg='#1a1a2e', fg='#80d8ff',
              font=Font(family='Microsoft YaHei', size=9, weight='bold'),
              anchor='w').pack(fill='x', padx=8, pady=(4, 0))

        self.seg_b_range_var = StringVar(value="")
        Label(seg_b_frame, textvariable=self.seg_b_range_var, bg='#1a1a2e', fg='#6699bb',
              font=Font(family='Consolas', size=8),
              anchor='w').pack(fill='x', padx=8, pady=(0, 4))

        # === C段信息 ===
        seg_c_frame = Frame(main_frame, bg='#1a1a2e', highlightbackground='#66bb6a',
                            highlightthickness=1)
        seg_c_frame.pack(fill='x', pady=2)

        self.seg_c_var = StringVar(value="C段: 未录制")
        Label(seg_c_frame, textvariable=self.seg_c_var, bg='#1a1a2e', fg='#a5d6a7',
              font=Font(family='Microsoft YaHei', size=9, weight='bold'),
              anchor='w').pack(fill='x', padx=8, pady=(4, 0))

        self.seg_c_range_var = StringVar(value="")
        Label(seg_c_frame, textvariable=self.seg_c_range_var, bg='#1a1a2e', fg='#77aa77',
              font=Font(family='Consolas', size=8),
              anchor='w').pack(fill='x', padx=8, pady=(0, 4))

        # === 对比结果 ===
        compare_frame = Frame(main_frame, bg='#0f3460')
        compare_frame.pack(fill='x', pady=5)

        self.compare_var = StringVar(value="录制段落后自动对比")
        self.compare_label = Label(compare_frame, textvariable=self.compare_var,
                                   bg='#0f3460', fg='#aaaaaa',
                                   font=Font(family='Microsoft YaHei', size=14, weight='bold'))
        self.compare_label.pack(pady=6)

        self.compare_detail_var = StringVar(value="")
        Label(compare_frame, textvariable=self.compare_detail_var,
              bg='#0f3460', fg='#888888',
              font=Font(family='Consolas', size=9)).pack(pady=(0, 6))

        # === 按钮区域 ===
        btn_row = Frame(main_frame, bg='#1a1a2e')
        btn_row.pack(fill='x', pady=(5, 2))

        btn_font = Font(family='Microsoft YaHei', size=9, weight='bold')

        self.btn_a = Button(btn_row, text="● 录制A段", font=btn_font,
                            bg='#e94560', fg='white', activebackground='#c73e54',
                            relief='flat', cursor='hand2', padx=6, pady=5,
                            command=self._toggle_record_a)
        self.btn_a.pack(side='left', padx=3, expand=True, fill='x')

        self.btn_c = Button(btn_row, text="● 录制C段", font=btn_font,
                            bg='#66bb6a', fg='white', activebackground='#55a55a',
                            relief='flat', cursor='hand2', padx=6, pady=5,
                            command=self._toggle_record_c)
        self.btn_c.pack(side='left', padx=3, expand=True, fill='x')

        self.btn_b = Button(btn_row, text="● 录制B段", font=btn_font,
                            bg='#0288d1', fg='white', activebackground='#0277bd',
                            relief='flat', cursor='hand2', padx=6, pady=5,
                            command=self._toggle_record_b)
        self.btn_b.pack(side='left', padx=3, expand=True, fill='x')

        # === 波形图区域 ===
        wave_frame = Frame(main_frame, bg='#1a1a2e')
        wave_frame.pack(fill='x', pady=(5, 2))

        # A段波形
        wave_a_row = Frame(wave_frame, bg='#1a1a2e')
        wave_a_row.pack(fill='x', padx=4)

        wave_a_left = Frame(wave_a_row, bg='#1a1a2e')
        wave_a_left.pack(side='left', fill='both', expand=True)

        Label(wave_a_left, text="A段波形", bg='#1a1a2e', fg='#ff8a80',
              font=Font(family='Microsoft YaHei', size=8)).pack(anchor='w')

        self.canvas_a = Canvas(wave_a_left, height=55, bg='#0d1b2a',
                               highlightthickness=1, highlightbackground='#e94560')
        self.canvas_a.pack(fill='x', pady=(0, 4))

        self.btn_export_a = Button(wave_a_row, text="导出\nA段", font=Font(family='Microsoft YaHei', size=8),
                                   bg='#333355', fg='#ff8a80', activebackground='#444466',
                                   relief='flat', cursor='hand2', padx=6, pady=2,
                                   command=lambda: self._export_wav('a'))
        self.btn_export_a.pack(side='right', padx=(4, 0), pady=(12, 0))

        # B段波形
        wave_b_row = Frame(wave_frame, bg='#1a1a2e')
        wave_b_row.pack(fill='x', padx=4)

        wave_b_left = Frame(wave_b_row, bg='#1a1a2e')
        wave_b_left.pack(side='left', fill='both', expand=True)

        Label(wave_b_left, text="B段波形", bg='#1a1a2e', fg='#80d8ff',
              font=Font(family='Microsoft YaHei', size=8)).pack(anchor='w')

        self.canvas_b = Canvas(wave_b_left, height=55, bg='#0d1b2a',
                               highlightthickness=1, highlightbackground='#4fc3f7')
        self.canvas_b.pack(fill='x', pady=(0, 4))

        self.btn_export_b = Button(wave_b_row, text="导出\nB段", font=Font(family='Microsoft YaHei', size=8),
                                   bg='#333355', fg='#80d8ff', activebackground='#444466',
                                   relief='flat', cursor='hand2', padx=6, pady=2,
                                   command=lambda: self._export_wav('b'))
        self.btn_export_b.pack(side='right', padx=(4, 0), pady=(12, 0))

        # C段波形
        wave_c_row = Frame(wave_frame, bg='#1a1a2e')
        wave_c_row.pack(fill='x', padx=4)

        wave_c_left = Frame(wave_c_row, bg='#1a1a2e')
        wave_c_left.pack(side='left', fill='both', expand=True)

        Label(wave_c_left, text="C段波形", bg='#1a1a2e', fg='#a5d6a7',
              font=Font(family='Microsoft YaHei', size=8)).pack(anchor='w')

        self.canvas_c = Canvas(wave_c_left, height=55, bg='#0d1b2a',
                               highlightthickness=1, highlightbackground='#66bb6a')
        self.canvas_c.pack(fill='x', pady=(0, 4))

        self.btn_export_c = Button(wave_c_row, text="导出\nC段", font=Font(family='Microsoft YaHei', size=8),
                                   bg='#333355', fg='#a5d6a7', activebackground='#444466',
                                   relief='flat', cursor='hand2', padx=6, pady=2,
                                   command=lambda: self._export_wav('c'))
        self.btn_export_c.pack(side='right', padx=(4, 0), pady=(12, 0))

        # === 状态栏 ===
        self.status_var = StringVar(value="正在初始化...")
        Label(main_frame, textvariable=self.status_var, bg='#1a1a2e', fg='#666666',
              font=Font(family='Microsoft YaHei', size=8)).pack(fill='x')

    # === 录制逻辑 ===

    def _toggle_record_a(self):
        if self.recording_target and self.recording_target != 'a':
            self.status_var.set(f"请先停止{self.recording_target.upper()}段录制")
            return
        if self.recording_target == 'a':
            self._stop_recording('a')
        else:
            self._start_recording('a')

    def _toggle_record_b(self):
        if self.recording_target and self.recording_target != 'b':
            self.status_var.set(f"请先停止{self.recording_target.upper()}段录制")
            return
        if self.recording_target == 'b':
            self._stop_recording('b')
        else:
            self._start_recording('b')

    def _toggle_record_c(self):
        if self.recording_target and self.recording_target != 'c':
            self.status_var.set(f"请先停止{self.recording_target.upper()}段录制")
            return
        if self.recording_target == 'c':
            self._stop_recording('c')
        else:
            self._start_recording('c')

    def _start_recording(self, target):
        if not self.capture_started:
            self.status_var.set("请等待音频捕获启动...")
            return

        self.recording_target = target
        self.record_start_time = __import__('time').time()
        self.capture.start_recording()

        btn_map = {'a': self.btn_a, 'b': self.btn_b, 'c': self.btn_c}
        other_map = {'a': [self.btn_b, self.btn_c], 'b': [self.btn_a, self.btn_c], 'c': [self.btn_a, self.btn_b]}
        color_map = {'a': '#e94560', 'b': '#0288d1', 'c': '#66bb6a'}

        btn_map[target].configure(text=f"■ 停止录制", bg='#ff8800')
        for btn in other_map[target]:
            btn.configure(state='disabled')
        self.status_var.set(f"正在录制{target.upper()}段...")

        self._update_record_timer()

    def _stop_recording(self, target):
        import time as _time
        self.recording_target = None
        audio_data, sr = self.capture.stop_recording()
        total_duration = len(audio_data) / sr if sr > 0 else 0

        btn_map = {'a': self.btn_a, 'b': self.btn_b, 'c': self.btn_c}
        color_map = {'a': '#e94560', 'b': '#0288d1', 'c': '#66bb6a'}
        label_map = {'a': 'A段', 'b': 'B段', 'c': 'C段'}
        var_map = {'a': self.seg_a_var, 'b': self.seg_b_var, 'c': self.seg_c_var}
        range_var_map = {'a': self.seg_a_range_var, 'b': self.seg_b_range_var, 'c': self.seg_c_range_var}

        btn_map[target].configure(text=f"● 录制{target.upper()}段", bg=color_map[target])
        for t, btn in btn_map.items():
            if t != target:
                btn.configure(state='normal')

        if len(audio_data) > 0 and sr > 0:
            result = analyze_audio_pitch(audio_data, sr)
            median_pitch, pitches, _, trim_info, min_pitch, max_pitch = result
            start_trim, end_trim = trim_info[0], trim_info[1]

            # 存储音频数据
            if target == 'a':
                self.seg_a_audio = audio_data
                self.seg_a_sr = sr
                self.seg_a_trim = (start_trim, end_trim)
            elif target == 'b':
                self.seg_b_audio = audio_data
                self.seg_b_sr = sr
                self.seg_b_trim = (start_trim, end_trim)
            else:
                self.seg_c_audio = audio_data
                self.seg_c_sr = sr
                self.seg_c_trim = (start_trim, end_trim)

            if median_pitch > 0:
                note, cents = freq_to_note(median_pitch)
                min_note, _ = freq_to_note(min_pitch)
                max_note, _ = freq_to_note(max_pitch)
                trimmed_dur = trim_info[3] if len(trim_info) > 3 else total_duration

                # 存储音调数据
                if target == 'a':
                    self.seg_a_pitch = median_pitch
                    self.seg_a_note = note
                    self.seg_a_cents = cents
                    self.seg_a_duration = trimmed_dur
                    self.seg_a_min_pitch = min_pitch
                    self.seg_a_max_pitch = max_pitch
                elif target == 'b':
                    self.seg_b_pitch = median_pitch
                    self.seg_b_note = note
                    self.seg_b_cents = cents
                    self.seg_b_duration = trimmed_dur
                    self.seg_b_min_pitch = min_pitch
                    self.seg_b_max_pitch = max_pitch
                else:
                    self.seg_c_pitch = median_pitch
                    self.seg_c_note = note
                    self.seg_c_cents = cents
                    self.seg_c_duration = trimmed_dur
                    self.seg_c_min_pitch = min_pitch
                    self.seg_c_max_pitch = max_pitch

                var_map[target].set(
                    f"{label_map[target]}: {note}  {median_pitch:.1f} Hz  "
                    f"(有效{trimmed_dur:.1f}s/{total_duration:.1f}s)")
                range_var_map[target].set(
                    f"  最低: {min_pitch:.1f}Hz({min_note})  "
                    f"最高: {max_pitch:.1f}Hz({max_note})  "
                    f"波动: {max_pitch - min_pitch:.1f}Hz")
                self.status_var.set(
                    f"{label_map[target]}完成 | 已自动裁剪静音 | 有效音频 {trimmed_dur:.1f}s")
            else:
                if target == 'a':
                    self.seg_a_pitch = 0
                elif target == 'b':
                    self.seg_b_pitch = 0
                else:
                    self.seg_c_pitch = 0
                var_map[target].set(f"{label_map[target]}: 未检测到音调")
                range_var_map[target].set("")
                self.status_var.set(f"{label_map[target]}未检测到明显音调（可能录制为静音）")

            self.root.after(50, self._redraw_both_waveforms)
        else:
            var_map[target].set(f"{label_map[target]}: 录制为空")
            range_var_map[target].set("")

        self._update_comparison()

    def _update_record_timer(self):
        import time as _time
        if self.recording_target:
            elapsed = _time.time() - self.record_start_time
            btn_map = {'a': self.btn_a, 'b': self.btn_b, 'c': self.btn_c}
            btn_map[self.recording_target].configure(text=f"■ 录制中 {elapsed:.0f}s")
            self.root.after(200, self._update_record_timer)

    def _update_comparison(self):
        """更新A/B/C段对比结果"""
        segments = []
        if self.seg_a_pitch > 0:
            segments.append(('A', self.seg_a_pitch, self.seg_a_min_pitch, self.seg_a_max_pitch))
        if self.seg_b_pitch > 0:
            segments.append(('B', self.seg_b_pitch, self.seg_b_min_pitch, self.seg_b_max_pitch))
        if self.seg_c_pitch > 0:
            segments.append(('C', self.seg_c_pitch, self.seg_c_min_pitch, self.seg_c_max_pitch))

        if len(segments) < 2:
            self.compare_var.set("录制段落后自动对比")
            self.compare_label.configure(fg='#aaaaaa')
            self.compare_detail_var.set("")
            return

        # 找最低和最高
        segments.sort(key=lambda x: x[1])
        lowest = segments[0]
        highest = segments[-1]

        diff_semitones = 12 * np.log2(highest[1] / lowest[1])
        diff_cents = diff_semitones * 100

        if abs(diff_cents) < 10:
            self.compare_var.set("所有段音调基本一致")
            self.compare_label.configure(fg='#00ff88')
        else:
            self.compare_var.set(f"{highest[0]}比{lowest[0]}高 {abs(diff_cents):.1f}¢")
            self.compare_label.configure(fg='#ff6b6b')

        # 详细对比：每对之间的音分差
        details = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                n1, f1, _, _ = segments[i]
                n2, f2, _, _ = segments[j]
                cents = 12 * np.log2(f2 / f1) * 100
                direction = "↑" if cents > 0 else "↓"
                details.append(f"{n1}→{n2}:{direction}{abs(cents):.0f}¢")

        self.compare_detail_var.set("  ".join(details))

    def _export_wav(self, target):
        """导出有效音频为WAV文件（16bit PCM，无损，最适合分析）"""
        data_map = {
            'a': (self.seg_a_audio, self.seg_a_sr, self.seg_a_trim, "A段"),
            'b': (self.seg_b_audio, self.seg_b_sr, self.seg_b_trim, "B段"),
            'c': (self.seg_c_audio, self.seg_c_sr, self.seg_c_trim, "C段"),
        }
        audio_data, sr, trim, name = data_map[target]

        if audio_data is None or sr == 0:
            self.status_var.set(f"{name}无数据，无法导出")
            return

        start, end = trim
        trimmed = audio_data[start:end]

        if len(trimmed) == 0:
            self.status_var.set(f"{name}有效音频为空，无法导出")
            return

        # 弹出保存对话框
        from tkinter import filedialog
        import time as _time
        default_name = f"{name}_trimmed_{_time.strftime('%H%M%S')}.wav"
        filepath = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".wav",
            filetypes=[("WAV音频", "*.wav"), ("所有文件", "*.*")],
            title=f"导出{name}有效音频"
        )

        if not filepath:
            return

        try:
            # 转为单声道int16
            if len(trimmed.shape) > 1:
                mono = trimmed.mean(axis=1)
            else:
                mono = trimmed.copy()

            # 确保是int16范围
            mono = np.clip(mono, -32768, 32767).astype(np.int16)

            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16bit = 2 bytes
                wf.setframerate(sr)
                wf.writeframes(mono.tobytes())

            duration = len(mono) / sr
            self.status_var.set(f"已导出{name}: {filepath} ({duration:.2f}s, {sr}Hz, 16bit单声道WAV)")
        except Exception as e:
            self.status_var.set(f"导出失败: {e}")

    def _get_common_pps(self):
        """计算A/B/C段共用的像素/秒比例，保证相同时长显示相同宽度"""
        max_dur = 0
        for seg_audio, seg_sr, seg_trim in [
            (self.seg_a_audio, self.seg_a_sr, self.seg_a_trim),
            (self.seg_b_audio, self.seg_b_sr, self.seg_b_trim),
            (self.seg_c_audio, self.seg_c_sr, self.seg_c_trim),
        ]:
            if seg_audio is not None and seg_sr > 0:
                dur = (seg_trim[1] - seg_trim[0]) / seg_sr
                if dur > max_dur:
                    max_dur = dur
        if max_dur <= 0:
            return 100
        canvas_w = max(self.canvas_a.winfo_width(), self.canvas_b.winfo_width(),
                       self.canvas_c.winfo_width(), 400)
        return (canvas_w * 0.8) / max_dur

    def _redraw_both_waveforms(self):
        """用统一比例重绘所有波形"""
        pps = self._get_common_pps()
        if self.seg_a_audio is not None:
            self._draw_waveform(self.canvas_a, self.seg_a_audio, self.seg_a_sr,
                                self.seg_a_trim, '#e94560', pps)
        if self.seg_b_audio is not None:
            self._draw_waveform(self.canvas_b, self.seg_b_audio, self.seg_b_sr,
                                self.seg_b_trim, '#4fc3f7', pps)
        if self.seg_c_audio is not None:
            self._draw_waveform(self.canvas_c, self.seg_c_audio, self.seg_c_sr,
                                self.seg_c_trim, '#66bb6a', pps)

    def _draw_waveform(self, canvas, audio_data, sample_rate, trim_range, color, pps=100):
        """在Canvas上绘制波形，有效区域居中显示，使用统一时间比例"""
        canvas.delete('all')
        canvas.update_idletasks()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            return

        if audio_data is None or len(audio_data) == 0:
            canvas.create_text(w // 2, h // 2, text="无数据", fill='#444444',
                               font=Font(family='Microsoft YaHei', size=8))
            return

        # 转单声道
        if len(audio_data.shape) > 1:
            mono = audio_data.mean(axis=1)
        else:
            mono = audio_data.copy()
        mono = mono.astype(np.float64)

        total_samples = len(mono)
        mid_y = h / 2
        start_trim, end_trim = trim_range
        trim_samples = end_trim - start_trim
        trim_duration = trim_samples / sample_rate if sample_rate > 0 else 0

        # 有效区域的像素宽度（基于统一比例）
        effective_px = int(trim_duration * pps)
        effective_px = max(10, min(effective_px, int(w * 0.9)))

        # 有效区域居中：计算起始x
        x_offset = (w - effective_px) / 2

        # 绘制中线
        canvas.create_line(0, mid_y, w, mid_y, fill='#222233', width=1)

        dim_color = '#1a2a3a'

        # 绘制有效区域波形
        if trim_samples > 0:
            num_pixels = effective_px
            samples_per_pixel = max(1, trim_samples // num_pixels)

            min_vals = np.zeros(num_pixels)
            max_vals = np.zeros(num_pixels)
            for i in range(num_pixels):
                s = start_trim + i * samples_per_pixel
                e = min(s + samples_per_pixel, end_trim)
                segment = mono[s:e]
                if len(segment) > 0:
                    min_vals[i] = np.min(segment)
                    max_vals[i] = np.max(segment)

            peak = max(np.max(np.abs(min_vals)), np.max(np.abs(max_vals)), 1e-6)
            min_vals = min_vals / peak * (mid_y - 2)
            max_vals = max_vals / peak * (mid_y - 2)

            for i in range(num_pixels):
                x = x_offset + i
                y_min = mid_y - max_vals[i]
                y_max = mid_y - min_vals[i]
                canvas.create_line(x, y_min, x, y_max, fill=color, width=1)

        # 绘制前导静音（缩略显示在左侧）
        if start_trim > 0:
            silence_px = max(1, int(x_offset * 0.6))
            sp_per_px = max(1, start_trim // silence_px)
            for i in range(silence_px):
                s = i * sp_per_px
                e = min(s + sp_per_px, start_trim)
                segment = mono[s:e]
                if len(segment) > 0:
                    val_min = np.min(segment)
                    val_max = np.max(segment)
                    peak_s = max(abs(val_min), abs(val_max), 1e-6)
                    y1 = mid_y - (val_max / peak_s) * (mid_y - 2) * 0.3
                    y2 = mid_y - (val_min / peak_s) * (mid_y - 2) * 0.3
                    x = x_offset * i / silence_px
                    canvas.create_line(x, y1, x, y2, fill=dim_color, width=1)

        # 绘制尾部静音（缩略显示在右侧）
        if end_trim < total_samples:
            tail_start = end_trim
            tail_px = max(1, int((w - x_offset - effective_px) * 0.6))
            sp_per_px = max(1, (total_samples - tail_start) // tail_px)
            x_base = x_offset + effective_px
            tail_width = w - x_base
            for i in range(tail_px):
                s = tail_start + i * sp_per_px
                e = min(s + sp_per_px, total_samples)
                segment = mono[s:e]
                if len(segment) > 0:
                    val_min = np.min(segment)
                    val_max = np.max(segment)
                    peak_s = max(abs(val_min), abs(val_max), 1e-6)
                    y1 = mid_y - (val_max / peak_s) * (mid_y - 2) * 0.3
                    y2 = mid_y - (val_min / peak_s) * (mid_y - 2) * 0.3
                    x = x_base + tail_width * i / tail_px
                    canvas.create_line(x, y1, x, y2, fill=dim_color, width=1)

        # 绘制有效区域边界线
        canvas.create_line(x_offset, 0, x_offset, h, fill='#ffdd57',
                           width=1, dash=(2, 2))
        canvas.create_line(x_offset + effective_px, 0, x_offset + effective_px, h,
                           fill='#ffdd57', width=1, dash=(2, 2))

    def _on_close(self):
        self.capture.stop()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._on_close()


if __name__ == '__main__':
    app = PitchMonitorApp()
    app.run()
