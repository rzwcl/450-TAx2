# -*- coding: utf-8 -*-
"""
450 MHz FFSK / CIR450TrainNo live decoder GUI

功能：
- 从 Windows 声音输入设备读取音频（可选择虚拟音频线，例如 VAC/CABLE/Line 1）。
- 将输入 PCM 送入用户已有的 SoX 进程，实时下混并重采样为 48 kHz、单声道、16-bit PCM。
- 对 FFSK 1200/1800 Hz 进行判位，搜索 31-bit sync，BCH(26,16) 纠错，CRC-16/XMODEM 验证。
- 展示 40 字节 TAX 载荷、已知字段及公里标/站数/信号机编号候选。
- 可选择 WAV 离线测试，同样经 SoX 统一到 48 kHz 单声道后解析。

依赖：Python 3.9+，numpy，sounddevice（tkinter 一般随 Windows Python 安装）。
安装依赖：py -m pip install numpy sounddevice
SoX 请在界面中选择 sox.exe 路径，或提前放入 PATH。

字段说明：只有通过 BCH 与 CRC 的帧才展示；未验证的业务含义会明确标注为“候选”。
"""

from __future__ import annotations

import itertools
import os
import queue
import shutil
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

TARGET_RATE = 48000
SAMPLES_PER_BIT = 40
BAUD = 1200
FRAME_BITS = 31 + 23 * 26  # sync + 23 BCH codewords
FRAME_SECONDS = FRAME_BITS / BAUD
SYNC = f"{0x0DD4259F:032b}"[-31:]
SYNC_INV = "".join("1" if b == "0" else "0" for b in SYNC)
GEN_POLY = 0x5B9
RING_SECONDS = 2.2
SCAN_EVERY_SAMPLES = 12000  # 250 ms at 48 kHz


def polynomial_remainder(value: int, poly: int = GEN_POLY) -> int:
    """计算多项式整数除法的余数。最高位对应最高次幂。"""
    poly_bits = poly.bit_length()
    while value and value.bit_length() >= poly_bits:
        value ^= poly << (value.bit_length() - poly_bits)
    return value


def build_bch_error_table() -> dict[int, int]:
    """建立 BCH(26,16) 一位/两位错误 syndrome -> 错误掩码表。"""
    table: dict[int, int] = {}
    positions = range(26)
    masks = [1 << i for i in positions]
    masks.extend((1 << i) | (1 << j) for i, j in itertools.combinations(positions, 2))
    for mask in masks:
        syndrome = polynomial_remainder(mask)
        # 若一位/两位错误掩码产生相同 syndrome，则将该 syndrome 标成冲突，
        # 不擅自选择一个纠错方案；最终仍由整帧 CRC 决定是否接受。
        if syndrome == 0:
            continue
        if syndrome in table and table[syndrome] != mask:
            # 保留冲突标记，遇到冲突 syndrome 时拒绝盲目纠错。
            table[syndrome] = -1
        else:
            table[syndrome] = mask
    return table


BCH_ERROR_TABLE = build_bch_error_table()


def build_tone_projection(frequency: float) -> np.ndarray:
    """生成 cos/sin/DC 正交投影基，减轻 1800 Hz 半周期窗口的相位偏差。"""
    n = np.arange(SAMPLES_PER_BIT, dtype=np.float64)
    w = 2.0 * np.pi * frequency * n / TARGET_RATE
    matrix = np.column_stack((np.cos(w), np.sin(w), np.ones(SAMPLES_PER_BIT)))
    q, _ = np.linalg.qr(matrix)
    return q.astype(np.float32)


Q_1200 = build_tone_projection(1200.0)
Q_1800 = build_tone_projection(1800.0)


def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def bits_to_string(bits: np.ndarray) -> str:
    # 当前分析窗口不长；字符位流便于精确定位固定 31 位同步字。
    return "".join("1" if int(x) else "0" for x in bits)


def demodulate_phase(samples: np.ndarray, phase: int) -> str:
    """从一个候选采样相位生成 bit 字符串。样本必须是 48 kHz 单声道。"""
    if len(samples) - phase < SAMPLES_PER_BIT * 40:
        return ""
    count = (len(samples) - phase) // SAMPLES_PER_BIT
    data = samples[phase:phase + count * SAMPLES_PER_BIT]
    windows = data.reshape(count, SAMPLES_PER_BIT).astype(np.float32, copy=False)

    p1 = windows @ Q_1200
    p0 = windows @ Q_1800
    e1 = np.sum(p1 * p1, axis=1)
    e0 = np.sum(p0 * p0, axis=1)
    bits = (e1 > e0).astype(np.uint8)
    return bits_to_string(bits)


def iter_sync_positions(bit_string: str, sync: str, allow_one_error: bool = False):
    """先找完全同步；可选回退到最多 1 bit 错误的同步候选。"""
    start = 0
    found = False
    while True:
        pos = bit_string.find(sync, start)
        if pos < 0:
            break
        found = True
        yield pos, 0
        start = pos + 1
    if found or not allow_one_error:
        return

    target = int(sync, 2)
    width = len(sync)
    limit = len(bit_string) - width + 1
    for pos in range(max(0, limit)):
        piece = bit_string[pos:pos + width]
        if len(piece) != width:
            break
        distance = (int(piece, 2) ^ target).bit_count()
        if distance == 1:
            yield pos, 1


def decode_bch_words(bit_string: str) -> tuple[bytes, int] | None:
    """解码 23 个 26-bit 码字，返回 46 字节信息区与修正位数。"""
    expected = 23 * 26
    if len(bit_string) < expected:
        return None

    out_bits: list[str] = []
    corrected_total = 0
    for offset in range(0, expected, 26):
        word_bits = bit_string[offset:offset + 26]
        word = int(word_bits, 2)
        syndrome = polynomial_remainder(word)
        if syndrome:
            mask = BCH_ERROR_TABLE.get(syndrome)
            if mask is None or mask == -1:
                return None
            word ^= mask
            corrected_total += mask.bit_count()
            if polynomial_remainder(word) != 0:
                return None
        out_bits.append(f"{(word >> 10) & 0xFFFF:016b}")

    all_bits = "".join(out_bits)
    data = bytes(int(all_bits[i:i + 8], 2) for i in range(0, len(all_bits), 8))
    if len(data) != 46:
        return None
    return data, corrected_total


@dataclass(frozen=True)
class DecodedFrame:
    sample_position: int
    payload: bytes
    crc_value: int
    corrected_bits: int
    sync_errors: int
    info: bytes


def try_frame(bit_string: str, sync_pos: int, sample_position: int, sync_errors: int) -> DecodedFrame | None:
    data_start = sync_pos + len(SYNC)
    bch = decode_bch_words(bit_string[data_start:data_start + 23 * 26])
    if bch is None:
        return None
    info, corrected_bits = bch
    if info[0:3] != b"\x0c\x2b\x00":
        return None
    if info[45] != 0:
        return None

    payload = info[3:43]
    expected_crc = int.from_bytes(info[43:45], "big")
    actual_crc = crc16_xmodem(info[:43])
    if actual_crc != expected_crc:
        return None
    return DecodedFrame(
        sample_position=sample_position,
        payload=payload,
        crc_value=expected_crc,
        corrected_bits=corrected_bits,
        sync_errors=sync_errors,
        info=info,
    )


def scan_frames(samples: np.ndarray, absolute_sample_start: int = 0, allow_one_sync_error: bool = False) -> list[DecodedFrame]:
    """扫描一段 48 kHz mono 音频；只返回 BCH、帧头、补齐位与 CRC 均匹配的帧。"""
    x = np.asarray(samples, dtype=np.int16)
    if len(x) < int(TARGET_RATE * (FRAME_SECONDS + 0.12)):
        return []
    frames: list[DecodedFrame] = []
    seen_positions: list[int] = []

    for phase in range(SAMPLES_PER_BIT):
        original = demodulate_phase(x, phase)
        if not original:
            continue
        inverted = "".join("1" if c == "0" else "0" for c in original)
        candidates: list[tuple[str, int, str]] = []
        candidates.extend((pos, err, original) for pos, err in iter_sync_positions(original, SYNC, allow_one_sync_error))
        candidates.extend((pos, err, inverted) for pos, err in iter_sync_positions(inverted, SYNC, allow_one_sync_error))

        for sync_pos, sync_errors, bit_source in candidates:
            abs_pos = absolute_sample_start + phase + sync_pos * SAMPLES_PER_BIT
            frame = try_frame(bit_source, sync_pos, abs_pos, sync_errors)
            if frame is not None:
                frames.append(frame)

    # 同一物理帧可能在相邻采样相位重复命中，按其原始采样位置再去重。
    frames.sort(key=lambda f: f.sample_position)
    deduped: list[DecodedFrame] = []
    for frame in frames:
        if deduped and abs(frame.sample_position - deduped[-1].sample_position) <= 100:
            old = deduped[-1]
            if (frame.sync_errors, frame.corrected_bits) < (old.sync_errors, old.corrected_bits):
                deduped[-1] = frame
        else:
            deduped.append(frame)
    return deduped


def decode_payload_fields(payload: bytes) -> dict[str, str]:
    if len(payload) != 40:
        return {"载荷长度": f"错误：{len(payload)} 字节"}

    prefix_byte = payload[0x0E]
    prefix = chr(prefix_byte) if 0x20 <= prefix_byte <= 0x7E else f"\\x{prefix_byte:02X}"
    if prefix == " ":
        prefix_display = "（无字母前缀）"
    else:
        prefix_display = prefix
    train_digits = int.from_bytes(payload[0x0F:0x12], "little", signed=False)
    train_no = f"{prefix}{train_digits}" if prefix != " " else str(train_digits)

    model_code = int.from_bytes(payload[0x12:0x14], "little")
    models = {237: "HXD1C（按现有型号对照）", 242: "HXD3D（CRRD 样本已知）"}
    model_name = models.get(model_code, "型号名未知")
    loco_no = int.from_bytes(payload[0x14:0x16], "little")

    mileage_raw = payload[0x16:0x19]
    mileage_raw24 = int.from_bytes(mileage_raw, "little")
    mileage_flags = mileage_raw24 >> 22
    # 样本 33 42 42 与 0A 42 02 相差 0x400029；清除最高两位后只差 41 m。
    # 2026-10-11 两个录音的 6A 37 03 与 6D 37 43 清除最高两位后也仅差 3 m。
    # 因此暂将最高两位作为旗标候选、低 22 位作为米数；旗标具体含义未知。
    mileage_m = mileage_raw24 & 0x3FFFFF
    low22_text = f"K{mileage_m // 1000}+{mileage_m % 1000:03d}"
    if mileage_raw24 == 9_999_999:
        mileage_text = (
            "原始 24 位数值恰为 9,999,999，疑似无效/未知里程哨兵；"
            f"若只取低 22 位则为 {low22_text}，但不作为有效公里标。"
        )
    else:
        mileage_text = f"{low22_text}（低 22 位按米换算候选）"
        if mileage_flags:
            mileage_text += f"；最高两位旗标候选={mileage_flags:02b}（语义未知）"
        else:
            mileage_text += "；最高两位当前为 00"

    speed = payload[0x19]
    other_1a_1b = int.from_bytes(payload[0x1A:0x1C], "little")
    weight = int.from_bytes(payload[0x1C:0x1E], "little")
    length_tenths = int.from_bytes(payload[0x1E:0x20], "little")
    cars = payload[0x20]
    station = payload[0x21]
    section = payload[0x23]
    section_sub = payload[0x24]
    driver = int.from_bytes(payload[0x25:0x28], "little")

    return {
        "车次": f"{train_no}；前缀字节 0x{prefix_byte:02X} {prefix_display}；车次数字原始字节 {payload[0x0F:0x12].hex(' ').upper()} = {train_digits}",
        "机车型号/机车号": f"{model_name}，型号代码 {model_code}（原始 {payload[0x12:0x14].hex(' ').upper()}）；机车号数值 {loco_no}（原始 {payload[0x14:0x16].hex(' ').upper()}，显示补零规则未强行确定）",
        "速度": f"{speed} km/h（载荷偏移 0x19，原始 {payload[0x19]:02X}）",
        "公里标候选": f"{mileage_text}；原始字节 {mileage_raw.hex(' ').upper()}，全 24 位值 {mileage_raw24}，低 22 位米数候选 {mileage_m}，高 2 位值 {mileage_flags:02b}",
        "站数/计数字节候选": f"0x08={payload[0x08]}，0x09={payload[0x09]}；0x0A 状态候选=0x{payload[0x0A]:02X}。当前样本中前两字节常相同，暂不能判定各自语义。",
        "信号机编号候选": f"0x1A–0x1B 原始 {payload[0x1A:0x1C].hex(' ').upper()}，小端数值 {other_1a_1b}；仅候选，尚无现场编号对照。",
        "总重/计长/辆数": f"总重 {weight} t（{payload[0x1C:0x1E].hex(' ').upper()}）；计长 {length_tenths / 10:.1f}（{payload[0x1E:0x20].hex(' ').upper()}）；辆数 {cars}（{payload[0x20]:02X}）",
        "车站/区段/司机号候选": f"车站号候选 {station}；区段值候选 {section}/{section_sub}；司机号候选 {driver}",
    }


def frame_to_text(frame: DecodedFrame, sample_rate: int = TARGET_RATE, source_label: str = "实时") -> str:
    relative = frame.sample_position / sample_rate
    wall = datetime.now().strftime("%H:%M:%S")
    fields = decode_payload_fields(frame.payload)
    lines = [
        "=" * 72,
        f"[{wall}] {source_label} 检出有效帧；音频位置约 {relative:.3f} s",
        f"CRC-16/XMODEM：0x{frame.crc_value:04X}（通过）；BCH 纠正位数：{frame.corrected_bits}；同步字误码：{frame.sync_errors}",
        f"原始 40 字节 TAX 载荷：{frame.payload.hex(' ').upper()}",
    ]
    lines.extend(f"{key}：{value}" for key, value in fields.items())
    lines.append("注意：公里标、站数、信号机编号等字段按当前证据等级显示候选，不因 CRC 通过而自动视为业务语义已确认。")
    return "\n".join(lines)


def sox_file_to_mono48(sox_path: str, wav_path: str) -> np.ndarray:
    command = [
        sox_path, "-V0", wav_path,
        "-t", "raw", "-r", str(TARGET_RATE), "-e", "signed-integer", "-b", "16", "-c", "1", "-L", "-",
        "rate", "-h", str(TARGET_RATE), "channels", "1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError("SoX 转换 WAV 失败：\n" + result.stderr.decode(errors="replace")[-2000:])
    if len(result.stdout) < 2:
        raise RuntimeError("SoX 没有输出 PCM 数据。请检查 SoX 路径与 WAV 文件。")
    # 取整到完整 int16 样本
    usable = len(result.stdout) - (len(result.stdout) % 2)
    return np.frombuffer(result.stdout[:usable], dtype="<i2").copy()


class LiveSession:
    """音频输入 -> SoX(stdin/stdout) -> 解码线程。"""

    def __init__(self, app: "FfskGui", device_index: int, channels: int, input_rate: int, sox_path: str):
        self.app = app
        self.device_index = device_index
        self.channels = channels
        self.input_rate = input_rate
        self.sox_path = sox_path
        self.input_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=100)
        self.output_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=100)
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.process: subprocess.Popen | None = None
        self.stream = None
        self.writer: threading.Thread | None = None
        self.reader: threading.Thread | None = None
        self.decoder: threading.Thread | None = None
        self.dropped_audio_blocks = 0
        self.total_output_samples = 0
        self.stop_requested = threading.Event()

    def start(self):
        command = [
            self.sox_path, "-V0",
            "-t", "raw", "-r", str(self.input_rate), "-e", "signed-integer", "-b", "16", "-c", str(self.channels), "-L", "-",
            "-t", "raw", "-r", str(TARGET_RATE), "-e", "signed-integer", "-b", "16", "-c", "1", "-L", "-",
            "rate", "-h", str(TARGET_RATE), "channels", "1",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.writer = threading.Thread(target=self._writer_loop, name="sox-writer", daemon=True)
        self.reader = threading.Thread(target=self._reader_loop, name="sox-reader", daemon=True)
        self.decoder = threading.Thread(target=self._decoder_loop, name="ffsk-decoder", daemon=True)
        self.writer.start()
        self.reader.start()
        self.decoder.start()

        def callback(indata, frames, time_info, status):
            if status:
                self.events.put(("status", f"PortAudio 状态：{status}"))
            block = indata.copy().tobytes()
            try:
                self.input_queue.put_nowait(block)
            except queue.Full:
                self.dropped_audio_blocks += 1

        self.stream = sd.InputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.input_rate,
            dtype="int16",
            blocksize=2048,
            callback=callback,
        )
        self.stream.start()
        self.events.put(("status", f"正在接收：输入 {self.input_rate} Hz / {self.channels} 声道；SoX 输出 {TARGET_RATE} Hz 单声道。"))

    def _writer_loop(self):
        assert self.process is not None and self.process.stdin is not None
        try:
            while True:
                block = self.input_queue.get()
                if block is None:
                    break
                self.process.stdin.write(block)
        except (BrokenPipeError, OSError) as exc:
            self.events.put(("error", f"写入 SoX 输入管道失败：{exc}"))
        finally:
            try:
                self.process.stdin.close()
            except Exception:
                pass

    def _reader_loop(self):
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                chunk = self.process.stdout.read(8192)
                if not chunk:
                    break
                self.output_queue.put(chunk)
        except Exception as exc:
            self.events.put(("error", f"读取 SoX 输出失败：{exc}"))
        finally:
            self.output_queue.put(None)

    def _decoder_loop(self):
        ring = np.empty(0, dtype=np.int16)
        total = 0
        since_scan = 0
        seen_positions: deque[int] = deque()
        last_rate_notice = 0.0
        while True:
            try:
                chunk = self.output_queue.get(timeout=0.25)
            except queue.Empty:
                if self.stop_requested.is_set():
                    continue
                continue
            if chunk is None:
                break
            usable = len(chunk) - (len(chunk) % 2)
            if usable <= 0:
                continue
            values = np.frombuffer(chunk[:usable], dtype="<i2")
            total += len(values)
            ring = np.concatenate((ring, values))
            max_samples = int(TARGET_RATE * RING_SECONDS)
            if len(ring) > max_samples:
                ring = ring[-max_samples:]
            since_scan += len(values)
            if since_scan >= SCAN_EVERY_SAMPLES and len(ring) >= int(TARGET_RATE * 0.7):
                since_scan %= SCAN_EVERY_SAMPLES
                abs_start = total - len(ring)
                frames = scan_frames(ring, absolute_sample_start=abs_start, allow_one_sync_error=False)
                if not frames:
                    frames = scan_frames(ring, absolute_sample_start=abs_start, allow_one_sync_error=True)
                for frame in frames:
                    while seen_positions and seen_positions[0] < total - TARGET_RATE * 8:
                        seen_positions.popleft()
                    if any(abs(frame.sample_position - old) <= 120 for old in seen_positions):
                        continue
                    seen_positions.append(frame.sample_position)
                    self.events.put(("frame", frame_to_text(frame, source_label="实时输入")))
                now = time.monotonic()
                if self.dropped_audio_blocks and now - last_rate_notice > 5:
                    self.events.put(("status", f"警告：音频输入队列曾满，累计丢弃 {self.dropped_audio_blocks} 个采样块。请降低其他 CPU 负载或增大缓冲。"))
                    last_rate_notice = now
        self.events.put(("done", "SoX / 解码线程已结束。"))

    def stop(self):
        self.stop_requested.set()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        # 用后台线程投递结束标记，避免 GUI 在管道拥塞时等待。
        def finish_input():
            while True:
                try:
                    self.input_queue.put(None, timeout=0.5)
                    break
                except queue.Full:
                    if self.writer and not self.writer.is_alive():
                        break
            if self.process is not None:
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
        threading.Thread(target=finish_input, name="sox-stop", daemon=True).start()


class FfskGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("450 MHz FFSK / TAX 实时解码器")
        self.root.geometry("1100x780")
        self.root.minsize(850, 620)
        self.session: LiveSession | None = None
        self.devices: list[dict] = []
        self.device_map: dict[str, tuple[int, int, int]] = {}
        self.gui_events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.closing = False
        self.sox_var = tk.StringVar(value=shutil.which("sox") or shutil.which("sox.exe") or "sox.exe")
        self.device_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪。选择虚拟音频输入设备和 sox.exe 后开始。")
        self._build_ui()
        self.refresh_devices()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="输入设备（请选择 VAC / CABLE / Line 1 等虚拟音频设备）").grid(row=0, column=0, sticky="w")
        self.device_combo = ttk.Combobox(outer, textvariable=self.device_var, state="readonly", width=76)
        self.device_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 7))
        ttk.Button(outer, text="刷新设备", command=self.refresh_devices).grid(row=1, column=2, padx=(6, 0), sticky="ew")

        ttk.Label(outer, text="SoX 可执行文件路径").grid(row=2, column=0, sticky="w")
        sox_frame = ttk.Frame(outer)
        sox_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(3, 7))
        self.sox_entry = ttk.Entry(sox_frame, textvariable=self.sox_var)
        self.sox_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(sox_frame, text="浏览…", command=self.browse_sox).pack(side="left", padx=(6, 0))

        buttons = ttk.Frame(outer)
        buttons.grid(row=4, column=0, columnspan=3, sticky="w", pady=(1, 8))
        self.start_button = ttk.Button(buttons, text="开始实时解码", command=self.start_live)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="停止", command=self.stop_live, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="选择 WAV 离线分析…", command=self.analyze_wav).pack(side="left", padx=(18, 0))
        ttk.Button(buttons, text="清空记录", command=self.clear_log).pack(side="left", padx=(6, 0))

        status_frame = ttk.Frame(outer)
        status_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 7))
        ttk.Label(status_frame, text="状态：").pack(side="left")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left", fill="x", expand=True)

        ttk.Label(outer, text="有效帧与字段候选（只有 BCH / CRC 校验通过才列出）").grid(row=6, column=0, columnspan=3, sticky="w")
        text_frame = ttk.Frame(outer)
        text_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        self.log = tk.Text(text_frame, wrap="word", font=("Consolas", 10), undo=False)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.columnconfigure(2, weight=0)
        outer.rowconfigure(7, weight=1)
        self.log.insert("end", "使用提示：\n")
        self.log.insert("end", "1. 先启动你播放/接收音频的软件，并把其输出路由到虚拟音频线。\n")
        self.log.insert("end", "2. 在输入设备列表选择虚拟线的录音端（例如 CABLE Output / Line 1），不是扬声器输出端。\n")
        self.log.insert("end", "3. SoX 会把输入 PCM 实时重采样/下混到 48000 Hz、单声道；程序按 1200 baud 解析。\n")
        self.log.insert("end", "4. 公里标目前采用 CRRD 已知样本支持的 24 位小端米数换算；异常大值会提示核查。站数和信号机编号只显示候选。\n\n")
        self.log.configure(state="normal")

    def refresh_devices(self):
        if sd is None:
            messagebox.showerror("缺少依赖", "未安装 sounddevice。请运行：\npy -m pip install numpy sounddevice")
            return
        try:
            all_devices = sd.query_devices()
            self.devices = []
            self.device_map.clear()
            values = []
            for idx, dev in enumerate(all_devices):
                in_ch = int(dev.get("max_input_channels", 0))
                if in_ch <= 0:
                    continue
                rate = int(round(float(dev.get("default_samplerate", TARGET_RATE))))
                channels = min(in_ch, 2)
                label = f"{idx}: {dev['name']}  | 输入通道 {channels}/{in_ch}  | 默认 {rate} Hz"
                self.devices.append(dict(index=idx, name=dev["name"], rate=rate, channels=channels, label=label))
                self.device_map[label] = (idx, channels, rate)
                values.append(label)
            self.device_combo["values"] = values
            if values:
                if self.device_var.get() not in self.device_map:
                    preferred = next((x for x in values if any(k in x.lower() for k in ("cable", "virtual", "line 1", "vac"))), values[0])
                    self.device_var.set(preferred)
                self.status_var.set(f"发现 {len(values)} 个可用输入设备。请确认选择虚拟音频线的录音端。")
            else:
                self.status_var.set("没有发现输入设备；检查声卡驱动或虚拟音频线是否启用。")
        except Exception as exc:
            messagebox.showerror("读取设备失败", str(exc))

    def browse_sox(self):
        path = filedialog.askopenfilename(
            title="选择 SoX 可执行文件",
            filetypes=[("SoX executable", "sox.exe"), ("Executable", "*.exe"), ("All files", "*.*")],
        )
        if path:
            self.sox_var.set(path)

    def _validate_sox(self) -> str:
        path = self.sox_var.get().strip().strip('"')
        if not path:
            raise RuntimeError("请指定 sox.exe 的完整路径，或将 SoX 加入 PATH。")
        resolved = shutil.which(path) or path
        if not (os.path.isfile(resolved) or shutil.which(resolved)):
            raise RuntimeError(f"找不到 SoX：{resolved}")
        try:
            check = subprocess.run([resolved, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5, check=False)
        except Exception as exc:
            raise RuntimeError(f"无法运行 SoX：{exc}") from exc
        if check.returncode != 0:
            # SoX 某些编译版本的 version 命令返回码可能不同，只要有可执行输出就继续。
            if not check.stdout:
                raise RuntimeError("SoX 启动失败，请检查路径。")
        return resolved

    def start_live(self):
        if sd is None:
            messagebox.showerror("缺少依赖", "请运行：py -m pip install numpy sounddevice")
            return
        if self.session is not None:
            return
        device = self.device_map.get(self.device_var.get())
        if device is None:
            messagebox.showerror("未选择设备", "请先选择一个输入设备。")
            return
        try:
            sox = self._validate_sox()
            idx, channels, rate = device
            self.session = LiveSession(self, idx, channels, rate, sox)
            self.session.start()
        except Exception as exc:
            if self.session is not None:
                try:
                    self.session.stop()
                except Exception:
                    pass
                self.session = None
            messagebox.showerror("启动失败", str(exc))
            return
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("实时监听中……等待检测同步字、BCH 和 CRC 均通过的帧。")
        self._log(f"开始实时监听。源设备索引={idx}，输入采样率={rate} Hz，SoX 输出 48000 Hz mono。")

    def stop_live(self):
        if self.session is None:
            return
        session = self.session
        session.stop()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status_var.set("正在停止 SoX 管道并排空剩余音频……")
        self._log("已请求停止监听；正在处理停止前已经进入管道的数据。")

    def analyze_wav(self):
        path = filedialog.askopenfilename(
            title="选择要离线解析的 WAV",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            sox = self._validate_sox()
        except Exception as exc:
            messagebox.showerror("SoX 配置错误", str(exc))
            return
        self.status_var.set(f"正在用 SoX 分析：{os.path.basename(path)}……")
        self._log(f"开始离线 WAV：{path}")
        threading.Thread(target=self._analyze_file_worker, args=(sox, path), daemon=True).start()

    def _analyze_file_worker(self, sox: str, path: str):
        try:
            samples = sox_file_to_mono48(sox, path)
            if len(samples) < TARGET_RATE:
                self.gui_events.put(("status", "转换后音频不足 1 秒，可能无法包含完整帧。"))
            self.gui_events.put(("status", f"SoX 转换完成：{len(samples)/TARGET_RATE:.2f} 秒，正在解析。"))
            # 用 2.2 秒窗口、0.75 秒重叠，避免长录音一次扫描耗时过长或截断完整帧。
            window = int(TARGET_RATE * RING_SECONDS)
            step = int(TARGET_RATE * 1.35)
            positions: list[int] = []
            found: list[DecodedFrame] = []
            if len(samples) <= window:
                chunks = [(0, samples)]
            else:
                chunks = []
                start = 0
                while start < len(samples):
                    chunks.append((start, samples[start:start + window]))
                    if start + window >= len(samples):
                        break
                    start += step
            for base, chunk in chunks:
                if len(chunk) < int(TARGET_RATE * (FRAME_SECONDS + 0.12)):
                    continue
                hits = scan_frames(chunk, absolute_sample_start=base, allow_one_sync_error=False)
                if not hits:
                    hits = scan_frames(chunk, absolute_sample_start=base, allow_one_sync_error=True)
                for frame in hits:
                    if any(abs(frame.sample_position - p) <= 120 for p in positions):
                        continue
                    positions.append(frame.sample_position)
                    found.append(frame)
            found.sort(key=lambda f: f.sample_position)
            if not found:
                self.gui_events.put(("log", f"[{os.path.basename(path)}] 未找到同时通过 BCH 和 CRC 的帧。检查音频源、音量、FFSK 音调或 SoX 路径。"))
            else:
                for frame in found:
                    self.gui_events.put(("log", frame_to_text(frame, source_label=os.path.basename(path))))
                self.gui_events.put(("log", f"离线分析完成：共找到 {len(found)} 份 BCH/CRC 均通过的独立帧。重复采样相位命中已去重。"))
            self.gui_events.put(("status", f"离线分析完成：{os.path.basename(path)}；有效帧 {len(found)} 份。"))
        except Exception as exc:
            self.gui_events.put(("error", f"离线分析失败：{exc}"))

    def clear_log(self):
        self.log.delete("1.0", "end")

    def _log(self, text: str):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _poll_events(self):
        # 实时线程事件与离线线程事件均通过 GUI 主线程更新 Tk widgets。
        if self.session is not None:
            while True:
                try:
                    kind, msg = self.session.events.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(kind, msg)
        while True:
            try:
                kind, msg = self.gui_events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(kind, msg)
        self.root.after(100, self._poll_events)

    def _handle_event(self, kind: str, msg: str):
        if kind == "frame" or kind == "log":
            self._log(msg)
        elif kind == "status":
            self.status_var.set(msg)
        elif kind == "error":
            self.status_var.set(msg)
            self._log("错误：" + msg)
        elif kind == "done":
            self.status_var.set(msg)
            # 等待读取线程把停止前已进入 SoX 的音频处理完，再允许重新启动。
            if self.session is not None:
                session = self.session
                self.session = None
                if session.stream is not None:
                    try:
                        session.stream.stop()
                        session.stream.close()
                    except Exception:
                        pass
                    session.stream = None
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")

    def close(self):
        if self.session is not None:
            self.closing = True
            self.stop_live()
            self.root.after(100, self._wait_for_close)
        else:
            self.root.destroy()

    def _wait_for_close(self):
        if self.session is None:
            self.root.destroy()
        else:
            self.root.after(100, self._wait_for_close)


def main():
    root = tk.Tk()
    app = FfskGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()