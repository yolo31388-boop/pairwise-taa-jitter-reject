"""TAA 抖动与历史拒绝（temporal anti-aliasing jitter & history rejection）。

对应六处修复：

1. ``jitter_by_time``      —— 抖动按时间戳连续取 Halton 点，掉帧不跳号；
2. ``motion_current_matrix`` —— 运动矢量用当帧视图投影矩阵重投影；
3. ``adaptive_blend``      —— 新帧/历史混合权重按像素速度自适应；
4. ``depth_weight_clip``   —— 邻域裁剪按深度差加权，保护细几何边缘；
5. ``clear_on_switch``     —— 场景切换 / 分辨率变化 / 机位硬切即清历史；
6. ``clamp_output``        —— 输出前钳制颜色范围并吞掉 NaN/Inf。

其余函数是把上述步骤串起来的管线（``reproject`` / ``blend`` /
``resolve``）以及质量与健壮性工具（``validate_history`` / ``metrics`` /
``audit`` / ``fallback`` / ``determinism``）。

颜色约定为线性、每通道范围 [0, 1]；运动矢量以“像素/帧”为单位，矩阵路径
内部会把归一化 uv 位移换算成像素。
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = [
    "JitterSequence",
    "halton",
    "jitter_by_time",
    "motion_current_matrix",
    "adaptive_blend",
    "depth_weight_clip",
    "clear_on_switch",
    "clamp_output",
    "jitter",
    "reproject",
    "blend",
    "resolve",
    "validate_history",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _as_float(value: Any) -> np.ndarray:
    """转成浮点 ndarray；None 变成 0 维零数组，永远不抛异常。"""
    if value is None:
        return np.zeros((), dtype=float)
    return np.asarray(value, dtype=float)


def _finite_like(value: Any, fill: float = 0.0) -> np.ndarray:
    """转浮点并把 NaN/Inf 替换成 fill。"""
    arr = _as_float(value)
    return np.where(np.isfinite(arr), arr, fill)


# ---------------------------------------------------------------------------
# 1. 时间戳驱动的连续 Halton 抖动
# ---------------------------------------------------------------------------
def halton(index: int, base: int) -> float:
    """标准基数反 radix Halton 序列，``index`` 为 1 基序号。"""
    if base < 2:
        raise ValueError("halton base must be >= 2")
    fraction = 1.0
    result = 0.0
    i = int(index)
    while i > 0:
        fraction /= base
        result += fraction * (i % base)
        i //= base
    return result


class JitterSequence:
    """按呈现时间戳推进的 Halton 抖动序列。

    每个“新时间戳”（即每一帧真正被呈现出来）只消费一个 Halton 点，序号
    严格 +1；帧序号因掉帧出现空洞时 Halton 索引仍然连续，因此不会出现
    固定条纹。同一时间戳重复求值返回缓存点（同一帧可被多次访问而结果
    确定）；时间戳回绕（时间基准重置）时序列自动归零。
    """

    def __init__(self, frame_period: float = 1.0 / 30.0) -> None:
        self.frame_period = float(frame_period)
        self._index = 0
        self._last_timestamp: Optional[float] = None
        self._offset = np.zeros(2, dtype=float)

    def reset(self) -> None:
        self._index = 0
        self._last_timestamp = None
        self._offset = np.zeros(2, dtype=float)

    @property
    def index(self) -> int:
        return self._index

    def sample(self, timestamp: float, frame_period: Optional[float] = None) -> np.ndarray:
        ts = float(timestamp)
        if frame_period is not None:
            self.frame_period = float(frame_period)

        # 首次调用或时间戳回绕：从头开始一条干净的序列。
        if self._last_timestamp is None or ts < self._last_timestamp:
            self._index = 1
        elif ts > self._last_timestamp:
            # 无论中间掉了多少帧，每个呈现帧只前进一个 Halton 点。
            self._index += 1
        # ts == self._last_timestamp：同一帧重复查询，返回缓存。

        self._last_timestamp = ts
        self._offset = np.array(
            [halton(self._index, 2) - 0.5, halton(self._index, 3) - 0.5],
            dtype=float,
        )
        return self._offset.copy()


_DEFAULT_JITTER = JitterSequence()


def jitter_by_time(
    timestamp: float,
    frame_period: Optional[float] = None,
    reset: bool = False,
    sequence: Optional[JitterSequence] = None,
) -> np.ndarray:
    """按时间戳取连续 Halton 抖动，返回 [-0.5, 0.5] 的 (x, y) 子像素偏移。"""
    seq = sequence if sequence is not None else _DEFAULT_JITTER
    if reset:
        seq.reset()
    return seq.sample(timestamp, frame_period)


def jitter(timestamp: Optional[float] = None, frame_index: Optional[int] = None) -> np.ndarray:
    """抖动入口；位置参数按时间戳处理，保证与 ``jitter_by_time`` 同一序列。"""
    if timestamp is None:
        timestamp = float(frame_index) if frame_index is not None else 0.0
    return jitter_by_time(timestamp)


# ---------------------------------------------------------------------------
# 2. 当帧矩阵驱动的运动矢量 / 重投影
# ---------------------------------------------------------------------------
def _project_to_uv(view_proj: np.ndarray, world_position: np.ndarray):
    """世界坐标 -> 裁剪空间 -> NDC -> [0,1] uv，同时返回裁剪 w 与有效掩码。"""
    homo = np.concatenate(
        [world_position, np.ones(world_position.shape[:-1] + (1,), dtype=float)],
        axis=-1,
    )
    clip = homo @ np.asarray(view_proj, dtype=float).T
    w = clip[..., 3:4]
    safe_w = np.where(np.abs(w) > _EPS, w, np.where(w >= 0.0, _EPS, -_EPS))
    ndc = clip[..., :3] / safe_w
    uv = ndc[..., :2] * 0.5 + 0.5
    valid = np.squeeze(w > 0.0, axis=-1) & np.isfinite(uv).all(axis=-1)
    return uv, np.squeeze(w, axis=-1), valid


def motion_current_matrix(
    world_position: Any,
    cur_view_proj: Any = None,
    prev_view_proj: Any = None,
) -> dict:
    """用**当帧**视图投影矩阵计算屏幕空间运动矢量。

    旧实现复用上一帧相机矩阵算出的矢量，快速旋转时重投影错位。这里对同一
    世界坐标分别用当帧 / 上一帧 VP 投影，``motion = uv_prev - uv_cur`` 即
    历史缓冲的回采偏移。位于裁剪面背后或投影非有限的像素标记为无效。
    """
    pos = _as_float(world_position)
    if pos.ndim == 0 or pos.shape[-1] != 3 or cur_view_proj is None or prev_view_proj is None:
        return {
            "current_uv": np.zeros(pos.shape[:-1] + (2,), dtype=float),
            "history_uv": np.zeros(pos.shape[:-1] + (2,), dtype=float),
            "motion": np.zeros(pos.shape[:-1] + (2,), dtype=float),
            "valid": np.zeros(pos.shape[:-1], dtype=bool),
        }

    cur_uv, _, cur_valid = _project_to_uv(cur_view_proj, pos)
    prev_uv, _, prev_valid = _project_to_uv(prev_view_proj, pos)
    valid = cur_valid & prev_valid
    motion = np.where(valid[..., None], prev_uv - cur_uv, 0.0)
    return {
        "current_uv": cur_uv,
        "history_uv": np.where(valid[..., None], prev_uv, cur_uv),
        "motion": motion,
        "valid": valid,
    }


def _bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """对 HxWxC 图像按 [0,1] uv 做双线性采样，越界钳到边缘。"""
    height, width, channels = image.shape
    x = np.clip(uv[..., 0], 0.0, 1.0) * (width - 1)
    y = np.clip(uv[..., 1], 0.0, 1.0) * (height - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def reproject(
    world_or_current: Any,
    history: Any = None,
    cur_view_proj: Any = None,
    prev_view_proj: Any = None,
    velocity: Any = None,
) -> dict:
    """用当帧矩阵把历史缓冲重投影到当帧像素。

    优先走世界坐标 + 双矩阵路径；没有矩阵时退化为按给定像素速度平移
    uv。无效区域（相机背后 / 越界）由调用方在混合阶段拒绝。
    """
    pos = _as_float(world_or_current)
    if pos.ndim < 2 or pos.shape[-1] != 3:
        return {
            "color": np.zeros_like(pos, dtype=float),
            "history_uv": np.zeros(pos.shape[:-1] + (2,), dtype=float),
            "motion_px": np.zeros(pos.shape[:-1] + (2,), dtype=float),
            "valid": np.zeros(pos.shape[:-1], dtype=bool),
        }

    height, width = pos.shape[:2]
    if cur_view_proj is not None and prev_view_proj is not None:
        mv = motion_current_matrix(pos, cur_view_proj, prev_view_proj)
        motion_px = mv["motion"] * np.array([width, height], dtype=float)
        history_uv = mv["history_uv"]
        valid = mv["valid"]
    else:
        vel = _as_float(velocity)
        if vel.shape == (height, width, 2):
            motion_px = vel
            cur_uv = np.stack(
                np.meshgrid(
                    np.arange(width, dtype=float) / max(width - 1, 1),
                    np.arange(height, dtype=float) / max(height - 1, 1),
                    indexing="xy",
                ),
                axis=-1,
            )
            history_uv = cur_uv - motion_px / np.array([width, height], dtype=float)
            valid = np.isfinite(history_uv).all(axis=-1)
        else:
            motion_px = np.zeros((height, width, 2), dtype=float)
            history_uv = np.stack(
                np.meshgrid(
                    np.linspace(0.0, 1.0, width),
                    np.linspace(0.0, 1.0, height),
                    indexing="xy",
                ),
                axis=-1,
            )
            valid = np.ones((height, width), dtype=bool)

    hist = _as_float(history)
    if hist.ndim != 3 or hist.shape[:2] != (height, width):
        color = np.zeros((height, width, pos.shape[-1]), dtype=float)
        valid = np.zeros((height, width), dtype=bool)
    else:
        color = _bilinear(hist, history_uv)
        in_bounds = (
            (history_uv[..., 0] >= 0.0)
            & (history_uv[..., 0] <= 1.0)
            & (history_uv[..., 1] >= 0.0)
            & (history_uv[..., 1] <= 1.0)
        )
        valid = valid & in_bounds

    return {
        "color": color,
        "history_uv": history_uv,
        "motion_px": motion_px,
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# 3. 速度自适应混合权重
# ---------------------------------------------------------------------------
def adaptive_blend(
    current: Any,
    history: Any = None,
    velocity: Any = 0.0,
    history_valid: bool = True,
    min_current_weight: float = 0.05,
    max_current_weight: float = 1.0,
    stiffness: float = 0.25,
) -> dict:
    """按像素速度自适应的时域混合。

    ``alpha`` 是当帧权重：静止区域 ``alpha`` 取最小值（强累积，收敛后不再
    缓慢漂移）；速度越大 ``alpha`` 越大，运动物体快速甩掉历史、不拖长尾。
    历史无效（disocclusion / 切换后首帧）时 ``alpha = 1``。
    """
    cur = _finite_like(current)
    if history is None:
        hist = np.zeros_like(cur)
        history_valid = False
    else:
        hist = _finite_like(history)

    vel = _as_float(velocity)
    speed = np.sqrt(np.sum(vel * vel, axis=-1)) if vel.ndim >= 1 and vel.shape[-1] == 2 else np.abs(vel)
    alpha = np.clip(min_current_weight + stiffness * speed, min_current_weight, max_current_weight)
    if not history_valid:
        alpha = np.ones_like(alpha) if np.ndim(alpha) else 1.0

    alpha_b = alpha[..., None] if np.ndim(alpha) else alpha
    color = alpha_b * cur + (1.0 - alpha_b) * hist
    return {
        "color": color,
        "speed": speed,
        "current_weight": alpha,
        "history_weight": (1.0 - alpha) if np.ndim(alpha) else 1.0 - alpha,
    }


# ---------------------------------------------------------------------------
# 4. 深度加权邻域裁剪
# ---------------------------------------------------------------------------
def depth_weight_clip(
    history_color: Any,
    neighborhood_colors: Any = None,
    neighborhood_depths: Any = None,
    center_depth: Any = None,
    sigma: Optional[float] = None,
    k: float = 1.75,
) -> np.ndarray:
    """把历史颜色裁进**深度加权**的邻域颜色包围盒。

    深度差越大的邻域样本权重越小（``w = exp(-|d-d0|/sigma)``），包围盒由
    加权均值与加权标准差给出，并锚定与中心同深度的样本（即当前像素自身，
    保证当前像素恒在盒内）。深度不连续对面的前景/背景颜色因此不会把包围盒
    撑松，细几何边缘不会被历史糊掉；无深度信息时退化为经典颜色 AABB。
    """
    history = _finite_like(history_color)
    if neighborhood_colors is None:
        return history

    colors = _finite_like(neighborhood_colors)
    if colors.ndim == 1:
        colors = colors[None, :]
    depths = _as_float(neighborhood_depths)
    have_depth = (
        depths.shape == (colors.shape[0],)
        and bool(np.all(np.isfinite(depths)))
        and center_depth is not None
    )

    if not have_depth:
        # 没有可用深度：均匀权重 + 实测 min/max，即经典颜色 AABB 裁剪。
        weights = np.full(colors.shape[0], 1.0 / colors.shape[0], dtype=float)
    else:
        if sigma is None:
            sigma = max(float(np.std(depths)), _EPS)
        weights = np.exp(
            -np.abs(depths - float(np.asarray(center_depth))) / max(sigma, _EPS)
        )
    weights = weights / max(float(weights.sum()), _EPS)

    mean = np.sum(colors * weights[:, None], axis=0)
    variance = np.sum(weights[:, None] * (colors - mean) ** 2, axis=0)
    std = np.sqrt(np.maximum(variance, 0.0))

    low = mean - k * std
    high = mean + k * std
    if have_depth:
        # 锚定与中心同深度的样本（即当前像素自身），包围盒只向其外扩。
        anchor = colors[int(np.argmax(weights))]
        low = np.minimum(low, anchor)
        high = np.maximum(high, anchor)
    else:
        low = np.minimum(low, colors.min(axis=0))
        high = np.maximum(high, colors.max(axis=0))
    return np.clip(history, low, high)


def _image_neighborhood_clip(
    current: np.ndarray,
    history: np.ndarray,
    depth: np.ndarray,
    sigma: Optional[float] = None,
    k: float = 1.75,
) -> np.ndarray:
    """整图版本：3x3 邻域、深度加权，逐像素构造包围盒并裁剪历史。"""
    height, width, _ = current.shape
    padded_color = np.pad(current, ((1, 1), (1, 1), (0, 0)), mode="edge")
    padded_depth = np.pad(depth, ((1, 1), (1, 1)), mode="edge")

    nb_colors, nb_depths = [], []
    for dy in range(3):
        for dx in range(3):
            nb_colors.append(padded_color[dy:dy + height, dx:dx + width])
            nb_depths.append(padded_depth[dy:dy + height, dx:dx + width])
    colors = np.stack(nb_colors, axis=-2)       # H x W x 9 x C
    depths = np.stack(nb_depths, axis=-1)       # H x W x 9

    if sigma is None:
        sigma = max(float(np.std(depth)), _EPS)
    weights = np.exp(-np.abs(depths - depth[..., None]) / max(sigma, _EPS))
    weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), _EPS)

    mean = np.sum(colors * weights[..., None], axis=-2)
    variance = np.sum(weights[..., None] * (colors - mean[..., None, :]) ** 2, axis=-2)
    std = np.sqrt(np.maximum(variance, 0.0))
    # 锚定当前像素颜色（中心样本权重最高）：当前像素永远在盒内，盒子只在
    # 同深度方向上扩展，深度不连续对面的颜色不会把边界撑松。
    low = np.minimum(mean - k * std, current)
    high = np.maximum(mean + k * std, current)
    return np.clip(history, low, high)


# ---------------------------------------------------------------------------
# 5. 场景切换 / 分辨率变化即清历史
# ---------------------------------------------------------------------------
def clear_on_switch(
    history: Any,
    current_scene: Any = None,
    previous_scene: Any = None,
    current_resolution: Any = None,
    previous_resolution: Any = None,
    camera_cut: bool = False,
    fill: float = 0.0,
) -> dict:
    """检测场景切换 / 分辨率变化 / 机位硬切，命中即返回清零后的历史。

    任一条件成立都会清空历史并重置默认抖动序列，保证新场景第一帧不会叠
    旧场景残影。返回值带 ``cleared`` 与 ``reasons`` 便于审计。
    """
    reasons: list[str] = []
    if camera_cut:
        reasons.append("camera_cut")
    if current_scene is not None and previous_scene is not None and current_scene != previous_scene:
        reasons.append("scene_changed")

    hist = _as_float(history)
    if current_resolution is not None and previous_resolution is not None:
        if tuple(current_resolution) != tuple(previous_resolution):
            reasons.append("resolution_changed")
    elif hist.ndim >= 2 and current_resolution is not None:
        if tuple(hist.shape[:2]) != tuple(current_resolution):
            reasons.append("resolution_changed")

    cleared = bool(reasons)
    if cleared:
        if current_resolution is not None and hist.ndim == 3:
            new_history = np.full(
                tuple(current_resolution) + (hist.shape[2],), fill, dtype=float
            )
        elif hist.ndim >= 1:
            new_history = np.full_like(hist, fill)
        else:
            new_history = np.array(fill, dtype=float)
        _DEFAULT_JITTER.reset()
    else:
        new_history = hist

    return {
        "history": new_history,
        "cleared": cleared,
        "valid": not cleared,
        "reasons": reasons,
        "jitter_reset": cleared,
    }


def validate_history(
    history: Any,
    expected_shape: Optional[Sequence[int]] = None,
    scene_id: Any = None,
    current_scene_id: Any = None,
) -> dict:
    """历史缓冲体检：形状、有限性、场景归属，任一不过即判无效。"""
    issues: list[str] = []
    hist = _as_float(history)

    if hist.ndim == 0:
        return {"valid": True, "issues": issues, "finite_ratio": 1.0 if np.isfinite(hist) else 0.0}

    if expected_shape is not None and tuple(hist.shape) != tuple(expected_shape):
        issues.append("shape_mismatch")
    finite = np.isfinite(hist)
    finite_ratio = float(finite.sum()) / max(float(finite.size), 1)
    if finite_ratio < 1.0:
        issues.append("non_finite_values")
    if scene_id is not None and current_scene_id is not None and scene_id != current_scene_id:
        issues.append("scene_mismatch")
    return {"valid": not issues, "issues": issues, "finite_ratio": finite_ratio}


# ---------------------------------------------------------------------------
# 6. 输出钳制
# ---------------------------------------------------------------------------
def clamp_output(color: Any, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """输出前钳制到 [low, high]，NaN/Inf 先替换成 low。"""
    safe = _finite_like(color, fill=low)
    return np.clip(safe, low, high)


# ---------------------------------------------------------------------------
# 像素级 / 整帧管线
# ---------------------------------------------------------------------------
def blend(
    current: Any,
    history: Any = None,
    velocity: Any = 0.0,
    neighborhood_colors: Any = None,
    neighborhood_depths: Any = None,
    center_depth: Any = None,
    history_valid: bool = True,
) -> dict:
    """先深度加权邻域裁剪，再按速度自适应混合。"""
    cur = _finite_like(current)
    if history is None:
        hist = np.zeros_like(cur)
        history_valid = False
    else:
        hist = _finite_like(history)

    clipped = depth_weight_clip(
        hist,
        neighborhood_colors=neighborhood_colors,
        neighborhood_depths=neighborhood_depths,
        center_depth=center_depth,
    )
    return adaptive_blend(cur, clipped, velocity=velocity, history_valid=history_valid)


def resolve(
    current: Any,
    history: Any = None,
    depth: Any = None,
    world_position: Any = None,
    cur_view_proj: Any = None,
    prev_view_proj: Any = None,
    velocity: Any = None,
    timestamp: Optional[float] = None,
    scene_id: Any = None,
    previous_scene_id: Any = None,
    camera_cut: bool = False,
) -> dict:
    """单帧 TAA resolve 全流程。

    切换检测 -> 时间戳抖动 -> 当帧矩阵重投影 -> 深度加权邻域裁剪 ->
    速度自适应混合 -> 输出钳制，并回写新历史。
    """
    cur = _finite_like(current)

    # 非图像（标量契约调用 / 退化输入）：只做钳制，保证管线不崩。
    if cur.ndim < 3:
        output = clamp_output(cur)
        return {
            "output": output,
            "history": output,
            "jitter_offset": np.zeros(2, dtype=float),
            "cleared": camera_cut,
            "mean_current_weight": 1.0,
        }

    height, width, channels = cur.shape
    prev_resolution = None
    hist_arr = _as_float(history)
    if hist_arr.ndim == 3:
        prev_resolution = hist_arr.shape[:2]

    switch = clear_on_switch(
        hist_arr if hist_arr.ndim == 3 else np.zeros((height, width, channels)),
        current_scene=scene_id,
        previous_scene=previous_scene_id,
        current_resolution=(height, width),
        previous_resolution=prev_resolution,
        camera_cut=camera_cut,
    )
    if switch["cleared"] or hist_arr.ndim != 3 or hist_arr.shape[:2] != (height, width):
        hist_arr = np.zeros((height, width, channels), dtype=float)

    jitter_offset = jitter_by_time(float(timestamp) if timestamp is not None else 0.0)

    pos = _as_float(world_position)
    repro = reproject(
        pos if pos.ndim == 3 and pos.shape[-1] == 3 else cur,
        history=hist_arr,
        cur_view_proj=cur_view_proj,
        prev_view_proj=prev_view_proj,
        velocity=velocity,
    )
    reprojected = repro["color"]
    valid = repro["valid"]
    motion_px = repro["motion_px"]

    depth_arr = _as_float(depth)
    if depth_arr.shape == (height, width):
        clipped = _image_neighborhood_clip(cur, reprojected, depth_arr)
    else:
        clipped = np.clip(reprojected, cur.min(axis=(0, 1)), cur.max(axis=(0, 1)))

    speed = np.sqrt(np.sum(motion_px * motion_px, axis=-1))
    alpha = np.clip(0.05 + 0.25 * speed, 0.05, 1.0)
    alpha = np.where(valid, alpha, 1.0)[..., None]
    merged = alpha * cur + (1.0 - alpha) * clipped

    output = clamp_output(merged)
    return {
        "output": output,
        "history": output,
        "jitter_offset": jitter_offset,
        "cleared": switch["cleared"],
        "clear_reasons": switch["reasons"],
        "mean_current_weight": float(np.mean(alpha)),
        "reproject_valid_ratio": float(np.mean(valid)),
    }


# ---------------------------------------------------------------------------
# 质量度量 / 审计 / 兜底 / 确定性
# ---------------------------------------------------------------------------
def metrics(output: Any, current: Any = None, velocity: Any = None) -> dict:
    """TAA 结果的轻量质量度量（越小越稳，越界比例必须为 0）。"""
    out = _as_float(output)
    if out.ndim == 0:
        return {
            "out_of_range_ratio": 0.0,
            "non_finite_ratio": 0.0,
            "mean_frame_delta": 0.0,
            "static_drift": 0.0,
            "motion_ghost": 0.0,
        }

    finite_ratio = float(np.isfinite(out).sum()) / max(float(out.size), 1)
    out_of_range = float(np.mean((out < 0.0) | (out > 1.0)))
    result = {
        "out_of_range_ratio": out_of_range,
        "non_finite_ratio": 1.0 - finite_ratio,
        "mean_frame_delta": 0.0,
        "static_drift": 0.0,
        "motion_ghost": 0.0,
    }

    cur = _as_float(current)
    if cur.shape == out.shape:
        delta = np.abs(out - cur)
        result["mean_frame_delta"] = float(np.mean(delta))
        vel = _as_float(velocity)
        if vel.ndim >= 1 and vel.shape[-1] == 2:
            speed = np.sqrt(np.sum(vel * vel, axis=-1))
            static_mask = speed < _EPS
            moving_mask = ~static_mask
            if static_mask.any():
                result["static_drift"] = float(delta[static_mask].mean())
            if moving_mask.any():
                result["motion_ghost"] = float(delta[moving_mask].mean())
    return result


def audit(frame_state: Any) -> dict:
    """按六条修复项审计一帧的 TAA 状态，缺省字段记为跳过（None）。"""
    checks = {
        "jitter_continuous": "jitter_continuous",
        "motion_uses_current_matrix": "motion_uses_current_matrix",
        "adaptive_blend_weight": "adaptive_blend_weight",
        "depth_weighted_clip": "depth_weighted_clip",
        "history_cleared_on_switch": "history_cleared_on_switch",
        "output_clamped": "output_clamped",
    }
    if not isinstance(frame_state, dict):
        return {"status": "no-data", "checks": {key: None for key in checks.values()}}

    results = {}
    for flag, key in checks.items():
        results[key] = bool(frame_state[flag]) if flag in frame_state else None

    if any(value is False for value in results.values()):
        status = "fail"
    elif all(value is True for value in results.values()):
        status = "pass"
    else:
        status = "partial"
    return {"status": status, "checks": results, "notes": frame_state.get("notes", [])}


def fallback(current: Any = None, fill: float = 0.0) -> dict:
    """历史不可信时的兜底：直通当前帧、清空历史，绝不输出脏数据。"""
    output = clamp_output(current if current is not None else fill)
    history = np.zeros_like(output, dtype=float)
    return {
        "output": output,
        "history": history,
        "valid": False,
        "reason": "history_rejected_fallback",
    }


def determinism(fn: Optional[Callable] = None, fargs: tuple = (), n_runs: int = 2, **kwargs) -> dict:
    """重复运行同一输入应得到完全一致的输出。

    不传函数时默认验证抖动序列：两条新序列喂相同时间戳必须逐点相同，且
    同一时间戳重复查询不改变结果。
    """
    if not callable(fn):
        timestamps = [0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0]
        seq_a = JitterSequence()
        seq_b = JitterSequence()
        points_a = [seq_a.sample(ts) for ts in timestamps]
        points_b = [seq_b.sample(ts) for ts in timestamps]
        repeated = seq_a.sample(timestamps[-1])
        max_diff = max(
            float(np.max(np.abs(a - b))) for a, b in zip(points_a, points_b)
        )
        max_diff = max(max_diff, float(np.max(np.abs(points_a[-1] - repeated))))
        return {"ok": max_diff <= _EPS, "max_abs_diff": max_diff, "runs": 2}

    outputs = [fn(*fargs, **kwargs) for _ in range(n_runs)]
    base = np.asarray(outputs[0], dtype=float)
    max_diff = 0.0
    for other in outputs[1:]:
        max_diff = max(max_diff, float(np.max(np.abs(base - np.asarray(other, dtype=float)))))
    return {"ok": max_diff <= _EPS, "max_abs_diff": max_diff, "runs": n_runs}
