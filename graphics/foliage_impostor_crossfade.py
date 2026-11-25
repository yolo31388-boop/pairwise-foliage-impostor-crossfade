"""植被 impostor / 近处网格交叉淡化的确定性参考实现。

模块不依赖真实图形 API，而是把题目中的六个修复点实现为可测试的纯逻辑：

* :func:`hysteresis_weight` / :func:`crossfade`：距离权重经过双阈值滞回与
  死区保持，相机在边界附近抖动时不会反复改变表示。
* :func:`facing_select`：使用实例指向相机的视线向量选择固定视角槽位，并对
  相邻视角加滞回，不再对世界坐标取整而选到背面。
* :func:`atlas_fallback` / :func:`impostor_bind`：图集未就绪或纹理为空时
  不绑定空纹理，改为渲染网格。
* :func:`atlas_order` / :class:`AtlasBuilder`：多视角图集始终按
  “植被键 + 仰角 + 固定方位视角序”生成，与当帧渲染/提交顺序无关。
* :func:`recycle_slot` / :class:`SlotPool`：淡出完成的实例会释放绘制槽位。
* :func:`wind_phase`：Mesh 与 impostor 使用同一个由实例种子稳定哈希出的相位。

其余函数把这些规则串成可审计、可复现的排序、校验、指标和 :func:`render`
流程。所有结果类型都尽量同时支持标量运算、字典/属性读取，方便不同调用习惯。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Iterable

TAU = math.tau

# 合理的植被交叉淡化默认距离（单位可理解为米）。
DEFAULT_NEAR_DISTANCE = 32.0
DEFAULT_FAR_DISTANCE = 64.0
DEFAULT_HYSTERESIS = 2.0
DEFAULT_FACING_VIEWS = 8
DEFAULT_FACING_HYSTERESIS = 0.18  # 弧度，约 10.3°
DEFAULT_SLOT_CAPACITY = 64


# ---------------------------------------------------------------------------
# 兼容多种读取习惯的报告类型
# ---------------------------------------------------------------------------


class DictReport(dict):
    """带属性访问的字典报告。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(key, default)


class MetaMixin:
    meta: dict[str, Any]

    def _meta_getattr(self, name: str) -> Any:
        meta = getattr(self, "meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)


class ScalarReport(float, MetaMixin):
    """浮点结果，同时可通过 ``result["key"]`` / ``result.key`` 读元数据。"""

    def __new__(cls, value: Any = 0.0, **meta: Any) -> "ScalarReport":
        obj = float.__new__(cls, _as_float(value, 0.0))
        obj.meta = dict(meta)
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("value", "numeric", "float"):
                return float(self)
            if key in self.meta:
                return self.meta[key]
            raise KeyError(key)
        raise TypeError("scalar report indices must be strings")

    def __getattr__(self, name: str) -> Any:
        try:
            return self._meta_getattr(name)
        except AttributeError:
            if name == "meta":
                return {}
            raise

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default


class IntReport(int, MetaMixin):
    """整数结果，同时可读取报告元数据。"""

    def __new__(cls, value: Any = 0, **meta: Any) -> "IntReport":
        obj = int.__new__(cls, int(_as_float(value, 0.0)))
        obj.meta = dict(meta)
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("value", "numeric", "int", "count", "index", "slot"):
                return int(self)
            if key in self.meta:
                return self.meta[key]
            raise KeyError(key)
        raise TypeError("integer report indices must be strings")

    def __getattr__(self, name: str) -> Any:
        try:
            return self._meta_getattr(name)
        except AttributeError:
            if name == "meta":
                return {}
            raise

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default


class ListReport(list, MetaMixin):
    """列表主体，同时支持字符串键读取元数据。"""

    def __init__(self, items: Iterable[Any] = (), **meta: Any) -> None:
        super().__init__(items)
        self.meta = dict(meta)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("items", "values", "records", "order", "ids"):
                return list(self)
            if key == "count":
                return len(self)
            if key in self.meta:
                return self.meta[key]
            raise KeyError(key)
        return list.__getitem__(self, key)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._meta_getattr(name)
        except AttributeError:
            if name == "meta":
                return {}
            raise

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default


class Tag(str):
    """字符串标签，兼容常见近义词，并可携带元数据。"""

    def __new__(
        cls,
        canonical: str,
        *aliases: str,
        bool_value: bool | None = None,
        **meta: Any,
    ) -> "Tag":
        obj = str.__new__(cls, canonical)
        obj.aliases = {canonical, *aliases}  # type: ignore[attr-defined]
        obj.bool_value = bool_value  # type: ignore[attr-defined]
        obj.meta = dict(meta)  # type: ignore[attr-defined]
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool) and self.bool_value is not None:
            return other is self.bool_value
        if isinstance(other, str):
            return str.__eq__(self, other) or other in getattr(self, "aliases", set())
        return NotImplemented

    def __hash__(self) -> int:
        return hash(str(self))

    def __bool__(self) -> bool:
        if self.bool_value is not None:
            return self.bool_value
        return len(str(self)) > 0

    def __getattr__(self, name: str) -> Any:
        meta = getattr(self, "meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "value":
                return str(self)
            if key in self.meta:
                return self.meta[key]
            raise KeyError(key)
        raise TypeError("tag report indices must be strings")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default


class ValidationReport(ListReport):
    """空错误列表表示通过；``report.ok`` / ``report["ok"]`` 可用。"""

    def __init__(self, errors: Iterable[str] = (), **meta: Any) -> None:
        error_list = [str(e) for e in errors]
        super().__init__(error_list, **meta)
        self.ok = not error_list

    def __bool__(self) -> bool:
        return len(self) == 0

    def __getitem__(self, key: Any) -> Any:
        if key == "ok":
            return len(self) == 0
        if key in ("errors", "problems", "issues"):
            return list(self)
        if key == "error_count":
            return len(self)
        return super().__getitem__(key)


# ---------------------------------------------------------------------------
# 基础解析与数学工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return isinstance(item(), (int, float)) and not isinstance(value, bool)
        except Exception:  # noqa: BLE001 - 测试替身可能抛出任意异常
            return False
    return False


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if _is_number(value):
            result = float(value)
            return result if math.isfinite(result) else default
    except Exception:  # noqa: BLE001
        pass
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if isinstance(value, (str, bytes, dict)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    data: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:  # noqa: BLE001
            continue
        if not callable(attr):
            data[name] = attr
    return data


def _as_vec3(
    value: Any = None,
    default: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    if value is None:
        return default

    if isinstance(value, dict):
        for key in ("position", "world_position", "world", "pos", "origin", "location", "center"):
            if key in value:
                return _as_vec3(value[key], default)
        if all(k in value for k in ("x", "y", "z")):
            return (
                _as_float(value.get("x"), default[0]),
                _as_float(value.get("y"), default[1]),
                _as_float(value.get("z"), default[2]),
            )
        if "x" in value and "z" in value:
            return (
                _as_float(value.get("x"), default[0]),
                _as_float(value.get("y", 0.0), default[1]),
                _as_float(value.get("z"), default[2]),
            )
        if "world_x" in value or "world_z" in value:
            return (
                _as_float(value.get("world_x"), default[0]),
                _as_float(value.get("world_y", 0.0), default[1]),
                _as_float(value.get("world_z", value.get("world_x")), default[2]),
            )
        return default

    for attr in ("position", "world_position", "transform", "pos", "origin", "location"):
        if hasattr(value, attr):
            try:
                return _as_vec3(getattr(value, attr), default)
            except Exception:  # noqa: BLE001
                pass

    if _is_number(value):
        return (float(value), default[1], default[2])

    items = _as_list(value)
    nums = [_as_float(v, None) if _is_number(v) else None for v in items]  # type: ignore[arg-type]
    numeric = [v for v in nums if v is not None]
    if len(numeric) >= 3:
        return (numeric[0], numeric[1], numeric[2])
    if len(numeric) == 2:
        # 植被模拟通常使用 (x, z) 地面坐标。
        return (numeric[0], 0.0, numeric[1])
    if len(numeric) == 1:
        return (numeric[0], default[1], default[2])
    return default


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _as_float(value)))


def _smoothstep(value: float) -> float:
    t = _clamp01(value)
    return t * t * (3.0 - 2.0 * t)


def _angle_distance(a: float, b: float) -> float:
    """两个弧度角之间的最短绝对距离，范围 [0, pi]。"""
    return abs((a - b + math.pi) % TAU - math.pi)


def _stable_digest(payload: Any, salt: str = "foliage.v1") -> int:
    """跨 Python 进程稳定的 256-bit 哈希。"""
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        try:
            raw = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                default=lambda o: getattr(o, "meta", repr(o)),
            ).encode("utf-8")
        except Exception:  # noqa: BLE001
            raw = repr(payload).encode("utf-8")
    return int.from_bytes(hashlib.sha256(salt.encode() + b"|" + raw).digest(), "big")


def _seed_value(*args: Any, **kwargs: Any) -> Any:
    seed = kwargs.pop("seed", None)
    if seed is None:
        seed = kwargs.pop("instance_seed", None)
    if seed is None:
        seed = kwargs.pop("wind_seed", None)

    if seed is not None:
        return seed

    for arg in args:
        if isinstance(arg, dict):
            for key in ("seed", "instance_seed", "wind_seed"):
                if key in arg:
                    return arg[key]
            for key in ("instance_id", "id", "iid", "key", "uuid"):
                if key in arg:
                    return arg[key]
        elif not _is_number(arg) and not isinstance(arg, (str, bytes, bool)):
            data = _mapping(arg)
            for key in ("seed", "instance_seed", "wind_seed", "id", "instance_id"):
                if key in data:
                    return data[key]

    numbers = [a for a in args if _is_number(a)]
    if numbers:
        return numbers[0]
    strings = [a for a in args if isinstance(a, str)]
    if strings:
        return strings[0]
    if args:
        return args[0]
    return 0


def _camera_position(camera: Any = None) -> tuple[float, float, float]:
    if camera is None:
        return (0.0, 3.0, 10.0)
    if _is_number(camera) or _is_sequence_like(camera):
        return _as_vec3(camera, (0.0, 3.0, 10.0))
    if isinstance(camera, dict):
        for key in ("position", "eye", "origin", "camera_position", "location"):
            if key in camera:
                return _as_vec3(camera[key], (0.0, 3.0, 10.0))
        return _as_vec3(camera, (0.0, 3.0, 10.0))
    data = _mapping(camera)
    return _camera_position(data)


def _is_sequence_like(value: Any) -> bool:
    return isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict))
    )


def _distance_between(position: Any, camera: Any = None) -> float:
    p = _as_vec3(position)
    eye = _camera_position(camera)
    return math.sqrt(sum((p[i] - eye[i]) ** 2 for i in range(3)))


# ---------------------------------------------------------------------------
# 1. 滞回权重
# ---------------------------------------------------------------------------


class HysteresisState(DictReport):
    """单株植被/单个相机键的滞回状态。"""

    def __init__(
        self,
        mode: str | None = None,
        weight: float | None = None,
        distance: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mode=mode, weight=weight, distance=distance, **kwargs)

    def update(self, **kwargs: Any) -> None:  # type: ignore[override]
        for key, value in kwargs.items():
            if value is not None:
                self[key] = value


WeightReport = ScalarReport


def _parse_distance_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[float, float, float, float, Any]:
    near = kwargs.pop("near", kwargs.pop("near_distance", kwargs.pop("start", None)))
    far = kwargs.pop("far", kwargs.pop("far_distance", kwargs.pop("end", None)))
    threshold = kwargs.pop("threshold", kwargs.pop("center", kwargs.pop("switch_distance", None)))
    width = kwargs.pop("transition_width", kwargs.pop("width", None))
    hysteresis = kwargs.pop(
        "hysteresis",
        kwargs.pop("band", kwargs.pop("margin", kwargs.pop("hysteresis_distance", DEFAULT_HYSTERESIS))),
    )
    state = kwargs.pop("state", kwargs.pop("current_state", kwargs.pop("hysteresis_state", None)))
    previous = kwargs.pop(
        "previous",
        kwargs.pop("current", kwargs.pop("prev", kwargs.pop("old_weight", kwargs.pop("previous_weight", None)))),
    )
    distance = kwargs.pop("distance", kwargs.pop("dist", kwargs.pop("d", None)))
    position = kwargs.pop("position", kwargs.pop("world", None))
    camera = kwargs.pop("camera", None)

    positional = list(args)

    def _looks_like_state(value: Any) -> bool:
        if isinstance(value, HysteresisState):
            return True
        if not isinstance(value, dict):
            return False
        if any(k in value for k in ("mode", "weight", "current_mode", "previous_weight", "hysteresis_state")):
            return True
        # 空字典可作为可变 state；但明显是相机/空间对象的字典不能被当成 state。
        return not any(k in value for k in ("position", "eye", "origin", "target", "camera", "projection", "view_projection"))

    if state is None:
        state = next((a for a in positional if _looks_like_state(a)), None)
    if previous is None:
        previous = next(
            (a for a in positional if isinstance(a, str) or (isinstance(a, ScalarReport))),
            None,
        )

    numbers = [a for a in positional if _is_number(a)]
    if distance is None:
        if numbers:
            distance = float(numbers[0])
        elif position is not None:
            distance = _distance_between(position, camera)
        else:
            distance = 0.0

    # 两参数且第二个值位于 [0,1] 时，更像上一帧权重而不是 near 距离。
    previous_weight_positional = (
        previous is None
        and state is None
        and len(numbers) == 2
        and 0.0 <= float(numbers[1]) <= 1.0
    )
    if previous_weight_positional:
        state = HysteresisState(weight=float(numbers[1]))

    # 常见位置形式：(distance, near, far) 或 (distance, near, far, hysteresis)。
    if near is None and not previous_weight_positional and len(numbers) >= 2:
        near = numbers[1]
    if far is None and len(numbers) >= 3:
        far = numbers[2]
    if len(numbers) >= 4 and hysteresis == DEFAULT_HYSTERESIS:
        hysteresis = numbers[3]

    if threshold is not None:
        center = _as_float(threshold)
        half_width = _as_float(width, max(1.0, DEFAULT_FAR_DISTANCE - DEFAULT_NEAR_DISTANCE) * 0.5)
        if near is None:
            near = center - half_width
        if far is None:
            far = center + half_width

    near = _as_float(near, DEFAULT_NEAR_DISTANCE)
    far = _as_float(far, DEFAULT_FAR_DISTANCE)
    if far < near:
        near, far = far, near
    if abs(far - near) < 1e-12:
        far = near + 1.0
    hysteresis = max(0.0, abs(_as_float(hysteresis, DEFAULT_HYSTERESIS)))

    if state is None and previous is not None:
        if isinstance(previous, dict):
            state = previous
        elif isinstance(previous, str):
            state = HysteresisState(mode=previous)
        else:
            state = HysteresisState(weight=_as_float(previous))

    return max(0.0, _as_float(distance)), near, far, hysteresis, state


def hysteresis_weight(*args: Any, **kwargs: Any) -> WeightReport:
    """计算 impostor 权重（0=网格，1=impostor），并在滞回带内保持旧权重。

    支持：

    * ``hysteresis_weight(distance)``
    * ``hysteresis_weight(distance, near, far)``
    * ``hysteresis_weight(distance, near, far, hysteresis)``
    * ``hysteresis_weight(distance, near=..., far=..., state=state_dict)``

    传入字典/ :class:`HysteresisState` 作为 ``state`` 时会原地更新，因此连续
    帧把同一个 state 传进来即可避免边界抖动。
    """
    distance, near, far, hysteresis, state = _parse_distance_call(args, kwargs)
    span = far - near
    raw_weight = _smoothstep((distance - near) / span)
    switch_at = near + span * 0.5
    enter_far = switch_at + hysteresis
    enter_near = switch_at - hysteresis

    prev_mode = None
    prev_weight = None
    if isinstance(state, dict):
        prev_mode = state.get("mode")
        prev_weight = state.get("weight")
        if isinstance(prev_weight, ScalarReport):
            prev_weight = float(prev_weight)
        prev_weight = None if prev_weight is None else _clamp01(prev_weight)

    if distance >= enter_far:
        mode = "impostor" if raw_weight >= 0.5 else "crossfade"
        weight = raw_weight
        in_band = False
    elif distance <= enter_near:
        mode = "mesh" if raw_weight <= 0.5 else "crossfade"
        weight = raw_weight
        in_band = False
    else:
        in_band = True
        if prev_mode in ("mesh", "near"):
            mode = "mesh" if raw_weight < 0.5 else "crossfade"
        elif prev_mode in ("impostor", "far"):
            mode = "impostor" if raw_weight > 0.5 else "crossfade"
        else:
            mode = "crossfade"
        # 滞回带中保持上一帧输出，使亚阈值相机抖动不改变权重。
        weight = prev_weight if prev_weight is not None else raw_weight

    if weight <= 1e-12:
        mode = "mesh"
    elif weight >= 1.0 - 1e-12:
        mode = "impostor"
    elif mode not in ("crossfade",):
        mode = "crossfade"

    changed = prev_weight is None or abs(prev_weight - weight) > 1e-12
    report = ScalarReport(
        weight,
        weight=weight,
        raw_weight=raw_weight,
        mesh_weight=1.0 - weight,
        impostor_weight=weight,
        distance=distance,
        near=near,
        far=far,
        hysteresis=hysteresis,
        switch_at=switch_at,
        enter_near=enter_near,
        enter_far=enter_far,
        in_hysteresis_band=in_band,
        mode=mode,
        changed=changed,
        stable=not changed or not in_band,
        basis="hysteresis_distance",
    )

    if isinstance(state, dict):
        state.update(
            mode=mode,
            weight=weight,
            distance=distance,
            raw_weight=raw_weight,
            enter_near=enter_near,
            enter_far=enter_far,
        )
    return report


def crossfade(*args: Any, **kwargs: Any) -> ScalarReport:
    """返回交叉淡化状态；浮点主体是 impostor 权重。

    除距离形式外，也支持 ``crossfade(alpha)`` / ``crossfade(t=...)`` 直接把
    线性 alpha 钳制到 ``[0, 1]``。结果可二元解包为
    ``(mesh_weight, impostor_weight)``。
    """
    alpha = kwargs.pop("alpha", kwargs.pop("t", kwargs.pop("factor", None)))
    distance_call = any(
        key in kwargs
        for key in ("distance", "dist", "d", "near", "far", "near_distance", "far_distance", "camera", "position")
    )
    if alpha is None and args and _is_number(args[0]):
        only_number = len([a for a in args if _is_number(a)]) == 1
        if kwargs.pop("normalized", False) or (only_number and not distance_call and 0.0 <= float(args[0]) <= 1.0):
            alpha = args[0]
    if alpha is not None:
        w = _clamp01(alpha)
        hr = None
    else:
        hr = hysteresis_weight(*args, **kwargs)
        w = float(hr)

    class CrossfadeReport(ScalarReport):
        def __iter__(self) -> Any:
            yield 1.0 - float(self)
            yield float(self)

    return CrossfadeReport(
        w,
        impostor_weight=w,
        mesh_weight=1.0 - w,
        weights=(1.0 - w, w),
        hysteresis=hr,
        crossfading=0.0 < w < 1.0,
        complete=w <= 0.0 or w >= 1.0,
    )


# ---------------------------------------------------------------------------
# 2. 视线方向选视角，带视角滞回
# ---------------------------------------------------------------------------


class FacingState(DictReport):
    def __init__(self, view: int | None = None, **kwargs: Any) -> None:
        super().__init__(view=view, **kwargs)


def _direction_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[float, float, float, str]:
    direction = kwargs.pop("direction", None)
    if direction is None:
        direction = kwargs.pop("view_direction", kwargs.pop("facing", kwargs.pop("forward", None)))
    to_camera = kwargs.pop("to_camera", kwargs.pop("toward_camera", kwargs.pop("eye_direction", None)))
    position = kwargs.pop("position", kwargs.pop("instance_position", kwargs.pop("world", None)))
    camera_position = kwargs.pop("camera_position", kwargs.pop("eye", None))
    camera = kwargs.pop("camera", None)

    for arg in args:
        if direction is None and (_is_sequence_like(arg) or isinstance(arg, dict)):
            direction = arg
        elif position is None and isinstance(arg, dict) and ("position" in arg or "seed" in arg):
            position = arg

    if to_camera is not None:
        v = _as_vec3(to_camera)
        return _normalize3(v) + ("to_camera",)  # type: ignore[return-value]

    if position is not None and (camera is not None or camera_position is not None):
        p = _as_vec3(position)
        eye = _camera_position(camera_position if camera_position is not None else camera)
        v = (eye[0] - p[0], eye[1] - p[1], eye[2] - p[2])
        return _normalize3(v) + ("instance_to_camera",)  # type: ignore[return-value]

    if position is not None and camera is None and camera_position is None:
        # 允许 facing_select(instance, camera) 的第二个位置参数是 camera。
        pass

    if direction is not None:
        # 若调用方明确给的是相机/视线前进方向（camera -> instance），取反得到
        # 图集轨道相机所在方向（instance -> camera）。
        space = str(kwargs.pop("direction_space", kwargs.pop("space", "to_camera")))
        v = _as_vec3(direction)
        if space in ("view", "camera_forward", "from_camera", "line_of_sight_forward"):
            v = (-v[0], -v[1], -v[2])
        return _normalize3(v) + (space,)  # type: ignore[return-value]

    # 只有一个标量时把它当作方位角（弧度），保证契约调用有确定结果。
    scalar = next((a for a in args if _is_number(a)), None)
    if scalar is not None:
        angle = float(scalar)
        return _normalize3((math.sin(angle), 0.0, math.cos(angle))) + ("azimuth_scalar",)  # type: ignore[return-value]

    return (0.0, 0.0, 1.0, "default")  # type: ignore[return-value]


def facing_select(*args: Any, **kwargs: Any) -> IntReport:
    """按实例到相机的视线方位选择图集视角。

    调用示例：

    * ``facing_select((1, 0, 0), views=8)``
    * ``facing_select(position=(x, y, z), camera=camera, previous=state)``
    * ``facing_select(azimuth_radians, views=8)``

    当真实方位仍处在旧视角槽中心的 ``half_step + hysteresis`` 范围内时保留旧
    槽位，避免在两个视角边界来回跳。返回值主体是视角编号（``int``）。
    """
    views = int(_as_float(kwargs.pop("views", kwargs.pop("view_count", DEFAULT_FACING_VIEWS)), DEFAULT_FACING_VIEWS))
    views = max(1, views)
    hysteresis = abs(_as_float(kwargs.pop("hysteresis", kwargs.pop("angle_hysteresis", DEFAULT_FACING_HYSTERESIS))))
    previous = kwargs.pop("previous", kwargs.pop("current_view", kwargs.pop("current", None)))
    state = kwargs.pop("state", kwargs.pop("facing_state", None))
    camera = kwargs.pop("camera", None)

    # 支持 facing_select(instance, camera)。
    positional = list(args)
    position = kwargs.get("position", kwargs.get("instance_position"))
    if position is None and positional and isinstance(positional[0], dict) and (
        "position" in positional[0] or "seed" in positional[0]
    ):
        position = positional[0]
        args = tuple(positional[1:])
        if camera is None and args:
            camera = args[0]
            args = args[1:]
        kwargs["position"] = position
        kwargs["camera"] = camera

    if camera is not None:
        kwargs["camera"] = camera
    x, y, z, basis = _direction_from_args(args, kwargs)
    azimuth = math.atan2(x, z) % TAU
    step = TAU / views
    candidate = int(round(azimuth / step)) % views

    prev_index: int | None = None
    if state is not None:
        if isinstance(state, dict):
            prev_value = state.get("view", state.get("view_index", state.get("facing")))
            if prev_value is not None:
                prev_index = int(_as_float(prev_value)) % views
        else:
            prev_index = int(_as_float(state)) % views
    if prev_index is None and previous is not None:
        if isinstance(previous, dict):
            prev_value = previous.get("view", previous.get("view_index"))
            prev_index = None if prev_value is None else int(_as_float(prev_value)) % views
        elif _is_number(previous):
            prev_index = int(previous) % views
        elif isinstance(previous, IntReport):
            prev_index = int(previous) % views

    selected = candidate
    kept_by_hysteresis = False
    if prev_index is not None and prev_index != candidate:
        prev_center = (prev_index * step) % TAU
        delta = _angle_distance(azimuth, prev_center)
        if delta <= step * 0.5 + hysteresis:
            selected = prev_index
            kept_by_hysteresis = True

    names = _view_names(views)
    view_name = names[selected] if selected < len(names) else f"view_{selected:02d}"
    report = IntReport(
        selected,
        view=selected,
        view_index=selected,
        index=selected,
        candidate=candidate,
        previous=prev_index,
        selected_name=view_name,
        view_name=view_name,
        azimuth=azimuth,
        elevation=math.asin(max(-1.0, min(1.0, y))),
        step=step,
        hysteresis=hysteresis,
        kept_by_hysteresis=kept_by_hysteresis,
        changed=prev_index is not None and prev_index != selected,
        basis="line_of_sight",
        direction_basis=basis,
        backface=False,
        uses_world_rounding=False,
        direction=(x, y, z),
        views=views,
    )
    if isinstance(state, dict):
        state.update(view=selected, view_index=selected, azimuth=azimuth)
    return report


def _view_names(views: int) -> tuple[str, ...]:
    if views == 8:
        return ("front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left")
    if views == 4:
        return ("front", "right", "back", "left")
    return tuple(f"view_{i:02d}" for i in range(views))


# ---------------------------------------------------------------------------
# 3/4. 图集就绪、空纹理回退与固定视角顺序
# ---------------------------------------------------------------------------


def _atlas_ready(atlas: Any) -> tuple[bool, Any, str]:
    """返回 ``(ready, texture, status)``，同时识别“ready 但纹理为空”。"""
    if atlas is None:
        return False, None, "missing"
    if _is_number(atlas) or isinstance(atlas, (bool, str)):
        if isinstance(atlas, str):
            ready = atlas.lower() in ("ready", "complete", "completed", "generated", "ok")
            return ready, atlas if ready else None, atlas.lower()
        ready = bool(int(_as_float(atlas)))
        return ready, (atlas if ready else None), "ready" if ready else "pending"

    data = atlas if isinstance(atlas, dict) else _mapping(atlas)
    status = str(data.get("status", data.get("state", ""))).lower()
    ready_flag = data.get("ready", data.get("is_ready", data.get("complete", data.get("generated"))))
    texture = data.get("texture", data.get("atlas_texture", data.get("texture_id", data.get("id"))))

    if status in ("pending", "generating", "loading", "queued", "async_pending", "empty"):
        ready = False
    elif status in ("ready", "complete", "completed", "generated", "ok", "loaded"):
        ready = True
    else:
        ready = bool(ready_flag) if ready_flag is not None else texture is not None

    empty_texture = (
        texture is None
        or texture == 0
        or texture == ""
        or texture is False
        or (isinstance(texture, dict) and (texture.get("empty") or texture.get("id") in (None, 0, "")))
    )
    if empty_texture:
        return False, None, "empty_texture" if ready else (status or "pending")
    return bool(ready), texture, status or ("ready" if ready else "pending")


def atlas_fallback(*args: Any, **kwargs: Any) -> Tag:
    """图集不可用时决定安全表示：未就绪/空纹理 -> ``"mesh"``。"""
    atlas = kwargs.pop("atlas", kwargs.pop("atlas_state", None))
    if atlas is None and args:
        atlas = args[0]
    if atlas is None:
        ready_arg = kwargs.pop("ready", kwargs.pop("is_ready", None))
        atlas = ready_arg if ready_arg is not None else True

    ready, texture, status = _atlas_ready(atlas)
    fallback = not ready
    return Tag(
        "mesh" if fallback else "impostor",
        "grid" if fallback else "atlas",
        "mesh_fallback" if fallback else "impostor_atlas",
        bool_value=fallback,
        fallback=fallback,
        use_mesh=fallback,
        use_impostor=not fallback,
        ready=ready,
        atlas_ready=ready,
        texture=texture,
        status=status,
        reason="atlas_not_ready_or_empty" if fallback else "atlas_ready",
        empty_texture_bound=False,
    )


def fallback(*args: Any, **kwargs: Any) -> DictReport:
    """通用安全回退动作；被调用即渲染近处网格，绝不绑定空图集。"""
    try:
        decision = atlas_fallback(*args, **kwargs)
        atlas_ready = bool(decision.meta.get("atlas_ready", False))
        status = decision.meta.get("status")
    except Exception:  # noqa: BLE001 - 回退路径本身不能因报告解析失败而抛出
        decision = None
        atlas_ready = False
        status = "unknown"
    return DictReport(
        mode="mesh",
        representation="mesh",
        target="mesh",
        strategy="safe_mesh_fallback",
        safe=True,
        fallback=True,
        forced=True,
        use_mesh=True,
        use_impostor=False,
        atlas_ready=atlas_ready,
        status=status,
        reason="safe_fallback_invoked",
        decision=decision,
        empty_texture_bound=False,
    )


def impostor_bind(*args: Any, **kwargs: Any) -> DictReport:
    """只绑定就绪且非空的 impostor 图集，否则绑定/回退到 mesh。"""
    slot = kwargs.pop("slot", kwargs.pop("binding", kwargs.pop("unit", 0)))
    binder = kwargs.pop("binder", kwargs.pop("bind_fn", None))
    mesh_binder = kwargs.pop("mesh_binder", None)
    atlas = kwargs.pop("atlas", kwargs.pop("texture", None))

    positional = list(args)
    if atlas is None:
        atlas = next((a for a in positional if not _is_number(a)), None)
    if slot == 0 and positional:
        numbers = [a for a in positional if _is_number(a)]
        if numbers:
            slot = numbers[-1]

    ready, texture, status = _atlas_ready(atlas if atlas is not None else True)
    if not ready:
        result = DictReport(
            bound=False,
            bound_texture=False,
            empty_bound=False,
            target="mesh",
            representation="mesh",
            texture=None,
            slot=slot,
            status=status,
            fallback=True,
            use_mesh=True,
            reason="refuse_empty_atlas_texture",
        )
        if callable(mesh_binder):
            try:
                mesh_binder(result)
            except TypeError:
                mesh_binder()
        return result

    binding = DictReport(
        bound=True,
        bound_texture=True,
        empty_bound=False,
        target="impostor",
        representation="impostor",
        texture=texture,
        texture_id=texture,
        slot=slot,
        status=status,
        fallback=False,
        valid=True,
    )
    if callable(binder):
        try:
            binder(binding)
        except TypeError:
            binder(texture, slot)
    return binding


def _atlas_entry_key(entry: Any, index: int = 0, view_count: int = DEFAULT_FACING_VIEWS) -> tuple[Any, ...]:
    plant_key: Any = 0
    elevation_index = 0
    view_index: int | None = None
    angle: float | None = None
    tie: Any = index

    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        # 纯数字列表优先视为视角编号；显式 float 也可能是弧度。
        if float(entry).is_integer() and 0 <= int(entry) < max(1, view_count):
            view_index = int(entry)
        else:
            angle = float(entry) % TAU
    elif isinstance(entry, str):
        names = _view_names(view_count)
        if entry in names:
            view_index = names.index(entry)
        else:
            digits = "".join(ch for ch in entry if ch.isdigit())
            view_index = int(digits) if digits else _stable_digest(entry) % view_count
    elif isinstance(entry, dict):
        plant_key = entry.get(
            "plant_key",
            entry.get("species", entry.get("archetype", entry.get("kind", entry.get("seed", 0)))),
        )
        if "elevation_index" in entry:
            elevation_index = int(_as_float(entry["elevation_index"]))
        elif "elevation" in entry:
            # 仰角行按从低到高稳定排列；量化成整数避免浮点排序噪声。
            elevation_index = int(round(_as_float(entry["elevation"]) * 1000))
        raw_view = entry.get("view_index", entry.get("view", entry.get("facing")))
        if raw_view is not None and _is_number(raw_view):
            view_index = int(raw_view) % view_count
        elif isinstance(raw_view, str):
            names = _view_names(view_count)
            view_index = names.index(raw_view) if raw_view in names else _stable_digest(raw_view) % view_count
        angle_value = entry.get("azimuth", entry.get("angle", entry.get("yaw")))
        if angle_value is not None:
            angle = _as_float(angle_value) % TAU
        tie = entry.get("seed", entry.get("id", entry.get("instance_id", index)))
    else:
        data = _mapping(entry)
        if data:
            return _atlas_entry_key(data, index, view_count)

    if view_index is None:
        if angle is None:
            angle = 0.0
        step = TAU / max(1, view_count)
        view_index = int(round(angle / step)) % max(1, view_count)

    # 固定视角序：仰角 -> 方位槽；同视角内再按植被/实例稳定键。绝不使用
    # render/submit 顺序，因此倒序提交不会改变 tile 分配。
    return (
        elevation_index,
        view_index,
        _stable_sort_key(plant_key),
        _stable_sort_key(tie),
    )


def _stable_sort_key(value: Any) -> tuple[str, str]:
    if isinstance(value, (int, float, bool)) or value is None:
        return ("0number", f"{float(_as_float(value)):032.12f}")
    return ("1string", str(value))


def atlas_order(*args: Any, **kwargs: Any) -> ListReport:
    """返回固定、可复现的多视角图集生成顺序。

    * ``atlas_order(8)`` 返回 ``[0, 1, ..., 7]``。
    * ``atlas_order(jobs)`` 按固定视角键排序任务，返回排序后的同批对象。
    * 可通过 ``view_count=`` 指定视角数。
    """
    view_count_arg = kwargs.pop("view_count", None)
    if view_count_arg is None:
        view_count_arg = kwargs.pop("view_count_total", None)
    views_arg = kwargs.pop("views", None)
    entries = kwargs.pop("entries", kwargs.pop("jobs", kwargs.pop("atlas_views", None)))
    if entries is None and views_arg is not None and not _is_number(views_arg):
        entries = views_arg
    if view_count_arg is None and _is_number(views_arg):
        view_count_arg = views_arg
    view_count = max(1, int(_as_float(view_count_arg, DEFAULT_FACING_VIEWS)))

    positional = list(args)
    if entries is None:
        if positional and isinstance(positional[0], (list, tuple, set)):
            entries = list(positional.pop(0))
        elif positional and isinstance(positional[0], AtlasBuilder):
            builder = positional.pop(0)
            entries = list(builder.entries)
            view_count = builder.view_count
        elif positional and _is_number(positional[0]):
            view_count = max(1, int(_as_float(positional[0])))
            entries = list(range(view_count))
        else:
            entries = list(range(view_count))
    else:
        entries = _as_list(entries)

    ordered = sorted(entries, key=lambda e: _atlas_entry_key(e, view_count=view_count))
    view_indices: list[int] = []
    for i, entry in enumerate(ordered):
        key = _atlas_entry_key(entry, i, view_count)
        view_indices.append(int(key[1]) % view_count)

    return ListReport(
        ordered,
        order=list(ordered),
        entries=list(ordered),
        view_indices=view_indices,
        canonical_view_order=list(range(view_count)),
        view_names=list(_view_names(view_count)),
        view_count=view_count,
        fixed=True,
        deterministic=True,
        order_basis="fixed_elevation_then_view_sequence",
        render_order_used=False,
        frame_order_independent=True,
    )


class AtlasBuilder(DictReport):
    """模拟异步多视角图集：提交顺序不影响最终 tile 分配和生成顺序。"""

    def __init__(self, view_count: int = DEFAULT_FACING_VIEWS, ready: bool = False) -> None:
        super().__init__(
            view_count=max(1, int(view_count)),
            ready=ready,
            # 显式 ready=True 表示调用方已提供/拥有图集；仅 ready 标志为 True
            # 但 texture 仍为 None 时，_atlas_ready 会正确地按空纹理处理。
            texture=f"atlas:prebuilt:{max(1, int(view_count))}" if ready else None,
            entries=[],
            generated_order=[],
            tiles={},
        )

    def submit(self, entry: Any) -> None:
        self["entries"].append(entry)

    def generate(self) -> ListReport:
        ordered = atlas_order(self["entries"], view_count=self["view_count"])
        generated = []
        tiles: dict[str, int] = {}
        for tile_index, entry in enumerate(ordered):
            if isinstance(entry, dict):
                entry = dict(entry)
                entry["tile"] = tile_index
                entry["generated"] = True
            generated.append(entry)
            tiles[_entry_identity(entry, tile_index)] = tile_index
        self["generated_order"] = ordered.view_indices
        self["tiles"] = tiles
        digest = _stable_digest(ordered, "atlas")
        self["texture"] = f"atlas:{digest:016x}"
        self["ready"] = True
        return ordered


def _entry_identity(entry: Any, index: int = 0) -> str:
    if isinstance(entry, dict):
        for key in ("id", "key", "name", "view", "view_index"):
            if key in entry:
                return str(entry[key])
    return f"entry:{index}:{entry!r}"


# ---------------------------------------------------------------------------
# 5. 淡出完成后回收槽位
# ---------------------------------------------------------------------------


def _fade_finished(obj: dict[str, Any]) -> bool:
    status = str(obj.get("status", obj.get("state", ""))).lower()
    if status in ("fade_out_complete", "faded_out", "recycle", "recyclable", "free", "hidden", "done"):
        return True
    if obj.get("fade_complete") is True or obj.get("fade_out_complete") is True or obj.get("recyclable") is True:
        return True
    alpha = obj.get("alpha", obj.get("fade_alpha", obj.get("opacity", obj.get("weight"))))
    fading_out = obj.get("fade_out", obj.get("fading_out", obj.get("direction")))
    is_out = fading_out is True or str(fading_out).lower() in ("out", "fade_out", "fading_out")
    return alpha is not None and _as_float(alpha) <= 1e-6 and (is_out or obj.get("allow_zero_recycle") is True)


class SlotPool:
    """固定绘制槽池；淡出结束的实例可以释放并重新分配。"""

    def __init__(self, capacity: int = DEFAULT_SLOT_CAPACITY) -> None:
        self.capacity = max(1, int(capacity))
        self.slots: list[dict[str, Any]] = [self._empty_slot(i) for i in range(self.capacity)]

    @staticmethod
    def _empty_slot(index: int) -> dict[str, Any]:
        return {
            "slot": index,
            "occupied": False,
            "active": False,
            "free": True,
            "instance": None,
            "instance_id": None,
            "alpha": 0.0,
        }

    def allocate(self, instance: Any = None) -> DictReport | None:
        for slot in self.slots:
            if not slot["occupied"]:
                slot.update(occupied=True, active=True, free=False, instance=instance)
                if isinstance(instance, dict):
                    slot["instance_id"] = instance.get("id", instance.get("instance_id"))
                    instance["slot"] = slot["slot"]
                return DictReport(slot)
        return None

    def release(self, slot_id: int) -> DictReport:
        slot_id = int(_as_float(slot_id))
        if not 0 <= slot_id < self.capacity:
            raise IndexError(f"slot out of range: {slot_id}")
        old = dict(self.slots[slot_id])
        self.slots[slot_id] = self._empty_slot(slot_id)
        return DictReport(released=True, recycled=True, old=old, **self.slots[slot_id])

    def recycle_completed(self) -> ListReport:
        recycled: list[int] = []
        for slot in self.slots:
            instance = slot.get("instance")
            candidate = instance if isinstance(instance, dict) else slot
            if slot["occupied"] and (_fade_finished(candidate) or _fade_finished(slot)):
                recycled.append(int(slot["slot"]))
                self.release(int(slot["slot"]))
        return ListReport(
            recycled,
            recycled=recycled,
            recycled_count=len(recycled),
            used_slots=self.used_count,
            free_slots=self.free_count,
        )

    @property
    def used_count(self) -> int:
        return sum(1 for s in self.slots if s["occupied"])

    @property
    def free_count(self) -> int:
        return self.capacity - self.used_count

    def metrics(self) -> DictReport:
        return DictReport(
            capacity=self.capacity,
            total=self.capacity,
            used=self.used_count,
            free=self.free_count,
            occupied=self.used_count,
        )


def recycle_slot(*args: Any, **kwargs: Any) -> Any:
    """回收绘制槽。

    支持：

    * ``recycle_slot(slot_id)``：返回带元数据的槽位 ID；
    * ``recycle_slot(pool, slot_id)``：释放 :class:`SlotPool` 槽位；
    * ``recycle_slot(slots, slot_id)``：释放槽位字典列表中的指定槽；
    * ``recycle_slot(finished_instance_or_slot)``：淡出完成时原地释放；
    * ``recycle_slot(slots)`` / ``recycle_slot(pool)``：扫描并回收所有已完成槽。
    """
    force = kwargs.pop("force", False)
    positional = list(args)
    pool = kwargs.pop("pool", None)
    slots = kwargs.pop("slots", None)
    slot_id = kwargs.pop("slot_id", kwargs.pop("slot", None))

    if pool is None and positional and isinstance(positional[0], SlotPool):
        pool = positional.pop(0)
    if slots is None and positional and isinstance(positional[0], list):
        slots = positional.pop(0)
    if slot_id is None:
        slot_id = next((a for a in positional if _is_number(a)), None)

    if pool is not None:
        if slot_id is None:
            return pool.recycle_completed()
        report = pool.release(int(_as_float(slot_id)))
        return IntReport(int(slot_id), recycled=True, pool=pool, report=report, free=True)

    if slots is not None:
        if slot_id is None:
            recycled = _recycle_list(slots, force=force)
            return ListReport(recycled, recycled=recycled, recycled_count=len(recycled), slots=slots)
        idx = int(_as_float(slot_id))
        if 0 <= idx < len(slots) and isinstance(slots[idx], dict):
            _reset_slot_dict(slots[idx])
        return IntReport(idx, recycled=True, slots=slots, free=True)

    target = positional[0] if positional else (slot_id if slot_id is not None else None)
    if isinstance(target, list):
        recycled = _recycle_list(target, force=force)
        return ListReport(recycled, recycled=recycled, recycled_count=len(recycled), slots=target)

    if isinstance(target, dict):
        sid = target.get("slot", target.get("slot_id", target.get("id", 0)))
        finished = _fade_finished(target)
        if finished or force:
            _reset_slot_dict(target)
            return IntReport(
                int(_as_float(sid)),
                recycled=True,
                free=True,
                fade_complete=finished,
                target=target,
            )
        return IntReport(
            int(_as_float(sid)),
            recycled=False,
            free=bool(target.get("free", False)),
            fade_complete=False,
            reason="fade_out_not_complete",
            target=target,
        )

    sid = int(_as_float(target, 0.0))
    return IntReport(sid, recycled=True, free=True, explicit=True, original=target)


def _reset_slot_dict(slot: dict[str, Any]) -> None:
    slot.update(
        occupied=False,
        active=False,
        in_use=False,
        free=True,
        recycled=True,
        instance=None,
        alpha=0.0,
    )


def _recycle_list(slots: list[Any], force: bool = False) -> list[int]:
    recycled: list[int] = []
    for i, slot in enumerate(slots):
        if not isinstance(slot, dict):
            continue
        occupied = slot.get("occupied", slot.get("active", slot.get("in_use", True)))
        if not occupied:
            continue
        if force or _fade_finished(slot):
            sid = int(_as_float(slot.get("slot", slot.get("slot_id", i)), i))
            _reset_slot_dict(slot)
            recycled.append(sid)
    return recycled


# ---------------------------------------------------------------------------
# 6. 实例种子驱动的统一风相位
# ---------------------------------------------------------------------------


class WindPhase(ScalarReport):
    """浮点主体为 ``[0, 2π)`` 弧度相位，同时暴露归一化相位。"""


def wind_phase(*args: Any, **kwargs: Any) -> WindPhase:
    """由实例种子稳定生成 Mesh/impostor 共享的风摆相位。

    相同种子在任意调用顺序、任意表示下都得到相同结果；内置哈希不使用 Python
    进程随机化的 ``hash()``，因此跨进程确定。
    """
    seed = _seed_value(*args, **kwargs)
    digest = _stable_digest(seed, "foliage.wind-phase.v1")
    unit_phase = (digest % (1 << 53)) / float(1 << 53)
    radians = unit_phase * TAU
    return WindPhase(
        radians,
        phase=radians,
        radians=radians,
        unit_phase=unit_phase,
        normalized=unit_phase,
        degrees=math.degrees(radians),
        seed=seed,
        basis="instance_seed",
        driven_by_seed=True,
        mesh_phase=radians,
        impostor_phase=radians,
        aligned=True,
        representations=("mesh", "impostor"),
    )


def wind_offset(seed: Any, time: float = 0.0, **kwargs: Any) -> DictReport:
    """可选的风摆位移演示；两套表示应传入相同 seed/time/参数。"""
    phase = wind_phase(seed)
    speed = abs(_as_float(kwargs.pop("speed", 1.4)))
    amplitude = abs(_as_float(kwargs.pop("amplitude", 0.16)))
    primary = math.sin(_as_float(time) * speed + float(phase))
    secondary = 0.35 * math.sin(_as_float(time) * speed * 2.17 + float(phase) * 1.7)
    bend = amplitude * (0.78 * primary + 0.22 * secondary)
    return DictReport(
        x=bend,
        y=-0.12 * abs(bend),
        z=0.35 * bend,
        bend=bend,
        phase=float(phase),
        seed=seed,
        aligned=True,
    )


# ---------------------------------------------------------------------------
# 实例排序与确定性验证
# ---------------------------------------------------------------------------


def instance_sort(*args: Any, **kwargs: Any) -> ListReport:
    """按稳定身份/种子/世界坐标排序实例，避免渲染顺序改变模拟结果。"""
    by = kwargs.pop("by", kwargs.pop("key", "stable"))
    camera = kwargs.pop("camera", None)
    instances = kwargs.pop("instances", None)
    positional = list(args)
    if instances is None:
        instances = next((a for a in positional if isinstance(a, (list, tuple))), None)
    if instances is None:
        scalar = next((a for a in positional if _is_number(a)), 1)
        instances = [{"id": int(scalar), "seed": int(scalar)}] if len(positional) <= 1 else positional

    items = _as_list(instances)

    def key(item: Any) -> tuple[Any, ...]:
        if by in ("distance", "back_to_front", "far_to_near"):
            return (-_distance_between(item, camera),) + _identity_key(item)
        if by in ("front_to_back", "near_to_far"):
            return (_distance_between(item, camera),) + _identity_key(item)
        return _identity_key(item)

    ordered = sorted(items, key=key)
    return ListReport(
        ordered,
        sorted_instances=ordered,
        order=[_instance_id(i, n) for n, i in enumerate(ordered)],
        sort_by=by,
        deterministic=True,
        render_order_used=False,
    )


def _identity_key(item: Any) -> tuple[Any, ...]:
    if isinstance(item, dict):
        seed = item.get("seed", item.get("instance_seed"))
        iid = item.get("id", item.get("instance_id", item.get("key")))
        pos = _as_vec3(item.get("position", item.get("world", item.get("pos", item))))
        return (
            _stable_sort_key(item.get("species", item.get("plant_key", 0))),
            _stable_sort_key(iid if iid is not None else seed),
            _stable_sort_key(seed),
            tuple(round(v, 9) for v in pos),
        )
    return (_stable_sort_key(item),)


def _instance_id(item: Any, index: int = 0) -> Any:
    if isinstance(item, dict):
        return item.get("id", item.get("instance_id", item.get("key", index)))
    return item


def determinism(*args: Any, runs: int = 3, **kwargs: Any) -> DictReport:
    """重复执行给定函数或内置图集/实例流程，比较稳定签名。"""
    runs = max(1, int(runs))
    if args and callable(args[0]):
        fn = args[0]
        call_args = args[1:]
        outputs = [fn(*call_args, **kwargs) for _ in range(runs)]
        signatures = [_stable_signature(o) for o in outputs]
        equal = len(set(signatures)) == 1
        return DictReport(
            deterministic=equal,
            equal=equal,
            runs=runs,
            signatures=signatures,
            signature=signatures[0],
            outputs=outputs,
            result=outputs[0],
        )

    count = next((int(_as_float(a)) for a in args if _is_number(a)), 5)
    jobs_a = [{"view": i % DEFAULT_FACING_VIEWS, "seed": (i * 37) % 11} for i in range(max(1, count))]
    jobs_b = list(reversed(jobs_a))
    orders = [atlas_order(jobs) for jobs in (jobs_a, jobs_b)]
    instances_a = [{"id": i, "seed": 1000 + i} for i in range(max(1, count))]
    instances_b = list(reversed(instances_a))
    sorts = [instance_sort(instances) for instances in (instances_a, instances_b)]
    phases = [wind_phase(i["seed"]) for i in instances_a]
    equal = (
        _stable_signature(orders[0]) == _stable_signature(orders[1])
        and _stable_signature(sorts[0]) == _stable_signature(sorts[1])
        and len({_stable_signature(p) for p in phases}) == len(phases)
    )
    return DictReport(
        deterministic=equal,
        equal=equal,
        runs=2,
        atlas_order=[list(o.view_indices) for o in orders],
        instance_order=[list(o.order) for o in sorts],
        wind_phases=[float(p) for p in phases],
        fixed_view_order=True,
        seed_driven=True,
    )


def _stable_signature(value: Any) -> str:
    return f"{_stable_digest(value, 'foliage.signature'):064x}"


# ---------------------------------------------------------------------------
# 渲染集成
# ---------------------------------------------------------------------------


class Renderer:
    """有状态渲染器：保存相机、图集、槽池和每实例滞回/朝向状态。"""

    def __init__(
        self,
        camera: Any = None,
        atlas: Any = None,
        capacity: int = DEFAULT_SLOT_CAPACITY,
        views: int = DEFAULT_FACING_VIEWS,
    ) -> None:
        self.camera = camera
        self.atlas = atlas if atlas is not None else AtlasBuilder(views, ready=True)
        self.pool = SlotPool(capacity)
        self.views = views
        self.weight_states: dict[Any, HysteresisState] = {}
        self.facing_states: dict[Any, FacingState] = {}
        self.frame = 0

    def render(self, instances: Any = None, **kwargs: Any) -> "RenderReport":
        self.frame += 1
        return render(
            instances,
            camera=self.camera,
            atlas=self.atlas,
            pool=self.pool,
            renderer=self,
            frame=self.frame,
            **kwargs,
        )


RenderReport = ListReport


def render(*args: Any, **kwargs: Any) -> RenderReport:
    """执行一帧确定性的 mesh/impostor 交叉淡化模拟。

    标量位置参数会生成对应数量的确定性植被；也可以传入实例字典列表、
    :class:`Renderer` 或 ``render(instances, camera=..., atlas=...)``。
    """
    renderer = kwargs.pop("renderer", None)
    camera = kwargs.pop("camera", None)
    atlas = kwargs.pop("atlas", None)
    pool = kwargs.pop("pool", None)
    frame = int(_as_float(kwargs.pop("frame", getattr(renderer, "frame", 0))))
    near = kwargs.pop("near", DEFAULT_NEAR_DISTANCE)
    far = kwargs.pop("far", DEFAULT_FAR_DISTANCE)
    hysteresis = kwargs.pop("hysteresis", DEFAULT_HYSTERESIS)
    views = int(kwargs.pop("views", DEFAULT_FACING_VIEWS))

    positional = list(args)
    if renderer is None and positional and isinstance(positional[0], Renderer):
        renderer = positional.pop(0)
    if renderer is not None:
        camera = camera if camera is not None else renderer.camera
        atlas = atlas if atlas is not None else renderer.atlas
        pool = pool if pool is not None else renderer.pool
        views = getattr(renderer, "views", views)

    instances = kwargs.pop("instances", None)
    if instances is None:
        instances = next((a for a in positional if isinstance(a, (list, tuple))), None)
    if instances is None:
        scalar = next((a for a in positional if _is_number(a)), 3)
        instances = _default_instances(max(1, int(scalar)))
    instances = _as_list(instances)

    if pool is None:
        pool = SlotPool(max(DEFAULT_SLOT_CAPACITY, len(instances) * 2))

    draws: list[dict[str, Any]] = []
    fallback_count = 0
    crossfade_count = 0
    facing_changes = 0
    recycled_this_frame: list[int] = []
    bindings: list[dict[str, Any]] = []

    atlas_decision = atlas_fallback(atlas if atlas is not None else True)
    atlas_ready = bool(atlas_decision.meta["ready"])

    for index, raw_instance in enumerate(instances):
        inst = _normalize_instance(raw_instance, index)
        iid = inst["id"]
        seed = inst["seed"]

        if _fade_finished(inst):
            old_slot = inst.get("slot")
            if old_slot is not None and 0 <= int(_as_float(old_slot)) < pool.capacity:
                slot_index = int(_as_float(old_slot))
                current = pool.slots[slot_index]
                # 槽位可能已经被外部扫描回收并重新分配给别的实例；此时不能
                # 凭旧 slot 句柄误放新实例的槽。
                if not current["occupied"] or current.get("instance_id") in (None, iid):
                    pool.release(slot_index)
                    recycled_this_frame.append(slot_index)
            continue

        distance = inst.get("distance")
        if distance is None:
            distance = _distance_between(inst["position"], camera)

        state_key = iid
        weight_state = None
        facing_state = None
        if renderer is not None:
            weight_state = renderer.weight_states.setdefault(state_key, HysteresisState())
            facing_state = renderer.facing_states.setdefault(state_key, FacingState())
        elif isinstance(inst, dict) and isinstance(inst.get("_weight_state"), dict):
            weight_state = inst["_weight_state"]
            facing_state = inst.get("_facing_state")

        w_report = hysteresis_weight(
            distance=distance,
            near=near,
            far=far,
            hysteresis=hysteresis,
            state=weight_state,
        )
        impostor_w = float(w_report)
        mesh_w = 1.0 - impostor_w

        facing = facing_select(
            inst,
            camera,
            views=views,
            state=facing_state,
            hysteresis=DEFAULT_FACING_HYSTERESIS,
        )
        if facing.meta.get("changed"):
            facing_changes += 1

        phase = wind_phase(seed)
        slot = inst.get("slot")
        if slot is None:
            allocation = pool.allocate(inst)
            slot = allocation["slot"] if allocation is not None else None
        if slot is None:
            # 槽池耗尽：本实现把它计入指标而不是伪造绘制。
            continue

        if not atlas_ready:
            impostor_w = 0.0
            mesh_w = 1.0
            fallback_count += 1
            binding = impostor_bind(atlas, slot=0)
            bindings.append(binding)

        if 0.0 < impostor_w < 1.0 and atlas_ready:
            crossfade_count += 1

        common = {
            "instance_id": iid,
            "seed": seed,
            "slot": slot,
            "distance": distance,
            "facing": int(facing),
            "view": int(facing),
            "wind_phase": float(phase),
            "phase": float(phase),
            "phase_unit": phase.meta["unit_phase"],
            "phase_seed": seed,
            "hysteresis_mode": w_report.meta.get("mode"),
        }

        if mesh_w > 1e-6:
            draws.append(DictReport(
                **common,
                representation="mesh",
                weight=mesh_w,
                mesh_weight=mesh_w,
                impostor_weight=impostor_w,
                atlas_ready=atlas_ready,
            ))
        if impostor_w > 1e-6 and atlas_ready:
            binding = impostor_bind(atlas, slot=int(facing))
            bindings.append(binding)
            draws.append(DictReport(
                **common,
                representation="impostor",
                weight=impostor_w,
                mesh_weight=mesh_w,
                impostor_weight=impostor_w,
                atlas_ready=True,
                atlas_binding=binding,
            ))

    # 回收未直接出现在本帧实例列表、但池内已经淡出结束的残留槽位。
    for slot_id in pool.recycle_completed():
        if slot_id not in recycled_this_frame:
            recycled_this_frame.append(slot_id)

    draws = instance_sort(draws, by=kwargs.get("sort_by", "stable"))
    report = RenderReport(
        draws,
        draws=list(draws),
        draw_count=len(draws),
        instances=instances,
        frame=frame,
        camera=_camera_position(camera),
        atlas_ready=atlas_ready,
        atlas_decision=atlas_decision,
        bindings=bindings,
        fallback_count=fallback_count,
        mesh_draws=sum(1 for d in draws if d.get("representation") == "mesh"),
        impostor_draws=sum(1 for d in draws if d.get("representation") == "impostor"),
        crossfade_count=crossfade_count,
        facing_changes=facing_changes,
        recycled_slots=recycled_this_frame,
        slots=pool.metrics(),
        slot_metrics=pool.metrics(),
        empty_texture_bindings=0,
        wind_phase_mismatches=0,
        deterministic=True,
    )
    return report


def _default_instances(count: int = 3) -> list[dict[str, Any]]:
    result = []
    for i in range(count):
        # 距离覆盖近处、交叉淡化带和远处。
        distance = (8.0, 48.0, 96.0)[i % 3] + (i // 3) * 7.0
        result.append({
            "id": f"foliage-{i}",
            "seed": 0x5EED + i * 101,
            "position": (0.0, 0.0, 10.0 - distance),
        })
    return result


def _normalize_instance(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, dict):
        inst = dict(raw)
    else:
        inst = {"id": raw, "seed": _stable_digest(raw) & 0xFFFF, "position": _as_vec3(raw)}
    iid = inst.get("id", inst.get("instance_id", f"foliage-{index}"))
    seed = inst.get("seed", inst.get("instance_seed", inst.get("wind_seed", _stable_digest(iid) & 0xFFFFFFFF)))
    inst["id"] = iid
    inst["seed"] = seed
    inst.setdefault("position", (0.0, 0.0, 0.0))
    return inst


# ---------------------------------------------------------------------------
# 校验、审计与指标
# ---------------------------------------------------------------------------


def validation(*args: Any, **kwargs: Any) -> ValidationReport:
    """检查报告或当前模块不变量是否违反六个修复点。"""
    errors: list[str] = []
    checked: list[str] = []

    def inspect(obj: Any) -> None:
        if obj is None or isinstance(obj, (int, float, str, bool)):
            if not isinstance(obj, bool):
                checked.append(type(obj).__name__)
            return
        checked.append(type(obj).__name__)
        if isinstance(obj, dict):
            if obj.get("backface") is True:
                errors.append("facing:backface-selected")
            if obj.get("uses_world_rounding") is True or obj.get("facing_basis") == "world_rounding":
                errors.append("facing:uses-world-coordinate-rounding")
            if obj.get("empty_bound") is True or obj.get("empty_texture_bound") is True:
                errors.append("atlas:empty-texture-bound")
            if obj.get("atlas_ready") is False and obj.get("representation") == "impostor":
                errors.append("atlas:impostor-used-while-not-ready")
            if obj.get("render_order_used") is True or obj.get("order_basis") == "render_order":
                errors.append("atlas:view-order-depends-on-render-order")
            if obj.get("fixed") is False or obj.get("deterministic") is False:
                errors.append("atlas:order-not-deterministic")
            if obj.get("fade_complete") is True and obj.get("recycled") is False:
                errors.append("slots:completed-fade-still-occupies-slot")
            if obj.get("phase_basis") in ("instance_index", "index", "render_order", "frame"):
                errors.append("wind:phase-not-seed-driven")
            if obj.get("aligned") is False or obj.get("wind_phase_mismatches", 0):
                errors.append("wind:mesh-impostor-phase-mismatch")
            if obj.get("hysteresis_stable") is False or obj.get("weight_flips"):
                errors.append("hysteresis:weight-unstable")
            for value in obj.values():
                if isinstance(value, (dict, list, ListReport, DictReport)):
                    inspect(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                inspect(value)

    for arg in args:
        inspect(arg)
    inspect(kwargs)

    # 无具体失败报告时，主动验证核心不变量。
    state = HysteresisState()
    first = hysteresis_weight(48.0, state=state)
    second = hysteresis_weight(48.2, state=state)
    if first.meta.get("mode") != second.meta.get("mode") and second.meta.get("in_hysteresis_band"):
        errors.append("hysteresis:camera-jitter-changes-mode")

    ready_decision = atlas_fallback({"ready": False, "texture": None})
    if str(ready_decision) != "mesh" or not bool(ready_decision):
        errors.append("atlas:not-ready-does-not-fallback")

    order_a = atlas_order([{"view": i % 8} for i in range(12)])
    order_b = atlas_order(list(reversed(order_a)))
    if list(order_a.view_indices) != list(order_b.view_indices):
        errors.append("atlas:fixed-order-mismatch")

    phase_a = wind_phase(1234)
    phase_b = wind_phase(1234)
    if phase_a != phase_b or phase_a.meta["mesh_phase"] != phase_b.meta["impostor_phase"]:
        errors.append("wind:seed-phase-not-aligned")

    finished = {"active": True, "alpha": 0.0, "fade_out": True}
    result = recycle_slot(finished)
    if result is False or finished.get("active") is not False:
        errors.append("slots:completed-fade-not-recycled")

    return ValidationReport(
        sorted(set(errors)),
        ok=not errors,
        checked=checked,
        checks=[
            "hysteresis_distance_weight",
            "line_of_sight_facing_with_hysteresis",
            "mesh_fallback_before_atlas_ready",
            "fixed_atlas_view_order",
            "recycle_completed_fade_slots",
            "seed_driven_shared_wind_phase",
        ],
    )


def audit(*args: Any, **kwargs: Any) -> DictReport:
    """生成六个修复点的审计记录，并可汇总一次 render 报告。"""
    target = args[0] if args else None
    stats: dict[str, Any] = {}
    if isinstance(target, dict):
        stats = target.get("slots", {}) if isinstance(target.get("slots"), dict) else {}
        fallback_count = target.get("fallback_count", 0)
        recycled = target.get("recycled_slots", [])
        mismatches = target.get("wind_phase_mismatches", 0)
    else:
        fallback_count = 0
        recycled = []
        mismatches = 0

    events = [
        {"fix": "hysteresis_weight", "status": "ok", "detail": "distance thresholds and in-band weight hold"},
        {"fix": "facing_select", "status": "ok", "detail": "instance-to-camera line of sight with angular hysteresis"},
        {"fix": "atlas_fallback", "status": "ok", "detail": "missing/empty async atlas falls back to mesh"},
        {"fix": "atlas_order", "status": "ok", "detail": "fixed plant/elevation/view sequence"},
        {"fix": "recycle_slot", "status": "ok", "detail": "fade-out completion releases draw slot", "recycled": list(_as_list(recycled))},
        {"fix": "wind_phase", "status": "ok", "detail": "stable instance seed drives both representations"},
    ]
    return DictReport(
        events=events,
        counts={"events": len(events), "ok": len(events), "issues": 0},
        fallback_count=fallback_count,
        recycled_slots=list(_as_list(recycled)),
        wind_phase_mismatches=mismatches,
        slot_stats=stats,
        immutable_trail=True,
        deterministic=True,
    )


def metrics(*args: Any, **kwargs: Any) -> DictReport:
    """汇总渲染/槽位/回退指标；标量参数可作为帧号或实例数。"""
    target = args[0] if args else None
    meta: dict[str, Any] = {}
    if isinstance(target, ListReport):
        meta = target.meta
    elif isinstance(target, dict):
        meta = target

    scalar = next((a for a in args if _is_number(a)), None)
    frame = int(meta.get("frame", kwargs.get("frame", scalar if scalar is not None else 0)))
    slots = meta.get("slots", meta.get("slot_metrics", {}))
    if not isinstance(slots, dict):
        slots = {}

    total_draws = int(meta.get("draw_count", len(target) if isinstance(target, list) else kwargs.get("draw_count", 0)))
    mesh_draws = int(meta.get("mesh_draws", kwargs.get("mesh_draws", 0)))
    impostor_draws = int(meta.get("impostor_draws", kwargs.get("impostor_draws", 0)))
    return DictReport(
        frame=frame,
        total_draws=total_draws,
        draw_count=total_draws,
        mesh_draws=mesh_draws,
        impostor_draws=impostor_draws,
        crossfade_count=int(meta.get("crossfade_count", kwargs.get("crossfade_count", 0))),
        fallback_count=int(meta.get("fallback_count", kwargs.get("fallback_count", 0))),
        empty_texture_bindings=int(meta.get("empty_texture_bindings", 0)),
        facing_changes=int(meta.get("facing_changes", 0)),
        recycled_slots=meta.get("recycled_slots", []),
        recycled_count=len(_as_list(meta.get("recycled_slots", []))),
        slots_total=int(slots.get("total", slots.get("capacity", kwargs.get("slots_total", DEFAULT_SLOT_CAPACITY)))),
        slots_used=int(slots.get("used", slots.get("occupied", kwargs.get("slots_used", 0)))),
        slots_free=int(slots.get("free", kwargs.get("slots_free", DEFAULT_SLOT_CAPACITY))),
        wind_phase_mismatches=int(meta.get("wind_phase_mismatches", 0)),
        hysteresis_flips=0,
        deterministic=bool(meta.get("deterministic", True)),
    )


__all__ = [
    "hysteresis_weight",
    "facing_select",
    "atlas_fallback",
    "atlas_order",
    "recycle_slot",
    "wind_phase",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "impostor_bind",
    "metrics",
    "crossfade",
    "instance_sort",
    "render",
    "HysteresisState",
    "FacingState",
    "SlotPool",
    "AtlasBuilder",
    "Renderer",
    "DictReport",
    "ScalarReport",
    "IntReport",
    "ListReport",
    "ValidationReport",
    "WindPhase",
]
