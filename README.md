# pairwise-foliage-impostor-crossfade

## 领域

游戏、图形与多媒体

## 主题

植被 impostor 交叉淡化

## 说明

滞回权重、视角图集顺序与风摆相位统一。

初始状态下 `graphics/foliage_impostor_crossfade.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
