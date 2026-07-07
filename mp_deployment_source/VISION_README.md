# 2021 电赛送药小车 K230 视觉程序

这个目录已经整理成 K230 可部署结构：

- `main.py`：比赛用视觉主程序，识别 1-8 号药房，做多帧稳定锁定，并通过串口输出给下位机。
- 同时集成底部 ROI 循迹，输出红线中心偏差和角度。
- `deploy_config.json`：嘉立创/K230 云平台导出的模型配置。
- `best_AnchorBaseDet_can2_5_s_20260625195349.kmodel`：药房数字检测模型。

## 部署

把整个 `mp_deployment_source` 文件夹放到 K230 的 `/sdcard/` 下，最终路径应为：

```text
/sdcard/mp_deployment_source/main.py
/sdcard/mp_deployment_source/deploy_config.json
/sdcard/mp_deployment_source/best_AnchorBaseDet_can2_5_s_20260625195349.kmodel
```

在 CanMV IDE 里运行 `/sdcard/mp_deployment_source/main.py`。如果要上电自启，可把这个文件复制成开发板启动位置要求的 `main.py`，或在启动脚本里 `execfile("/sdcard/mp_deployment_source/main.py")`。

## 关键调参

都在 `main.py` 顶部：

- `DISPLAY_MODE = "lcd"`：嘉立创 K230 小屏一般用 `lcd`，HDMI 用 `hdmi` 或 `lt9611`。
- `RGB888P_SIZE = [1280, 720]`：识别稳优先用这个；如果帧率不够，改 `[640, 360]`。
- `DEFAULT_TARGET_ID = 0`：0 表示自动选择画面里最靠谱的药房；改成 1-8 可固定目标。
- `SELECT_MIN_SCORE`：检测阈值，误检多就升到 `0.50` 左右，漏检多就降到 `0.38` 左右。
- `CENTER_TOLERANCE_RATIO`：横向居中容忍区，车头抖就适当加大。
- `ARRIVE_BOX_HEIGHT_RATIO`：判定到达药房的框高比例，越大越靠近才停车。
- `DISTANCE_K_BY_HEIGHT`：粗略距离估计系数，需要现场标定。
- `ENABLE_LINE_TRACK`：是否启用循迹。
- `LINE_THRESHOLDS`：循迹颜色阈值，默认是红线。
- `LINE_USE_RAW_RGB_SCAN`：默认开启，直接扫描 AI 图像里的 RGB 红色像素，不依赖 `find_blobs`。
- `LINE_RGB565_SIZE = [640, 360]`：只有关闭 RAW 扫描、改用 RGB565 通道时才使用。
- `LINE_ROI_BANDS`：底部循迹区域和权重，越靠下权重越大。

## 下位机串口协议

默认 `UART2 / 115200`。如果你的接线需要指定引脚，在 `main.py` 顶部设置：

```python
UART_TX_PIN = 11
UART_RX_PIN = 12
```

每帧输出：

```text
$MV,frame,state,target,id,score,cx,cy,w,h,err_x,err_y,area,dist,hits,fps*CS
$LN,frame,seen,err_x,angle,cx,width,area,fps*CS
```

字段含义：

- `state=0`：没看到目标。
- `state=1`：看到候选，但还没稳定。
- `state=2`：稳定锁定，车可以闭环对准。
- `state=3`：稳定、居中、足够近，车可以停车/投药。
- `target`：当前目标号，0 是自动。
- `id`：识别出的药房号。
- `score`：置信度乘以 1000。
- `cx, cy, w, h`：目标框中心和宽高，坐标基于 `RGB888P_SIZE`。
- `err_x`：目标中心相对画面中心的横向误差，负数在左，正数在右。
- `dist`：粗略距离毫米值，未标定前只作参考。
- `hits`：最近多帧里同一目标命中的次数。
- `CS`：`$` 和 `*` 之间所有字符的 XOR 校验，十六进制两位。

循迹包字段：

- `seen=0`：没有看到线。
- `seen=1`：看到线。
- `err_x`：线中心相对画面中心的横向误差，负数在左，正数在右。
- `angle`：线方向角，负数偏左，正数偏右。
- `cx`：融合后的线中心 x 坐标，默认基于 `RGB888P_SIZE`。
- `width`：检测到的最大线宽。
- `area`：多个 ROI 红线面积总和。

下位机推荐控制逻辑：

```text
state 0: 原地慢扫或继续巡线找目标
state 1: 根据 err_x 小幅修正车头
state 2: 根据 err_x 闭环对准，同时低速靠近
state 3: 停车，执行送药动作
```

巡线推荐控制逻辑：

```text
LN seen 0: 降速，按上一帧方向短暂找线
LN seen 1: 使用 line_err_x 做主闭环，line_angle 做微分/前瞻修正
接近药房时: 以 MV 包为主，LN 包只做保底航向
```

一个简单的转向量可以这样算：

```text
turn = Kp * line_err_x + Ka * line_angle
```

`Kp` 先从 `0.4` 附近试，`Ka` 先从 `2.0` 附近试，具体要看你的电机 PWM 和底盘。

## 上位机给 K230 设置目标

下位机可发一行命令，以 `\n` 结尾：

```text
T3
T7
T0
```

`T1` 到 `T8` 固定目标药房，`T0` 回到自动模式。程序也兼容 `TARGET,3` 这种格式。

## 模型类别顺序

这个模型的类别顺序不是 `1,2,3,4,5,6,7,8`，而是：

```text
7, 8, 6, 5, 1, 2, 3, 4
```

程序会按 `deploy_config.json` 自动映射，所以下位机收到的 `id` 是真实药房号，不是模型类别索引。

## 现场建议

先不要接小车电机，只接 K230 屏幕/IDE 看框和串口输出。确认药房数字能稳定框住以后，再让下位机用 `err_x` 做舵机或差速闭环。比赛环境光变化大时，优先调 `SELECT_MIN_SCORE` 和摄像头角度，不要先改模型。

循迹调试建议：

1. 先让画面底部看到红线，屏幕上会出现绿色 ROI、绿色线中心点和 `LINE ex/ang`。
2. 如果一直 `LINE LOST`，先把 `LINE_RED_MIN` 从 `90` 降到 `70`，再把 `LINE_RED_DOMINANCE` 从 `25` 降到 `15`。
3. 如果把橙色/粉色干扰也识别成线，把 `LINE_RED_DOMINANCE` 提到 `35`，或者提高 `LINE_MIN_PIXELS`。
4. 如果帧率不够，把 `RGB888P_SIZE` 改成 `[640, 360]`。
