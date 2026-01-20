# ComfyUI-Seedream-Video-API

ComfyUI 火山引擎 Seedream 视频生成 API 节点插件

## 功能特性

支持 6 个视频生成节点，涵盖文生视频、图生视频、首尾帧视频、参考图视频、草稿转换和任务查询。

---

## 节点列表

| 节点名称 | 功能描述 | 支持模型 |
|---------|---------|---------|
| **Seedream 文生视频** | 基于文本提示词生成视频 | Seedance 1.5/1.0 Pro/Lite |
| **Seedream 图生视频-首帧** | 使用首帧图片生成视频 | Seedance 1.5/1.0 Pro/Lite I2V |
| **Seedream 首尾帧图生视频** | 使用首尾帧生成过渡视频 | Seedance 1.5/1.0 Pro/Lite I2V |
| **Seedream 参考图生视频** | 使用1-4张参考图片生成视频 | Seedance 1.0 Lite I2V |
| **Seedream 草稿转正式视频** | 基于Draft任务ID生成高质量视频 | Seedance 1.5 Pro |
| **Seedream 视频任务查询** | 查询视频生成任务状态和结果 | 所有模型 |

---

## 支持的模型

| 模型名称 | 模型ID | 特性 |
|---------|-------|------|
| Seedance 1.5 Pro | doubao-seedance-1-5-pro-251215 | 最新模型，支持音频生成、草稿模式 |
| Seedance 1.0 Pro | doubao-seedance-1-0-pro-250527 | 专业视频模型 |
| Seedance 1.0 Pro Fast | doubao-seedance-1-0-pro-fast-250527 | 快速视频模型 |
| Seedance 1.0 Lite T2V | doubao-seedance-1-0-lite-t2v-250527 | 轻量级文生视频 |
| Seedance 1.0 Lite I2V | doubao-seedance-1-0-lite-i2v-250527 | 轻量级图生视频 |

---

## 参数说明

### 基础参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `api_url` | STRING | API 请求地址 |
| `api_key` | STRING | 火山引擎 API Key |
| `prompt` | STRING | 文本提示词 |
| `model` | 选择 | 视频生成模型 |
| `resolution` | 选择 | 视频分辨率 (480p/720p/1080p) |
| `ratio` | 选择 | 视频宽高比 (16:9, 4:3, 1:1, 9:16 等) |
| `duration` | 整数 | 视频时长 (2-12秒) |
| `seed` | 整数 | 随机种子 (-1为随机) |
| `camera_fixed` | 布尔 | 是否固定摄像头视角 |
| `watermark` | 布尔 | 是否添加水印 |
| `service_tier` | 选择 | 服务等级 (default/flex) |

### Seedance 1.5 Pro 专属参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `generate_audio` | 布尔 | 是否生成同步音频 |
| `draft_mode` | 布尔 | 是否生成草稿预览视频 |

### 任务控制参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `timeout_seconds` | 整数 | 任务超时时间（默认172800秒） |
| `poll_interval` | 整数 | 轮询间隔（默认2秒） |
| `filename_prefix` | 文本 | 输出文件名前缀 |

---

## 输出说明

| 输出 | 类型 | 说明 |
|------|------|------|
| `video` | VIDEO | 生成的视频文件 |
| `last_frame` | IMAGE | 视频尾帧图片 |
| `response` | STRING | API 完整响应 JSON |

---

## 使用示例

### 文生视频
```
输入: prompt = "小猫对着镜头打哈欠"
输出: 5秒视频 (720p, 16:9)
```

### 图生视频
```
输入: 首帧图片 + prompt = "镜头缓慢推进"
输出: 5秒视频，从首帧开始生成
```

### 首尾帧视频
```
输入: 首帧图片 + 尾帧图片 + prompt = "平滑过渡"
输出: 5秒视频，从首帧过渡到尾帧
```

### 参考图视频
```
输入: 2张参考图 + prompt = "[图1]的人物和[图2]的背景"
输出: 5秒视频，融合参考图元素
```

---

## 常见问题

**Q: 视频生成时间很长？**
- 使用 `Seedance 1.0 Pro Fast` 模型
- 启用 `draft_mode` 草稿模式
- 降低分辨率到 480p
- 缩短视频时长

**Q: 如何生成带音频的视频？**
- 使用 `Seedance 1.5 Pro` 模型
- 启用 `generate_audio` 参数

**Q: 草稿模式是什么？**
- 快速生成低质量预览视频
- 使用 `Seedream 草稿转正式视频` 节点生成高质量版本

**Q: 任务超时怎么办？**
- 增加 `timeout_seconds` 参数
- 默认为 172800 秒（48小时）

---

## 文件结构

```
ComfyUI-Seedrance-API-TUTU/
├── __init__.py           # 模块初始化
├── seedream_shared.py    # 共享工具函数
├── seedream_nodes.py     # 视频节点定义
└── README.md            # 本文档
```

---

## API 文档

- [火山引擎视频生成 API](https://www.volcengine.com/docs/82379)

---

## 许可证

MIT License
