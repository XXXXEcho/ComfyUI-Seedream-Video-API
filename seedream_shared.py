"""
Seedance 视频生成节点 - 共享模块
包含工具函数、模型配置等
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import time
import requests
import os
from typing import Optional, List, Tuple, Any

# ====================================================================================
# 模型配置
# ====================================================================================

# Seedance 1.5 Pro 模型
SEEDANCE_1_5_PRO = "doubao-seedance-1-5-pro-251215"

# Seedance 1.0 Pro 模型
SEEDANCE_1_0_PRO = "doubao-seedance-1-0-pro-250528"
SEEDANCE_1_0_PRO_FAST = "doubao-seedance-1-0-pro-fast-251015"

# Seedance 1.0 Lite 模型
SEEDANCE_1_0_LITE_T2V = "doubao-seedance-1-0-lite-t2v-250428"
SEEDANCE_1_0_LITE_I2V = "doubao-seedance-1-0-lite-i2v-250428"

# 模型映射表
VIDEO_MODEL_MAP = {
    "Seedance 1.5 Pro": SEEDANCE_1_5_PRO,
    "Seedance 1.0 Pro": SEEDANCE_1_0_PRO,
    "Seedance 1.0 Pro Fast": SEEDANCE_1_0_PRO_FAST,
    "Seedance 1.0 Lite T2V": SEEDANCE_1_0_LITE_T2V,
    "Seedance 1.0 Lite I2V": SEEDANCE_1_0_LITE_I2V,
}

# 文生视频支持的模型
T2V_MODELS = ["Seedance 1.5 Pro", "Seedance 1.0 Pro", "Seedance 1.0 Pro Fast", "Seedance 1.0 Lite T2V"]

# 图生视频（首帧）支持的模型
I2V_MODELS = ["Seedance 1.5 Pro", "Seedance 1.0 Pro", "Seedance 1.0 Pro Fast", "Seedance 1.0 Lite I2V"]

# 首尾帧图生视频支持的模型
FIRST_LAST_FRAME_MODELS = ["Seedance 1.5 Pro", "Seedance 1.0 Pro", "Seedance 1.0 Lite I2V"]

# 参考图生视频支持的模型
REFERENCE_IMAGE_MODELS = ["Seedance 1.0 Lite I2V"]

# 草稿转正式视频支持的模型
DRAFT_MODELS = ["Seedance 1.5 Pro"]

# 分辨率选项
RESOLUTION_OPTIONS = ["480p", "720p", "1080p"]

# 宽高比选项
ASPECT_RATIO_OPTIONS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]

# 任务状态选项
TASK_STATUS_OPTIONS = ["all", "succeeded", "failed", "running", "queued", "cancelled", "expired"]

# 服务等级选项
SERVICE_TIER_OPTIONS = ["default", "flex"]

# ====================================================================================
# 图片处理工具函数
# ====================================================================================

def encode_image_to_base64(image_tensor: torch.Tensor) -> Optional[str]:
    """
    将 ComfyUI 的 Tensor 格式图片编码为 API 要求的 Base64 字符串

    Args:
        image_tensor: ComfyUI IMAGE 类型的 tensor

    Returns:
        Base64 编码的图片字符串 (data:image/jpeg;base64,...)
    """
    try:
        # 处理 batch 维度
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]

        # 转换为 numpy 并缩放到 0-255
        i = 255.0 * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # 转换为字节
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='JPEG', quality=95)
        byte_arr = byte_arr.getvalue()

        # Base64 编码
        base64_bytes = base64.b64encode(byte_arr)
        base64_string = base64_bytes.decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"
    except Exception as e:
        print(f"[Seedance] ERROR: Image encoding to Base64 failed: {e}")
        return None


def process_image_to_tensor(image_bytes: bytes) -> Optional[torch.Tensor]:
    """
    将二进制图片数据转换为 ComfyUI 的 Tensor 格式

    Args:
        image_bytes: 图片二进制数据

    Returns:
        ComfyUI IMAGE 类型的 tensor
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(np_image)[None,]
    except Exception as e:
        print(f"[Seedance] ERROR: Failed to process image bytes into tensor: {e}")
        return None


def decode_base64_to_tensor(base64_string: str) -> Optional[torch.Tensor]:
    """
    将 API 返回的 Base64 字符串解码为 ComfyUI 的 Tensor 格式图片

    Args:
        base64_string: Base64 编码的图片字符串

    Returns:
        ComfyUI IMAGE 类型的 tensor
    """
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        img_data = base64.b64decode(base64_string)
        return process_image_to_tensor(img_data)
    except Exception as e:
        print(f"[Seedance] ERROR: Image decoding from Base64 failed: {e}")
        return None


def download_image_to_tensor(url: str, timeout: int = 60) -> Optional[torch.Tensor]:
    """
    从 URL 下载图片并将其转换为 ComfyUI 的 Tensor 格式

    Args:
        url: 图片 URL
        timeout: 请求超时时间

    Returns:
        ComfyUI IMAGE 类型的 tensor
    """
    try:
        # 禁用 SSL 验证以解决 SSL 连接问题
        response = requests.get(url, timeout=timeout, verify=False)
        response.raise_for_status()
        return process_image_to_tensor(response.content)
    except requests.exceptions.RequestException as e:
        print(f"[Seedance] ERROR: Failed to download image from URL {url}: {e}")
        return None


# ====================================================================================
# API 调用工具函数
# ====================================================================================

def create_video_generation_request(
    model: str,
    prompt: str,
    resolution: str = "720p",
    ratio: str = "16:9",
    duration: int = 5,
    seed: int = -1,
    camera_fixed: bool = False,
    watermark: bool = False,
    generate_audio: bool = False,
    draft: bool = False,
    return_last_frame: bool = True,
    service_tier: str = "default",
    execution_expires_after: int = 172800,
) -> dict:
    """
    创建视频生成请求的 payload

    Args:
        model: 模型 ID
        prompt: 文本提示词
        resolution: 分辨率 (480p/720p/1080p)
        ratio: 宽高比
        duration: 视频时长（秒）
        seed: 随机种子
        camera_fixed: 是否固定摄像头
        watermark: 是否添加水印
        generate_audio: 是否生成音频（仅 1.5 pro 支持）
        draft: 是否为草稿模式（仅 1.5 pro 支持）
        return_last_frame: 是否返回尾帧
        service_tier: 服务等级 (default/flex)
        execution_expires_after: 任务超时时间（秒）

    Returns:
        API 请求的 payload 字典
    """
    content = [{"type": "text", "text": prompt}]

    payload = {
        "model": model,
        "content": content,
        "resolution": resolution,
        "ratio": ratio,
        "duration": duration,
        "seed": seed,
        "camera_fixed": camera_fixed,
        "watermark": watermark,
        "return_last_frame": return_last_frame,
        "service_tier": service_tier,
        "execution_expires_after": execution_expires_after,
    }

    # 仅 Seedance 1.5 Pro 支持的参数
    if "1-5-pro" in model:
        payload["generate_audio"] = generate_audio
        payload["draft"] = draft

    return payload


def add_image_to_content(content: list, image_tensor: torch.Tensor, role: str) -> bool:
    """
    将图片添加到请求的 content 中

    Args:
        content: API 请求的 content 列表
        image_tensor: 图片 tensor
        role: 图片角色 (first_frame/last_frame/reference_image)

    Returns:
        是否成功添加
    """
    if image_tensor is None:
        return False

    base64_data = encode_image_to_base64(image_tensor)
    if not base64_data:
        return False

    content.append({
        "type": "image_url",
        "image_url": {"url": base64_data},
        "role": role,
    })
    return True


def submit_video_generation_task(api_url: str, api_key: str, payload: dict, timeout: int = 180) -> Tuple[bool, Any, str]:
    """
    提交视频生成任务

    Args:
        api_url: API 地址
        api_key: API Key
        payload: 请求 payload
        timeout: 请求超时时间

    Returns:
        (成功状态, 响应数据, 错误信息)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 调试输出
    print(f"[DEBUG] Sending request to: {api_url}")
    print(f"[DEBUG] Request payload keys: {list(payload.keys())}")
    print(f"[DEBUG] Model: {payload.get('model')}")
    print(f"[DEBUG] Content: {payload.get('content')}")
    print(f"[DEBUG] Resolution: {payload.get('resolution')}, Ratio: {payload.get('ratio')}, Duration: {payload.get('duration')}")

    try:
        # 禁用 SSL 验证以解决 SSL 连接问题
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=timeout,
            verify=False  # 禁用 SSL 验证
        )
        response.raise_for_status()
        return True, response.json(), ""
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = json.dumps(error_detail, ensure_ascii=False, indent=2)
            except:
                error_msg = f"Status {e.response.status_code}: {e.response.text}"
        return False, None, error_msg


def query_video_task(api_url: str, api_key: str, task_id: str, timeout: int = 30) -> Tuple[bool, Any, str]:
    """
    查询视频生成任务状态

    Args:
        api_url: API 地址
        api_key: API Key
        task_id: 任务 ID
        timeout: 请求超时时间

    Returns:
        (成功状态, 响应数据, 错误信息)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # API URL 已经包含 /tasks，直接拼接 task_id
    query_url = f"{api_url.rstrip('/')}/{task_id}"

    try:
        # 禁用 SSL 验证以解决 SSL 连接问题
        response = requests.get(
            query_url,
            headers=headers,
            timeout=timeout,
            verify=False  # 禁用 SSL 验证
        )
        response.raise_for_status()
        return True, response.json(), ""
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = json.dumps(error_detail, ensure_ascii=False, indent=2)
            except:
                error_msg = f"Status {e.response.status_code}: {e.response.text}"
        return False, None, error_msg


def poll_video_task_completion(
    api_url: str,
    api_key: str,
    task_id: str,
    poll_interval: int = 2,
    max_polls: int = 3600,
    timeout: int = 30
) -> Tuple[bool, Any, str]:
    """
    轮询等待视频任务完成

    Args:
        api_url: API 地址
        api_key: API Key
        task_id: 任务 ID
        poll_interval: 轮询间隔（秒）
        max_polls: 最大轮询次数
        timeout: 每次请求超时时间

    Returns:
        (成功状态, 最终响应数据, 错误信息)
    """
    try:
        for i in range(max_polls):
            success, response, error = query_video_task(api_url, api_key, task_id, timeout)

            if not success:
                print()  # 换行
                return False, None, error

            status = response.get("status", "")

            if status == "succeeded":
                print()  # 换行
                return True, response, ""
            elif status in ["failed", "cancelled", "expired"]:
                print()  # 换行
                error_info = response.get("error", {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", f"Task {status}")
                else:
                    error_msg = str(error_info) if error_info else f"Task {status}"
                return False, response, error_msg

            # 打印轮询进度
            elapsed = i * poll_interval
            print(f"[Seedance] Polling task {task_id[:12]}... Status: {status}, Elapsed: {elapsed}s", end="\r")

            if i < max_polls - 1:
                time.sleep(poll_interval)

        print()  # 换行
        return False, None, f"Task timeout after {max_polls * poll_interval} seconds"
    except Exception as e:
        print()  # 换行
        return False, None, f"Polling error: {str(e)}"


# ====================================================================================
# 视频下载工具函数
# ====================================================================================

def download_video_to_file(video_url: str, output_path: str, timeout: int = 300) -> bool:
    """
    下载视频到本地文件

    Args:
        video_url: 视频 URL
        output_path: 输出文件路径
        timeout: 下载超时时间

    Returns:
        是否成功下载
    """
    try:
        # 创建目录（如果存在）
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # 禁用 SSL 验证以解决 SSL 连接问题
        response = requests.get(video_url, timeout=timeout, stream=True, verify=False)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"[Seedance] Video saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[Seedance] ERROR: Failed to download video: {e}")
        return False


# ====================================================================================
# 日志和错误处理
# ====================================================================================

def log_info(message: str):
    """打印信息日志"""
    print(f"[Seedance] {message}")


def log_error(message: str):
    """打印错误日志"""
    print(f"[Seedance] ERROR: {message}")


def format_error(e: Exception) -> str:
    """格式化异常信息"""
    if isinstance(e, requests.exceptions.RequestException):
        return f"Request failed: {str(e)}"
    return str(e)
