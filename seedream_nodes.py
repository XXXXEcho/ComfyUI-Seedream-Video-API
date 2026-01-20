"""
Seedance 视频生成节点 - 主节点模块
包含文生视频、图生视频、首尾帧图生视频、参考图生视频等节点
"""

import os
import time
import json
import folder_paths
import torch
from typing import Optional, Tuple, List, Any

try:
    from comfy_api.latest._input_impl.video_types import VideoFromFile
except ImportError:
    # 兼容性处理：如果没有 VideoFromFile，则使用替代方案
    class VideoFromFile:
        def __init__(self, *args, **kwargs):
            raise ImportError("VideoFromFile is not available in your ComfyUI version")

# 导入共享模块
from .seedream_shared import (
    # 模型配置
    VIDEO_MODEL_MAP,
    T2V_MODELS,
    I2V_MODELS,
    FIRST_LAST_FRAME_MODELS,
    REFERENCE_IMAGE_MODELS,
    DRAFT_MODELS,
    RESOLUTION_OPTIONS,
    ASPECT_RATIO_OPTIONS,
    TASK_STATUS_OPTIONS,
    SERVICE_TIER_OPTIONS,
    # 图片处理
    encode_image_to_base64,
    download_image_to_tensor,
    # API 调用
    create_video_generation_request,
    add_image_to_content,
    submit_video_generation_task,
    poll_video_task_completion,
    download_video_to_file,
    query_video_task,
    # 日志
    log_info,
    log_error,
    format_error,
)

# ====================================================================================
# 默认 API 配置
# ====================================================================================

DEFAULT_API_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
DEFAULT_QUERY_API_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"


# ====================================================================================
# 基础视频节点类
# ====================================================================================

class BaseSeedanceVideoNode:
    """视频生成节点的基础类，包含公共逻辑"""

    @classmethod
    def get_api_key(cls, api_key_input: str) -> str:
        """获取 API Key，优先使用输入值，否则使用环境变量"""
        if api_key_input and "在此输入" not in api_key_input and api_key_input.strip():
            return api_key_input.strip()
        env_key = os.getenv("ARK_API_KEY", "")
        if env_key:
            return env_key
        return ""

    @classmethod
    def validate_api_key(cls, api_key: str) -> Tuple[bool, str]:
        """验证 API Key"""
        if not api_key:
            return False, "API Key is missing. Please provide ARK_API_KEY in environment or input field."
        return True, ""

    @classmethod
    def save_video_output(cls, video_url: str, filename_prefix: str) -> Optional[str]:
        """保存视频到输出目录"""
        try:
            output_dir = folder_paths.get_output_directory()
            safe_prefix = filename_prefix.replace("/", "_").replace("\\", "_")

            # 生成文件名
            timestamp = int(time.time())
            filename = f"{safe_prefix}_{timestamp}.mp4"
            output_path = os.path.join(output_dir, filename)

            # 下载视频
            if download_video_to_file(video_url, output_path):
                return output_path
            return None
        except Exception as e:
            log_error(f"Failed to save video: {e}")
            return None

    @classmethod
    def handle_last_frame(cls, response_data: dict) -> torch.Tensor:
        """处理响应中的尾帧图片"""
        try:
            if response_data.get("return_last_frame"):
                # 尝试从响应中获取尾帧 URL
                last_frame_url = response_data.get("content", {}).get("last_frame_url")

                if last_frame_url:
                    result = download_image_to_tensor(last_frame_url)
                    if result is not None:
                        return result
            # 如果没有尾帧，返回一个空白占位图像（1x1 黑色）
            return torch.zeros(1, 64, 64, 3, dtype=torch.float32)
        except Exception as e:
            log_error(f"Failed to process last frame: {e}")
            return torch.zeros(1, 64, 64, 3, dtype=torch.float32)


# ====================================================================================
# 1. 文生视频节点
# ====================================================================================

class SeedanceText2Video(BaseSeedanceVideoNode):
    """
    Seedance 文生视频节点
    支持所有 Seedance 系列模型的文生视频功能
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": DEFAULT_API_URL
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("ARK_API_KEY", "在此输入你的火山引擎API Key")
                }),
                "model": (T2V_MODELS, {
                    "default": T2V_MODELS[0] if T2V_MODELS else "Seedance 1.5 Pro"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "小猫对着镜头打哈欠"
                }),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "ratio": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "duration": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 12
                }),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 4294967295
                }),
                "service_tier": (SERVICE_TIER_OPTIONS, {"default": "default"}),
                "timeout_seconds": ("INT", {
                    "default": 172800,
                    "min": 3600,
                    "max": 259200
                }),
                "poll_interval": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60
                }),
                "filename_prefix": ("STRING", {
                    "default": "Seedance/T2V"
                }),
            },
            "optional": {
                # Seedance 1.5 Pro 专属参数
                "generate_audio": ("BOOLEAN", {
                    "default": True
                }),
                "draft_mode": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Seedance/Video"
    OUTPUT_NODE = True

    def generate_video(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        resolution: str,
        ratio: str,
        duration: int,
        camera_fixed: bool,
        watermark: bool,
        seed: int,
        service_tier: str,
        timeout_seconds: int,
        poll_interval: int,
        filename_prefix: str,
        generate_audio: bool = False,
        draft_mode: bool = False,
    ) -> Tuple[Optional[VideoFromFile], Optional[torch.Tensor], str]:

        # 验证 API Key
        api_key = self.get_api_key(api_key)
        valid, error_msg = self.validate_api_key(api_key)
        if not valid:
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 获取模型 ID
        model_id = VIDEO_MODEL_MAP.get(model, model)

        # 构建请求 payload
        try:
            payload = create_video_generation_request(
                model=model_id,
                prompt=prompt,
                resolution=resolution,
                ratio=ratio,
                duration=duration,
                seed=seed,
                camera_fixed=camera_fixed,
                watermark=watermark,
                generate_audio=generate_audio,
                draft=draft_mode,
                return_last_frame=True,
                service_tier=service_tier,
                execution_expires_after=timeout_seconds,
            )
        except ValueError as e:
            error_msg = f"Invalid parameter: {str(e)}"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 提交任务
        log_info(f"Submitting text-to-video task with model: {model}")
        success, response, error = submit_video_generation_task(api_url, api_key, payload)

        if not success:
            log_error(f"Task submission failed: {error}")
            return None, None, json.dumps({"error": error}, ensure_ascii=False)

        task_id = response.get("id", "")
        if not task_id:
            error_msg = "No task ID in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": response}, ensure_ascii=False)

        log_info(f"Task submitted successfully. Task ID: {task_id}")

        # 轮询等待任务完成
        log_info("Polling for task completion...")
        success, final_response, error = poll_video_task_completion(
            api_url,
            api_key,
            task_id,
            poll_interval=poll_interval,
            max_polls=timeout_seconds // poll_interval,
        )

        if not success:
            log_error(f"Task failed: {error}")
            return None, None, json.dumps({"error": error, "task_id": task_id}, ensure_ascii=False)

        # 获取视频 URL
        content = final_response.get("content", {})
        video_url = content.get("video_url", "")

        if not video_url:
            error_msg = "No video URL in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": final_response}, ensure_ascii=False)

        # 保存视频
        video_path = self.save_video_output(video_url, filename_prefix)

        if not video_path:
            return None, None, json.dumps({"error": "Failed to save video", "response": final_response}, ensure_ascii=False)

        # 处理尾帧
        last_frame = self.handle_last_frame(final_response)

        log_info(f"Video generation completed: {video_path}")

        # 创建视频对象
        try:
            video = VideoFromFile(video_path)
        except Exception as e:
            log_error(f"Failed to create video object: {e}")
            video = None

        response_json = json.dumps(final_response, ensure_ascii=False, indent=2)
        return video, last_frame, response_json


# ====================================================================================
# 2. 图生视频节点（首帧）
# ====================================================================================

class SeedanceImage2Video(BaseSeedanceVideoNode):
    """
    Seedance 图生视频节点（首帧）
    支持所有图生视频模型，使用首帧图片生成视频
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": DEFAULT_API_URL
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("ARK_API_KEY", "在此输入你的火山引擎API Key")
                }),
                "model": (I2V_MODELS, {
                    "default": I2V_MODELS[0] if I2V_MODELS else "Seedance 1.5 Pro"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "image": ("IMAGE",),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "ratio": (ASPECT_RATIO_OPTIONS, {"default": "adaptive"}),
                "duration": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 12
                }),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 4294967295
                }),
                "service_tier": (SERVICE_TIER_OPTIONS, {"default": "default"}),
                "timeout_seconds": ("INT", {
                    "default": 172800,
                    "min": 3600,
                    "max": 259200
                }),
                "poll_interval": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60
                }),
                "filename_prefix": ("STRING", {
                    "default": "Seedance/I2V"
                }),
            },
            "optional": {
                "generate_audio": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Seedance/Video"
    OUTPUT_NODE = True

    def generate_video(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        image: torch.Tensor,
        resolution: str,
        ratio: str,
        duration: int,
        camera_fixed: bool,
        watermark: bool,
        seed: int,
        service_tier: str,
        timeout_seconds: int,
        poll_interval: int,
        filename_prefix: str,
        generate_audio: bool = False,
    ) -> Tuple[Optional[VideoFromFile], Optional[torch.Tensor], str]:

        print(f"[DEBUG] SeedanceImage2Video.generate_video called!")
        print(f"[DEBUG] api_url={api_url}")
        print(f"[DEBUG] model={model}")
        print(f"[DEBUG] prompt={prompt[:50] if prompt else 'empty'}...")
        print(f"[DEBUG] image shape={image.shape if image is not None else 'None'}")

        log_info("Seedance image-to-video node starting...")

        # 验证 API Key
        api_key = self.get_api_key(api_key)
        log_info(f"API Key configured: {bool(api_key)}")
        valid, error_msg = self.validate_api_key(api_key)
        if not valid:
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 获取模型 ID
        model_id = VIDEO_MODEL_MAP.get(model, model)

        # 构建请求 payload
        try:
            payload = create_video_generation_request(
                model=model_id,
                prompt=prompt,
                resolution=resolution,
                ratio=ratio,
                duration=duration,
                seed=seed,
                camera_fixed=camera_fixed,
                watermark=watermark,
                generate_audio=generate_audio,
                draft=False,
                return_last_frame=True,
                service_tier=service_tier,
                execution_expires_after=timeout_seconds,
            )

            # 添加首帧图片
            content = payload.get("content", [])
            if not add_image_to_content(content, image, "first_frame"):
                error_msg = "Failed to encode image"
                log_error(error_msg)
                return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

            payload["content"] = content

        except ValueError as e:
            error_msg = f"Invalid parameter: {str(e)}"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 提交任务
        log_info(f"Submitting image-to-video task with model: {model}")
        success, response, error = submit_video_generation_task(api_url, api_key, payload)

        if not success:
            log_error(f"Task submission failed: {error}")
            return None, None, json.dumps({"error": error}, ensure_ascii=False)

        task_id = response.get("id", "")
        if not task_id:
            error_msg = "No task ID in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": response}, ensure_ascii=False)

        log_info(f"Task submitted successfully. Task ID: {task_id}")

        # 轮询等待任务完成
        log_info("Polling for task completion...")
        success, final_response, error = poll_video_task_completion(
            api_url,
            api_key,
            task_id,
            poll_interval=poll_interval,
            max_polls=timeout_seconds // poll_interval,
        )

        if not success:
            log_error(f"Task failed: {error}")
            return None, None, json.dumps({"error": error, "task_id": task_id}, ensure_ascii=False)

        # 获取视频 URL
        content = final_response.get("content", {})
        video_url = content.get("video_url", "")

        if not video_url:
            error_msg = "No video URL in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": final_response}, ensure_ascii=False)

        # 保存视频
        video_path = self.save_video_output(video_url, filename_prefix)

        if not video_path:
            return None, None, json.dumps({"error": "Failed to save video", "response": final_response}, ensure_ascii=False)

        # 处理尾帧
        last_frame = self.handle_last_frame(final_response)

        log_info(f"Video generation completed: {video_path}")

        # 创建视频对象
        try:
            video = VideoFromFile(video_path)
        except Exception as e:
            log_error(f"Failed to create video object: {e}")
            video = None

        response_json = json.dumps(final_response, ensure_ascii=False, indent=2)
        return video, last_frame, response_json


# ====================================================================================
# 3. 首尾帧图生视频节点
# ====================================================================================

class SeedanceFirstLastFrame2Video(BaseSeedanceVideoNode):
    """
    Seedance 首尾帧图生视频节点
    支持使用首帧和尾帧图片生成过渡视频
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": DEFAULT_API_URL
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("ARK_API_KEY", "在此输入你的火山引擎API Key")
                }),
                "model": (FIRST_LAST_FRAME_MODELS, {
                    "default": FIRST_LAST_FRAME_MODELS[0] if FIRST_LAST_FRAME_MODELS else "Seedance 1.5 Pro"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "ratio": (ASPECT_RATIO_OPTIONS, {"default": "adaptive"}),
                "duration": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 12
                }),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 4294967295
                }),
                "service_tier": (SERVICE_TIER_OPTIONS, {"default": "default"}),
                "timeout_seconds": ("INT", {
                    "default": 172800,
                    "min": 3600,
                    "max": 259200
                }),
                "poll_interval": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60
                }),
                "filename_prefix": ("STRING", {
                    "default": "Seedance/FirstLast"
                }),
            },
            "optional": {
                "generate_audio": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Seedance/Video"
    OUTPUT_NODE = True

    def generate_video(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        resolution: str,
        ratio: str,
        duration: int,
        camera_fixed: bool,
        watermark: bool,
        seed: int,
        service_tier: str,
        timeout_seconds: int,
        poll_interval: int,
        filename_prefix: str,
        generate_audio: bool = False,
    ) -> Tuple[Optional[VideoFromFile], Optional[torch.Tensor], str]:

        # 验证 API Key
        api_key = self.get_api_key(api_key)
        valid, error_msg = self.validate_api_key(api_key)
        if not valid:
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 获取模型 ID
        model_id = VIDEO_MODEL_MAP.get(model, model)

        # 构建请求 payload
        try:
            payload = create_video_generation_request(
                model=model_id,
                prompt=prompt,
                resolution=resolution,
                ratio=ratio,
                duration=duration,
                seed=seed,
                camera_fixed=camera_fixed,
                watermark=watermark,
                generate_audio=generate_audio,
                draft=False,
                return_last_frame=True,
                service_tier=service_tier,
                execution_expires_after=timeout_seconds,
            )

            # 添加首帧和尾帧图片
            content = payload.get("content", [])
            if not add_image_to_content(content, first_frame, "first_frame"):
                error_msg = "Failed to encode first frame image"
                log_error(error_msg)
                return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

            if not add_image_to_content(content, last_frame, "last_frame"):
                error_msg = "Failed to encode last frame image"
                log_error(error_msg)
                return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

            payload["content"] = content

        except ValueError as e:
            error_msg = f"Invalid parameter: {str(e)}"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 提交任务
        log_info(f"Submitting first-last-frame video task with model: {model}")
        success, response, error = submit_video_generation_task(api_url, api_key, payload)

        if not success:
            log_error(f"Task submission failed: {error}")
            return None, None, json.dumps({"error": error}, ensure_ascii=False)

        task_id = response.get("id", "")
        if not task_id:
            error_msg = "No task ID in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": response}, ensure_ascii=False)

        log_info(f"Task submitted successfully. Task ID: {task_id}")

        # 轮询等待任务完成
        log_info("Polling for task completion...")
        success, final_response, error = poll_video_task_completion(
            api_url,
            api_key,
            task_id,
            poll_interval=poll_interval,
            max_polls=timeout_seconds // poll_interval,
        )

        if not success:
            log_error(f"Task failed: {error}")
            return None, None, json.dumps({"error": error, "task_id": task_id}, ensure_ascii=False)

        # 获取视频 URL
        content = final_response.get("content", {})
        video_url = content.get("video_url", "")

        if not video_url:
            error_msg = "No video URL in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": final_response}, ensure_ascii=False)

        # 保存视频
        video_path = self.save_video_output(video_url, filename_prefix)

        if not video_path:
            return None, None, json.dumps({"error": "Failed to save video", "response": final_response}, ensure_ascii=False)

        # 处理尾帧
        last_frame_output = self.handle_last_frame(final_response)

        log_info(f"Video generation completed: {video_path}")

        # 创建视频对象
        try:
            video = VideoFromFile(video_path)
        except Exception as e:
            log_error(f"Failed to create video object: {e}")
            video = None

        response_json = json.dumps(final_response, ensure_ascii=False, indent=2)
        return video, last_frame_output, response_json


# ====================================================================================
# 4. 参考图生视频节点
# ====================================================================================

class SeedanceReferenceImage2Video(BaseSeedanceVideoNode):
    """
    Seedance 参考图生视频节点
    支持 1-4 张参考图片生成视频（仅 Seedance 1.0 Lite I2V 支持）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": DEFAULT_API_URL
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("ARK_API_KEY", "在此输入你的火山引擎API Key")
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "[图1]戴着眼镜穿着蓝色T恤的男生和[图2]的柯基小狗，坐在草坪上，3D卡通风格"
                }),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "ratio": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "duration": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 12
                }),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 4294967295
                }),
                "service_tier": (SERVICE_TIER_OPTIONS, {"default": "default"}),
                "timeout_seconds": ("INT", {
                    "default": 172800,
                    "min": 3600,
                    "max": 259200
                }),
                "poll_interval": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60
                }),
                "filename_prefix": ("STRING", {
                    "default": "Seedance/RefImg"
                }),
            },
            "optional": {
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Seedance/Video"
    OUTPUT_NODE = True

    def generate_video(
        self,
        api_url: str,
        api_key: str,
        prompt: str,
        resolution: str,
        ratio: str,
        duration: int,
        watermark: bool,
        seed: int,
        service_tier: str,
        timeout_seconds: int,
        poll_interval: int,
        filename_prefix: str,
        ref_image_1: Optional[torch.Tensor] = None,
        ref_image_2: Optional[torch.Tensor] = None,
        ref_image_3: Optional[torch.Tensor] = None,
        ref_image_4: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[VideoFromFile], Optional[torch.Tensor], str]:

        # 验证 API Key
        api_key = self.get_api_key(api_key)
        valid, error_msg = self.validate_api_key(api_key)
        if not valid:
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 收集参考图
        ref_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        ref_images = [img for img in ref_images if img is not None]

        if len(ref_images) == 0:
            error_msg = "At least one reference image is required"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        if len(ref_images) > 4:
            error_msg = "Maximum 4 reference images are supported"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 获取模型 ID（参考图生视频仅支持 Seedance 1.0 Lite I2V）
        model_id = VIDEO_MODEL_MAP.get("Seedance 1.0 Lite I2V")

        # 构建请求 payload
        try:
            payload = create_video_generation_request(
                model=model_id,
                prompt=prompt,
                resolution=resolution,
                ratio=ratio,
                duration=duration,
                seed=seed,
                camera_fixed=False,  # 参考图模式不支持固定摄像头
                watermark=watermark,
                generate_audio=False,
                draft=False,
                return_last_frame=True,
                service_tier=service_tier,
                execution_expires_after=timeout_seconds,
            )

            # 添加参考图片
            content = payload.get("content", [])
            for img in ref_images:
                if not add_image_to_content(content, img, "reference_image"):
                    error_msg = "Failed to encode reference image"
                    log_error(error_msg)
                    return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

            payload["content"] = content

        except ValueError as e:
            error_msg = f"Invalid parameter: {str(e)}"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 提交任务
        log_info(f"Submitting reference image video task with {len(ref_images)} reference images")
        success, response, error = submit_video_generation_task(api_url, api_key, payload)

        if not success:
            log_error(f"Task submission failed: {error}")
            return None, None, json.dumps({"error": error}, ensure_ascii=False)

        task_id = response.get("id", "")
        if not task_id:
            error_msg = "No task ID in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": response}, ensure_ascii=False)

        log_info(f"Task submitted successfully. Task ID: {task_id}")

        # 轮询等待任务完成
        log_info("Polling for task completion...")
        success, final_response, error = poll_video_task_completion(
            api_url,
            api_key,
            task_id,
            poll_interval=poll_interval,
            max_polls=timeout_seconds // poll_interval,
        )

        if not success:
            log_error(f"Task failed: {error}")
            return None, None, json.dumps({"error": error, "task_id": task_id}, ensure_ascii=False)

        # 获取视频 URL
        content = final_response.get("content", {})
        video_url = content.get("video_url", "")

        if not video_url:
            error_msg = "No video URL in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": final_response}, ensure_ascii=False)

        # 保存视频
        video_path = self.save_video_output(video_url, filename_prefix)

        if not video_path:
            return None, None, json.dumps({"error": "Failed to save video", "response": final_response}, ensure_ascii=False)

        # 处理尾帧
        last_frame = self.handle_last_frame(final_response)

        log_info(f"Video generation completed: {video_path}")

        # 创建视频对象
        try:
            video = VideoFromFile(video_path)
        except Exception as e:
            log_error(f"Failed to create video object: {e}")
            video = None

        response_json = json.dumps(final_response, ensure_ascii=False, indent=2)
        return video, last_frame, response_json


# ====================================================================================
# 5. 草稿转正式视频节点
# ====================================================================================

class SeedanceDraft2Video(BaseSeedanceVideoNode):
    """
    Seedance 草稿转正式视频节点
    基于已生成的 Draft 视频任务 ID，生成高质量正式视频（仅 Seedance 1.5 Pro 支持）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": DEFAULT_API_URL
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("ARK_API_KEY", "在此输入你的火山引擎API Key")
                }),
                "draft_task_id": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "watermark": ("BOOLEAN", {"default": False}),
                "service_tier": (SERVICE_TIER_OPTIONS, {"default": "default"}),
                "timeout_seconds": ("INT", {
                    "default": 172800,
                    "min": 3600,
                    "max": 259200
                }),
                "poll_interval": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 60
                }),
                "filename_prefix": ("STRING", {
                    "default": "Seedance/Draft2Video"
                }),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "convert_draft"
    CATEGORY = "Seedance/Video"
    OUTPUT_NODE = True

    def convert_draft(
        self,
        api_url: str,
        api_key: str,
        draft_task_id: str,
        watermark: bool,
        service_tier: str,
        timeout_seconds: int,
        poll_interval: int,
        filename_prefix: str,
    ) -> Tuple[Optional[VideoFromFile], Optional[torch.Tensor], str]:

        # 验证 API Key
        api_key = self.get_api_key(api_key)
        valid, error_msg = self.validate_api_key(api_key)
        if not valid:
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 验证草稿任务 ID
        if not draft_task_id or not draft_task_id.strip():
            error_msg = "Draft task ID is required"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg}, ensure_ascii=False)

        # 构建请求 payload
        # 使用 draft_task content 类型
        payload = {
            "model": VIDEO_MODEL_MAP.get("Seedance 1.5 Pro"),
            "content": [
                {
                    "type": "draft_task",
                    "draft_task": {
                        "id": draft_task_id.strip()
                    }
                }
            ],
            "watermark": watermark,
            "return_last_frame": True,
            "service_tier": service_tier,
            "execution_expires_after": timeout_seconds,
        }

        # 提交任务
        log_info(f"Converting draft task {draft_task_id[:12]}... to final video")
        success, response, error = submit_video_generation_task(api_url, api_key, payload)

        if not success:
            log_error(f"Task submission failed: {error}")
            return None, None, json.dumps({"error": error}, ensure_ascii=False)

        task_id = response.get("id", "")
        if not task_id:
            error_msg = "No task ID in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": response}, ensure_ascii=False)

        log_info(f"Task submitted successfully. Task ID: {task_id}")

        # 轮询等待任务完成
        log_info("Polling for task completion...")
        success, final_response, error = poll_video_task_completion(
            api_url,
            api_key,
            task_id,
            poll_interval=poll_interval,
            max_polls=timeout_seconds // poll_interval,
        )

        if not success:
            log_error(f"Task failed: {error}")
            return None, None, json.dumps({"error": error, "task_id": task_id}, ensure_ascii=False)

        # 获取视频 URL
        content = final_response.get("content", {})
        video_url = content.get("video_url", "")

        if not video_url:
            error_msg = "No video URL in response"
            log_error(error_msg)
            return None, None, json.dumps({"error": error_msg, "response": final_response}, ensure_ascii=False)

        # 保存视频
        video_path = self.save_video_output(video_url, filename_prefix)

        if not video_path:
            return None, None, json.dumps({"error": "Failed to save video", "response": final_response}, ensure_ascii=False)

        # 处理尾帧
        last_frame = self.handle_last_frame(final_response)

        log_info(f"Draft to video conversion completed: {video_path}")

        # 创建视频对象
        try:
            video = VideoFromFile(video_path)
        except Exception as e:
            log_error(f"Failed to create video object: {e}")
            video = None

        response_json = json.dumps(final_response, ensure_ascii=False, indent=2)
        return video, last_frame, response_json


# ====================================================================================
# 6. 视频任务查询节点
# ====================================================================================

class SeedanceVideoQuery:
    """
    Seedance 视频任务查询节点
    查询视频生成任务的状态和结果
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": DEFAULT_QUERY_API_URL
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("ARK_API_KEY", "在此输入你的火山引擎API Key")
                }),
                "task_id": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("status", "response_json", "video_url", "video")
    FUNCTION = "query_task"
    CATEGORY = "Seedance/Video"
    OUTPUT_NODE = True

    def query_task(
        self,
        api_url: str,
        api_key: str,
        task_id: str,
    ) -> Tuple[str, str, str, Optional[VideoFromFile]]:

        # 验证 API Key
        api_key = api_key.strip() if api_key else ""
        if not api_key or "在此输入" in api_key:
            api_key = os.getenv("ARK_API_KEY", "")

        if not api_key:
            error_msg = "API Key is missing"
            log_error(error_msg)
            return "error", json.dumps({"error": error_msg}, ensure_ascii=False), "", None

        # 验证任务 ID
        if not task_id or not task_id.strip():
            error_msg = "Task ID is required"
            log_error(error_msg)
            return "error", json.dumps({"error": error_msg}, ensure_ascii=False), "", None

        # 查询任务
        log_info(f"Querying task {task_id[:12]}...")

        success, response, error = query_video_task(api_url, api_key, task_id.strip())

        if not success:
            log_error(f"Query failed: {error}")
            return "error", json.dumps({"error": error}, ensure_ascii=False), "", None

        # 获取状态
        status = response.get("status", "unknown")

        # 获取视频 URL
        video_url = ""
        if status == "succeeded":
            content = response.get("content", {})
            video_url = content.get("video_url", "")

        response_json = json.dumps(response, ensure_ascii=False, indent=2)

        # 如果有视频 URL，尝试创建视频对象
        video = None
        if video_url:
            try:
                # 下载并保存视频
                output_dir = folder_paths.get_output_directory()
                timestamp = int(time.time())
                filename = f"Seedance_Query_{timestamp}.mp4"
                output_path = os.path.join(output_dir, filename)

                if download_video_to_file(video_url, output_path):
                    video = VideoFromFile(output_path)
            except Exception as e:
                log_error(f"Failed to load video: {e}")

        log_info(f"Task status: {status}")

        return status, response_json, video_url, video


# ====================================================================================
# 节点映射
# ====================================================================================

NODE_CLASS_MAPPINGS = {
    "SeedanceText2Video": SeedanceText2Video,
    "SeedanceImage2Video": SeedanceImage2Video,
    "SeedanceFirstLastFrame2Video": SeedanceFirstLastFrame2Video,
    "SeedanceReferenceImage2Video": SeedanceReferenceImage2Video,
    "SeedanceDraft2Video": SeedanceDraft2Video,
    "SeedanceVideoQuery": SeedanceVideoQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedanceText2Video": "Seedance 文生视频 (Text2Video)",
    "SeedanceImage2Video": "Seedance 图生视频-首帧 (Image2Video)",
    "SeedanceFirstLastFrame2Video": "Seedance 首尾帧图生视频 (FirstLastFrame2Video)",
    "SeedanceReferenceImage2Video": "Seedance 参考图生视频 (ReferenceImage2Video)",
    "SeedanceDraft2Video": "Seedance 草稿转正式视频 (Draft2Video)",
    "SeedanceVideoQuery": "Seedance 视频任务查询 (VideoQuery)",
}
