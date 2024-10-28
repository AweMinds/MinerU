import concurrent
import os
import json
import threading
import time
import uuid

import aiofiles
import torch
import base64
import filetype
import asyncio
from typing import Literal, Dict, List, Optional
from loguru import logger
import litserve as ls
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, Form, UploadFile
from unittest.mock import patch
from collections import defaultdict
from fastapi.responses import FileResponse  # 添加导入以支持文件下载
import shutil  # 确保shutil已导入以处理文件删除

from starlette.concurrency import run_in_threadpool

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton

import magic_pdf.model as model_config
from fastapi import BackgroundTasks

model_config.__use_inside_model__ = True


class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskQueue:
    MAX_QUEUE_SIZE = 5  # Maximum number of tasks in the queue

    def __init__(self):
        self.queue = asyncio.Queue()
        self.pending_tasks = {}  # task_id -> task_info
        self.lock = asyncio.Lock()

    async def add_task(self, task_id: str, task_info: dict):
        """添加任务到队列"""
        async with self.lock:
            self.pending_tasks[task_id] = task_info
            await self.queue.put(task_id)

    async def get_task(self) -> Optional[str]:
        """获取下一个待处理的任务"""
        try:
            task_id = await self.queue.get()
            return task_id
        except asyncio.QueueEmpty:
            return None

    async def remove_task(self, task_id: str):
        """从队列中移除任务"""
        async with self.lock:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]

    @property
    def queue_size(self) -> int:
        """获取队列中的任务数"""
        return len(self.pending_tasks)

    def get_queue_info(self) -> dict:
        """获取队列状态信息"""
        return {
            'queue_size': self.queue_size,
            'pending_tasks': list(self.pending_tasks.keys())
        }


class MinerUService:
    def __init__(self, output_base_dir='/tmp/minerU', gpu_id=0):
        # First assign gpu_id
        self.gpu_id = gpu_id
        # Then use it to set environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        # Rest of initialization
        self.output_base_dir = output_base_dir
        os.makedirs(output_base_dir, exist_ok=True)
        self.tasks: Dict[str, dict] = {}
        self.task_queue = TaskQueue()
        self.processor_tasks = []  # List to hold multiple processor tasks
        self.models_initialized = False
        self.task_count = 0  # Track the number of tasks processed
        # 添加一个用于保存结果文件的目录
        self.results_dir = os.path.join(self.output_base_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.start_server()
        # loop = asyncio.get_event_loop()
        # self.start_processor(loop)

    def start_server(self):
        """
        启动服务器，使用指定的GPU
        Args:
            gpu_id: 要使用的GPU编号
        """

        logger.info("Starting MinerU server...")

        # 检查指定的GPU是否可用
        if torch.cuda.is_available() and self.gpu_id < torch.cuda.device_count():
            logger.info(f"Using GPU {self.gpu_id}")
        else:
            logger.warning(f"GPU {self.gpu_id} is not available. Running in CPU mode.")
            # gpu_id = 0  # 使用CPU模式

        # 初始化服务
        # service = MinerUService(gpu_id=gpu_id)
        logger.info(f"Starting server with device: {self.gpu_id}")

        # 初始化模型
        try:
            self.initialize_models()
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return

    def initialize_models(self):
        """初始化GPU上的模型"""
        logger.info("Initializing models on the device...")
        try:
            logger.info(f"Initializing models on GPU {self.gpu_id}")
            self.setup(self.gpu_id)
        except Exception as e:
            logger.error(f"Failed to initialize models on GPU {self.gpu_id}: {e}")
            raise
        self.models_initialized = True
        logger.info("Model initialized successfully")

    def setup(self, device):
        """Initialize models on specified device"""
        logger.info(f"Setting up models on device {device}")
        if torch.cuda.is_available():
            # 设置环境变量以控制GPU使用
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
            torch.cuda.set_device(device)
            logger.info(f"Set CUDA device to {device}")

        # Remove the patch if get_device is not needed
        try:
            model_manager = ModelSingleton()
            logger.info("Initializing model manager...")
            # Initialize models without patching get_device
            model_manager.get_model(True, False)
            logger.info("First model initialized")
            model_manager.get_model(False, False)
            logger.info("Second model initialized")
            logger.info(f'Model initialization complete on device {device}!')
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def task_processor(self):
        """持续处理队列中的任务"""
        while True:
            try:
                # 获取下一个任务
                task_id = self.task_queue.get_task()
                if task_id and task_id in self.tasks:
                    task_info = self.tasks[task_id]

                    # Ensure the task is in the pending state before processing
                    if task_info['status'] != TaskStatus.PENDING:
                        logger.warning(f"Task {task_id} is not pending, skipping...")
                        continue

                    # Update task status to processing
                    task_info['status'] = TaskStatus.PROCESSING

                    # Log the task assignment for debugging
                    logger.info(f"Assigning task {task_id} to GPU {self.gpu_id}")

                    try:
                        # 处理任务
                        self.process_pdf_task(
                            task_info['pdf_bytes'],
                            task_info['mode'],
                            task_id
                        )
                    finally:
                        # Increment task count
                        self.task_count += 1
                        logger.info(f"Task count: {self.task_count}")

                        # Check if we need to clean memory and reinitialize models
                        if self.task_count >= 50:
                            logger.info("Cleaning memory and reinitializing models after 50 tasks")
                            self.clean_memory(self.gpu_id)
                            self.initialize_models()
                            self.task_count = 0  # Reset task count

                    # 从队列中移除任务
                    self.task_queue.remove_task(task_id)
            except Exception as e:
                logger.exception(f"Error in task processor: {e}")
            # 短暂休息，避免空队列时过度循环
            asyncio.sleep(0.1)  # Reduced sleep time for faster task processing

    def start_processor(self, loop, num_processors=4):
        """在指定的事件循环中启动多个任务处理器"""
        for _ in range(num_processors):
            task = loop.create_task(self.task_processor())
            self.processor_tasks.append(task)
        logger.info("Task processor started successfully")
        return self.processor_tasks

    @staticmethod
    def clean_memory(device):
        """Clean GPU memory"""
        import gc
        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        gc.collect()

    async def decode_request(self, file_bytes, mode):
        """Decode incoming request"""
        try:
            # 读取文件内容为二进制
            # pdf_file = await file.read()
            # mode = request.get('mode', 'auto')

            # 验证模式
            if mode not in ['ocr', 'txt', 'auto']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid mode: {mode}. Use 'ocr', 'txt' or 'auto'"
                )

            # 验证文件类型
            mime_type = filetype.guess_mime(file_bytes)
            if not mime_type:
                raise HTTPException(
                    status_code=400,
                    detail="Could not determine file type"
                )
            if mime_type != 'application/pdf':
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {mime_type}. Only PDF files are supported"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error decoding request: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def process_pdf_task(self, file_bytes, mode, task_id):
        """异步处理PDF任务"""
        # Log the task assignment for debugging
        await self.task_queue.add_task(task_id, {'created_at': self.tasks[task_id]['created_at']})
        await self.decode_request(file_bytes, mode)
        # self.task_queue.pending_tasks[task_id] = {'created_at': self.tasks[task_id]['created_at']}
        logger.info(f"Assigning task {task_id} to GPU {self.gpu_id}")

        temp_output_dir = os.path.join(self.output_base_dir, 'temp', task_id)
        final_output_dir = os.path.join(self.results_dir, task_id)

        try:
            self.tasks[task_id]["status"] = TaskStatus.PROCESSING
            self.tasks[task_id]["gpu_id"] = self.gpu_id

            # 创建临时目录和最终目录
            os.makedirs(temp_output_dir, exist_ok=True)
            os.makedirs(final_output_dir, exist_ok=True)

            # Process PDF
            model_json = []
            jso_useful_key = {
                "_pdf_type": mode,
                "model_list": model_json,
                "device": self.gpu_id
            }

            local_image_dir = os.path.join(temp_output_dir, 'images')
            image_dir = 'images'
            image_writer = DiskReaderWriter(local_image_dir)

            pipe = UNIPipe(file_bytes, jso_useful_key, image_writer)
            pipe.pipe_classify()

            if model_config.__use_inside_model__:
                pipe.pipe_analyze()
            else:
                raise Exception("Model list input required")

            pipe.pipe_parse()
            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")

            # 保存Markdown内容到最终目录
            md_filename = f"{task_id}.md"
            final_md_path = os.path.join(final_output_dir, md_filename)
            with open(final_md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            # 移动图片到最终目录
            final_images_dir = os.path.join(final_output_dir, 'images')
            if os.path.exists(local_image_dir):
                # 如果最终图片目录已存在，先删除它
                if os.path.exists(final_images_dir):
                    shutil.rmtree(final_images_dir)
                # 移动整个图片目录
                shutil.move(local_image_dir, final_images_dir)

            # 收集图片数据用于响应
            images_data = {}
            if os.path.exists(final_images_dir):
                for img_file in os.listdir(final_images_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        img_path = os.path.join(final_images_dir, img_file)
                        with open(img_path, 'rb') as f:
                            img_data = f.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            images_data[img_file] = img_base64

            # 更新任务结果
            self.tasks[task_id].update({
                'status': TaskStatus.COMPLETED,
                'md_content': md_content,
                'images': images_data,
                'result_path': final_output_dir
            })

            logger.info(f"Task {task_id} completed. Results saved to {final_output_dir}")

        except Exception as e:
            logger.exception(f"Error processing task {task_id} on GPU {self.gpu_id}: {e}")
            self.tasks[task_id].update({
                'status': TaskStatus.FAILED,
                'error': str(e)
            })
        finally:
            # 检查是否需要清理显存
            if self.should_clean_memory():
                logger.info(f"Cleaning memory for GPU {self.gpu_id}")
                self.clean_memory(self.gpu_id)

            # 只清理临时目录，保留最终结果目录
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)

    def should_clean_memory(self) -> bool:
        """检查是否需要清理显存"""
        # Simplified logic for single GPU
        return True

    def predict(self, task_id, file_bytes, mode, background_tasks: BackgroundTasks):
        """将任务加入队列"""
        # pdf_bytes, mode = inputs
        # pdf_bytes = await file.read()
        # task_id = str(uuid.uuid4())

        # 初始化任务状态
        self.tasks[task_id] = {
            'status': TaskStatus.PENDING,
            'created_at': time.time(),
            'pdf_bytes': "",
            'mode': mode
        }
        #
        # # 将任务加入队列
        loop = asyncio.get_event_loop()
        loop.set_debug(True)
        loop.create_task(self.process_pdf_task(file_bytes, mode, task_id))
        print("----------create_task------------------")

        return {
            'task_id': task_id,
            'queue_position': self.task_queue.queue_size
        }

    # def get_task_info(self, task_id: str):
    #     """获取任务状态和结果"""
    #     if task_id not in self.tasks:
    #         raise HTTPException(status_code=404, detail="Task not found")
    #
    #     task_info = self.tasks[task_id].copy()
    #     # 移除大型二进制数据
    #     task_info.pop('pdf_bytes', None)
    #
    #     # 添加GPU使用情况
    #     if 'gpu_id' in task_info:
    #         gpu_id = task_info['gpu_id']
    #         task_info['gpu_info'] = {
    #             'current_tasks': 1,
    #             'total_tasks': 1
    #         }
    #
    #     # 添加队列信息
    #     if task_info['status'] == TaskStatus.PENDING:
    #         queue_info = self.task_queue.get_queue_info()
    #         try:
    #             queue_position = queue_info['pending_tasks'].index(task_id) + 1
    #         except ValueError:
    #             queue_position = None
    #         task_info['queue_info'] = {
    #             'position': queue_position,
    #             'total_queued': queue_info['queue_size']
    #         }
    #
    #     return task_info

    # def encode_response(self, response):
    #     """Encode response data"""
    #     try:
    #         if isinstance(response, bytes):
    #             # 如果是二进制数据，转换为base64
    #             return base64.b64encode(response).decode('utf-8')
    #         elif isinstance(response, dict):
    #             # 如果是字典，递归处理其中的二进制数据
    #             encoded_response = {}
    #             for key, value in response.items():
    #                 if isinstance(value, bytes):
    #                     encoded_response[key] = base64.b64encode(value).decode('utf-8')
    #                 elif isinstance(value, dict):
    #                     # 递归处理嵌套字典
    #                     encoded_response[key] = self.encode_response(value)
    #                 else:
    #                     encoded_response[key] = value
    #             return encoded_response
    #         elif isinstance(response, list):
    #             # 处理列表
    #             return [self.encode_response(item) for item in response]
    #         return response
    #     except Exception as e:
    #         logger.error(f"Error encoding response: {str(e)}")
    #         raise HTTPException(status_code=500, detail=f"Error encoding response: {str(e)}")

    def get_result_file_paths(self, task_id: str) -> dict:
        """获取任务结果文件的路径"""
        md_path = os.path.join(self.results_dir, f"{task_id}.md")
        images_dir = os.path.join(self.results_dir, task_id, 'images')
        return {
            'md_file': md_path,
            'images_dir': images_dir
        }

# 创建FastAPI应用
# app = FastAPI()
# # service = MinerUService(gpu_id=gpu_id)  # 将在启动时初始化
#
#
# @app.post("/predict")
# async def predict(
#         file: UploadFile = File(...),
#         mode: Optional[str] = Form('auto')
# ):
#     try:
#         # Check if the queue is full
#         if service.task_queue.queue_size >= TaskQueue.MAX_QUEUE_SIZE:
#             raise HTTPException(
#                 status_code=429,
#                 detail="Task queue is full. Please try again later."
#             )
#
#         # Construct request object
#         request = {
#             'file': file,
#             'mode': mode
#         }
#         result = service.predict(service.decode_request(request))
#         return service.encode_response(result)
#     except Exception as e:
#         logger.error(f"Error in predict endpoint: {str(e)}")
#         if isinstance(e, HTTPException):
#             raise e
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.get("/task/{task_id}")
# async def get_task_status(task_id: str):
#     try:
#         result = service.get_task_info(task_id)
#         return service.encode_response(result)
#     except Exception as e:
#         logger.error(f"Error in get_task_status endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.get("/status")
# async def get_server_status():
#     """获取服务器状态"""
#     return {
#         'gpu_status': {
#             service.gpu_id: {
#                 'current_tasks': 1,
#                 'total_tasks': 1
#             }
#         }
#     }
#
#
# @app.get("/queue")
# async def get_queue_status():
#     """获取队列状态"""
#     queue_info = service.task_queue.get_queue_info()
#     return {
#         'queue': queue_info,
#         'gpu_status': {
#             service.gpu_id: {
#                 'current_tasks': 1,
#                 'total_tasks': 1
#             }
#         }
#     }
#
#
# @app.get("/download/{task_id}")
# async def download_results(task_id: str):
#     """
#     下载处理后的结果文件。
#     成功下载后，删除服务器上的结果文件。
#     """
#     try:
#         task_info = service.get_task_info(task_id)
#         if task_info['status'] != TaskStatus.COMPLETED:
#             raise HTTPException(status_code=400, detail="Task is not completed yet.")
#
#         # 获取任务的结果目录
#         task_dir = os.path.join(service.results_dir, task_id)
#         if not os.path.exists(task_dir):
#             raise HTTPException(status_code=404, detail="Result directory not found.")
#
#         # 检查markdown文件是否存在
#         md_file = os.path.join(task_dir, f"{task_id}.md")
#         if not os.path.exists(md_file):
#             raise HTTPException(status_code=404, detail="Markdown file not found.")
#
#         # 创建临时目录用于打包
#         temp_dir = os.path.join(service.output_base_dir, 'temp_zip')
#         os.makedirs(temp_dir, exist_ok=True)
#
#         # 创建用于打包的目录结构
#         zip_content_dir = os.path.join(temp_dir, task_id)
#         os.makedirs(zip_content_dir, exist_ok=True)
#
#         # 复制文件到打包目录
#         shutil.copy2(md_file, zip_content_dir)
#         images_dir = os.path.join(task_dir, 'images')
#         if os.path.exists(images_dir):
#             shutil.copytree(images_dir, os.path.join(zip_content_dir, 'images'), dirs_exist_ok=True)
#
#         # 创建ZIP文件
#         zip_path = os.path.join(temp_dir, f"{task_id}.zip")
#         shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', temp_dir, task_id)
#
#         # 确保ZIP文件存在
#         if not os.path.exists(zip_path):
#             raise HTTPException(status_code=500, detail="Failed to create ZIP file.")
#
#         # 返回ZIP文件
#         response = FileResponse(
#             path=zip_path,
#             filename=f"{task_id}.zip",
#             media_type='application/zip'
#         )
#
#         # 设置清理函数
#         async def cleanup():
#             max_retries = 3
#             retry_delay = 1
#
#             for attempt in range(max_retries):
#                 try:
#                     await asyncio.sleep(retry_delay)  # 等待一段时间确保文件传输完成
#                     if os.path.exists(temp_dir):
#                         shutil.rmtree(temp_dir)
#                     break
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         logger.error(f"Failed to clean up temporary files after {max_retries} attempts: {e}")
#                     else:
#                         retry_delay *= 2  # 指数退避
#                         continue
#
#         # 创建清理任务
#         asyncio.create_task(cleanup())
#
#         return response
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception(f"Error in download_results endpoint: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# def start_server(port=8000, reload=False, gpu_id=0):
#     """
#     启动服务器，使用指定的GPU
#     Args:
#         port: 服务端口
#         reload: 是否启用自动重载
#         gpu_id: 要使用的GPU编号
#     """
#     global service
#
#     logger.info("Starting MinerU server...")
#
#     # 检查指定的GPU是否可用
#     if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
#         logger.info(f"Using GPU {gpu_id}")
#     else:
#         logger.warning(f"GPU {gpu_id} is not available. Running in CPU mode.")
#         gpu_id = 0  # 使用CPU模式
#
#     # 初始化服务
#     service = MinerUService(gpu_id=gpu_id)
#     logger.info(f"Starting server with device: {gpu_id}")
#
#     # 初始化模型
#     try:
#         service.initialize_models()
#     except Exception as e:
#         logger.error(f"Failed to initialize models: {e}")
#         return
#
#     import uvicorn
#     import platform
#
#     if platform.system() == 'Windows':
#         # Windows特定配置
#         config = uvicorn.Config(
#             app=app,
#             host="0.0.0.0",
#             port=port,
#             loop="asyncio",  # Windows上使用默认的asyncio事件循环
#             log_level="info",
#             timeout_keep_alive=300,  # 增加保持连接的超时时间
#             reload=reload,  # 添加reload支持
#             reload_dirs=['file_handle']  # 监视的目录
#         )
#     else:
#         # Linux/Unix特定配置
#         try:
#             import uvloop
#             asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
#             logger.info("Using uvloop event loop")
#         except ImportError:
#             logger.warning("uvloop not installed. Using default event loop.")
#
#         config = uvicorn.Config(
#             app=app,
#             host="0.0.0.0",
#             port=port,
#             log_level="info",
#             timeout_keep_alive=300,
#             reload=reload,  # 添加reload支持
#             reload_dirs=['file_handle']  # 监视的目录
#         )
#
#     # 创建服务器
#     server = uvicorn.Server(config)
#
#     # 设置启动事件处理
#     async def startup():
#         try:
#             loop = asyncio.get_event_loop()
#             service.start_processor(loop)  # 移除 'await' 以避免阻塞
#             logger.info("Task processor started successfully")
#             logger.info(f"Server is ready to accept connections at http://0.0.0.0:{port}")
#         except Exception as e:
#             logger.error(f"Failed to start task processor: {e}")
#             raise
#
#     # 注册启动事件
#     app.add_event_handler("startup", startup)
#
#     # 运行服务器
#     logger.info("Starting uvicorn server...")
#     try:
#         server.run()
#     except Exception as e:
#         logger.error(f"Server failed to start: {e}")
#         raise


# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Start MinerU Service')
#     parser.add_argument('--port', type=int, default=8000, help='Port to run the service on')
#     parser.add_argument('--reload', action='store_true', help='Enable auto-reload on code changes')
#     parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use for the service')
#
#     args = parser.parse_args()
#     start_server(port=args.port, reload=args.reload, gpu_id=args.gpu_id)
