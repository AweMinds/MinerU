import base64
import os
import shutil
import time
import uuid
from typing import Optional

import asyncio

import filetype
import requests
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse  # 添加导入以支持文件下载

from loguru import logger

from minerU_server import MinerUService, TaskQueue, TaskStatus

loop = asyncio.get_event_loop()
loop.set_debug(True)

# 创建FastAPI应用
app = FastAPI()

service = MinerUService(gpu_id=0)  # 将在启动时初始化

# # 注册启动事件
# app.add_event_handler("startup", startup)


# start_server(port=args.port, reload=args.reload, gpu_id=args.gpu_id)
REQUEST_TIMEOUT = 5


@app.get("/app_status")
async def root():
    return {"status": "UP"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = 'auto', background_tasks: BackgroundTasks = None):
    # try:
    task_id = str(uuid.uuid4())

    start = time.perf_counter()
    logger.info(f"Task-{task_id} Started")

    # Check if the queue is full
    if service.task_queue.queue_size >= TaskQueue.MAX_QUEUE_SIZE:
        logger.error(f"Task-{task_id} Reject. Task queue is full. Please try again later.")
        raise HTTPException(
            status_code=429,
            detail="Task queue is full. Please try again later."
        )

    file_bytes = file.file.read()

    result = service.predict(task_id, file_bytes, mode, background_tasks)
    file_read_end = time.perf_counter()
    logger.info(f"Task-{task_id} read file Finished. Elapsed time: {file_read_end - start}")

    return encode_response(result)


@app.post("/predict_proxy")
async def predict(file: UploadFile = File(...), mode: str = 'auto', background_tasks: BackgroundTasks = None):
    # try:
    task_id = str(uuid.uuid4())

    start = time.perf_counter()
    logger.info(f"Task-{task_id} Started")

    # Check if the queue is full
    if service.task_queue.queue_size >= TaskQueue.MAX_QUEUE_SIZE:
        logger.error(f"Task-{task_id} Reject. Task queue is full. Please try again later.")
        raise HTTPException(
            status_code=429,
            detail="Task queue is full. Please try again later."
        )

    file_bytes = file.file.read()

    files = {'file': file_bytes}
    data = {'mode': mode}
    server_url = "http://127.0.0.1:8001"
    response = await asyncio.to_thread(requests.post, f'{server_url}/predict', files=files, data=data,
                                       timeout=REQUEST_TIMEOUT)

    file_read_end = time.perf_counter()
    logger.info(f"Task-{task_id} read file Finished. Elapsed time: {file_read_end - start}")
    # result = {
    #     'task_id': task_id,
    #     'queue_position': self.task_queue.queue_size
    # }

    return {"result": "OK"}


# except Exception as e:
#     logger.error(f"Error in predict endpoint: {str(e)}")
#     if isinstance(e, HTTPException):
#         raise e
#     raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    try:
        result = get_task_info(task_id)
        return encode_response(result)
    except Exception as e:
        logger.error(f"Error in get_task_status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def get_task_info(task_id: str):
    """获取任务状态和结果"""
    if task_id not in service.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = service.tasks[task_id].copy()
    # 移除大型二进制数据
    task_info.pop('pdf_bytes', None)

    # 添加GPU使用情况
    if 'gpu_id' in task_info:
        gpu_id = task_info['gpu_id']
        task_info['gpu_info'] = {
            'current_tasks': 1,
            'total_tasks': 1
        }

    # 添加队列信息
    if task_info['status'] == TaskStatus.PENDING:
        queue_info = service.task_queue.get_queue_info()
        try:
            queue_position = queue_info['pending_tasks'].index(task_id) + 1
        except ValueError:
            queue_position = None
        task_info['queue_info'] = {
            'position': queue_position,
            'total_queued': queue_info['queue_size']
        }

    return task_info


@app.get("/status")
async def get_server_status():
    """获取服务器状态"""
    return {
        'status': "UP"
    }


@app.get("/queue")
async def get_queue_status():
    """获取队列状态"""
    queue_info = service.task_queue.get_queue_info()
    return {
        'queue': queue_info,
        'gpu_status': {
            service.gpu_id: {
                'current_tasks': 1,
                'total_tasks': 1
            }
        }
    }


@app.get("/download/{task_id}")
async def download_results(task_id: str):
    """
    下载处理后的结果文件。
    成功下载后，删除服务器上的结果文件。
    """
    try:
        task_info = get_task_info(task_id)
        if task_info['status'] != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Task is not completed yet.")

        # 获取任务的结果目录
        task_dir = os.path.join(service.results_dir, task_id)
        if not os.path.exists(task_dir):
            raise HTTPException(status_code=404, detail="Result directory not found.")

        # 检查markdown文件是否存在
        md_file = os.path.join(task_dir, f"{task_id}.md")
        if not os.path.exists(md_file):
            raise HTTPException(status_code=404, detail="Markdown file not found.")

        # 创建临时目录用于打包
        temp_dir = os.path.join(service.output_base_dir, 'temp_zip')
        os.makedirs(temp_dir, exist_ok=True)

        # 创建用于打包的目录结构
        zip_content_dir = os.path.join(temp_dir, task_id)
        os.makedirs(zip_content_dir, exist_ok=True)

        # 复制文件到打包目录
        shutil.copy2(md_file, zip_content_dir)
        images_dir = os.path.join(task_dir, 'images')
        if os.path.exists(images_dir):
            shutil.copytree(images_dir, os.path.join(zip_content_dir, 'images'), dirs_exist_ok=True)

        # 创建ZIP文件
        zip_path = os.path.join(temp_dir, f"{task_id}.zip")
        shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', temp_dir, task_id)

        # 确保ZIP文件存在
        if not os.path.exists(zip_path):
            raise HTTPException(status_code=500, detail="Failed to create ZIP file.")

        # 返回ZIP文件
        response = FileResponse(
            path=zip_path,
            filename=f"{task_id}.zip",
            media_type='application/zip'
        )

        # 设置清理函数
        async def cleanup():
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    await asyncio.sleep(retry_delay)  # 等待一段时间确保文件传输完成
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to clean up temporary files after {max_retries} attempts: {e}")
                    else:
                        retry_delay *= 2  # 指数退避
                        continue

        # 创建清理任务
        asyncio.create_task(cleanup())

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in download_results endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def decode_request(file, mode):
    """Decode incoming request"""
    # 读取文件内容为二进制
    pdf_file = file.read()

    # 验证模式
    if mode not in ['ocr', 'txt', 'auto']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Use 'ocr', 'txt' or 'auto'"
        )

    # 验证文件类型
    mime_type = filetype.guess_mime(pdf_file)
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
    return pdf_file, mode


def encode_response(response):
    """Encode response data"""
    try:
        if isinstance(response, bytes):
            # 如果是二进制数据，转换为base64
            return base64.b64encode(response).decode('utf-8')
        elif isinstance(response, dict):
            # 如果是字典，递归处理其中的二进制数据
            encoded_response = {}
            for key, value in response.items():
                if isinstance(value, bytes):
                    encoded_response[key] = base64.b64encode(value).decode('utf-8')
                elif isinstance(value, dict):
                    # 递归处理嵌套字典
                    encoded_response[key] = encode_response(value)
                else:
                    encoded_response[key] = value
            return encoded_response
        elif isinstance(response, list):
            # 处理列表
            return [encode_response(item) for item in response]
        return response
    except Exception as e:
        logger.error(f"Error encoding response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error encoding response: {str(e)}")
