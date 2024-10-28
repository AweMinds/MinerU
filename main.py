import asyncio
import base64
import os
import shutil
import time
import uuid
from typing import Optional

import dotenv
import filetype
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse  # 添加导入以支持文件下载
from loguru import logger

from minerU_server import MinerUService, TaskQueue, TaskStatus
from redis_service import redis

loop = asyncio.get_event_loop()
loop.set_debug(True)
start_worker_index = 0
dotenv.load_dotenv()

worker_hosts = os.environ.get("WORKER_HOSTS").strip("[]").split(",")
start_mode = os.environ.get("START_MODE")
output_dir = os.environ.get("OUTPUT_DIR")
gpu_id = os.environ.get("GPU_ID")
host_name = os.environ.get("HOST_NAME")

# 创建FastAPI应用
app = FastAPI()
service = MinerUService(gpu_id=int(gpu_id))  # 将在启动时初始化
REQUEST_TIMEOUT = 5


def init_worker_status():
    if start_mode == "WORKER":
        redis.store_data(host_name, "workload", 0)


init_worker_status()


@app.get("/app_status")
async def root():
    return {"status": "UP"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: Optional[str] = Form('auto'),
                  task_id: Optional[str] = Form(None)):
    print("---------")
    if task_id is None:
        task_id = str(uuid.uuid4())

    filename = file.filename

    if start_mode == "SERVER":
        return await proxy_predict(filename, file, mode, task_id)
    else:
        return handler_predict(filename, file, mode, task_id)


def handler_predict(filename, file, mode, task_id):
    start = time.perf_counter()
    logger.info(f"Task-{task_id} Started")

    file_bytes = file.file.read()

    result = service.predict(task_id, file_bytes, mode)
    file_read_end = time.perf_counter()
    logger.info(f"Task-{task_id} read file Finished. Elapsed time: {file_read_end - start}")

    return result


async def proxy_predict(filename, file, mode, task_id):
    # task_id = str(uuid.uuid4())

    file_bytes = file.file.read()
    files = {'file': file_bytes}
    data = {'mode': mode, 'task_id': task_id}

    if len(worker_hosts) == 0:
        logger.error("No worker hosts detected")

    logger.info(f"Task-{task_id} Started")
    worker_host = get_available_worker()
    if worker_host is None:
        logger.error(f"Task-{task_id} Reject. Task queue is full. Please try again later.")
        raise HTTPException(
            status_code=429,
            detail="Task queue is full. Please try again later."
        )

    redis.store_data(task_id, "status", TaskStatus.PROCESSING)
    redis.store_data(task_id, "data", {"worker_host": worker_host, "filename": filename, "mode": mode})

    response = await asyncio.to_thread(requests.post, f'{worker_host}/predict', files=files, data=data,
                                       timeout=REQUEST_TIMEOUT)
    logger.info(f"Send predit request to worker: {worker_host}")

    return response.json()


@app.get("/workers")
def get_available_worker():
    global start_worker_index
    host_size = len(worker_hosts)

    for index in range(host_size):
        worker_index = (index + start_worker_index) % host_size
        worker_host = worker_hosts[worker_index]
        worker_data = redis.get_task_data(worker_host)
        if worker_data is None:
            continue

        if worker_data.get("workload") == 0:
            start_worker_index = (worker_index + 1) % host_size
            return worker_host

    return None


# @app.post("/predict_proxy")
# async def predict(file: UploadFile = File(...), mode: str = 'auto', background_tasks: BackgroundTasks = None):
#     # try:
#     task_id = str(uuid.uuid4())
#
#     start = time.perf_counter()
#     logger.info(f"Task-{task_id} Started")
#
#     # Check if the queue is full
#     if service.task_queue.queue_size >= TaskQueue.MAX_QUEUE_SIZE:
#         logger.error(f"Task-{task_id} Reject. Task queue is full. Please try again later.")
#         raise HTTPException(
#             status_code=429,
#             detail="Task queue is full. Please try again later."
#         )
#
#     file_bytes = file.file.read()
#
#     files = {'file': file_bytes}
#     data = {'mode': mode}
#     if len(worker_hosts) == 0:
#         logger.error("No worker hosts detected")
#
#     worker_host = worker_hosts[0]
#     response = await asyncio.to_thread(requests.post, f'{worker_host}/predict', files=files, data=data,
#                                        timeout=REQUEST_TIMEOUT)
#     logger.info(f"Send predit request to worker: {worker_host}")
#
#     return response.json()


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    try:
        result = get_task_info(task_id)
        return encode_response(result)
    except Exception as e:
        logger.error(f"Error in get_task_status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/task/{task_id}/status")
async def task_status_update(task_id: str, data: dict):
    logger.info(f"Start to download Task-{task_id} files")

    output_root_dir = os.path.join(output_dir, 'auto')
    os.makedirs(output_root_dir, exist_ok=True)

    task_data = redis.get_task_data(task_id)
    worker_host = task_data.get("data", {}).get("worker_host", "")
    if worker_host is None:
        logger.error(f"Task-{task_id}: No worker hosts detected")

    result_dir = os.path.join(output_root_dir, task_id)

    # 下载结果
    success = await download_and_save_results(
        server_url=worker_host,
        task_id=task_id,
        output_dir=result_dir
    )
    if success:
        redis.store_data(task_id, "result", {"result_dir": result_dir})
        redis.store_data(task_id, "status", TaskStatus.COMPLETED)
        logger.info(f"Task-{task_id} file download successfully. Save to {result_dir}")
    else:
        redis.store_data(task_id, "status", TaskStatus.FAILED)
        logger.error(f"Task-{task_id} fail to download files.")


async def download_and_save_results(server_url: str, task_id: str, output_dir: str, max_retries=3) -> bool:
    """下载并存任务结果，包含重试机制"""
    zip_path = None
    for attempt in range(max_retries):
        try:
            # 创建输出目录（确保父目录存在）
            os.makedirs(output_dir, exist_ok=True)

            # 下载ZIP文件
            download_url = f"{server_url}/download/{task_id}"
            response = await asyncio.to_thread(
                requests.get,
                download_url,
                stream=True,
                timeout=30
            )

            if response.status_code == 404:
                print(f"Results not ready yet for task {task_id} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            if response.status_code != 200:
                print(f"Failed to download results for task {task_id}: {response.text}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            # 保存ZIP文件到临时目录
            temp_dir = os.path.join(output_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            zip_path = os.path.join(temp_dir, f"{task_id}.zip")

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            await asyncio.sleep(0.5)  # 确保文件写入完成

            try:
                # 解压到临时目录
                extract_dir = os.path.join(temp_dir, 'extract')
                os.makedirs(extract_dir, exist_ok=True)
                shutil.unpack_archive(zip_path, output_dir)

                # 移动文件到正确的位置
                task_dir = os.path.join(output_dir, task_id)
                if os.path.exists(task_dir):
                    # 移动Markdown文件
                    md_file = os.path.join(task_dir, f"{task_id}.md")
                    if os.path.exists(md_file):
                        target_md_path = os.path.join(output_dir, f"{task_id}.md")
                        if os.path.exists(target_md_path):
                            os.remove(target_md_path)
                        shutil.move(md_file, target_md_path)

                    # 移动images目录
                    images_dir = os.path.join(task_dir, 'images')
                    if os.path.exists(images_dir):
                        target_images_dir = os.path.join(output_dir, 'images')
                        if os.path.exists(target_images_dir):
                            shutil.rmtree(target_images_dir)
                        shutil.move(images_dir, target_images_dir)

                    # 清理临时目录
                    shutil.rmtree(task_dir)

                # 删除ZIP文件（添加重试机制）
                max_delete_retries = 3
                for delete_attempt in range(max_delete_retries):
                    try:
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        break
                    except Exception as e:
                        if delete_attempt == max_delete_retries - 1:
                            print(f"Failed to delete ZIP file after {max_delete_retries} attempts: {e}")
                        else:
                            await asyncio.sleep(1)  # 等待一秒后重试

                return True  # 成功完成所有操作

            except Exception as e:
                print(f"Error extracting ZIP file: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue  # 继续下一次重试

        except Exception as e:
            print(f"Error downloading results (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            continue  # 继续下一次重试

    return False  # 所有重试都失败


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
