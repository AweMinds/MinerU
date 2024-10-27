import os
import json
import time
import requests
import base64
import asyncio
from typing import Optional
import itertools
import shutil  # 添加导入以处理ZIP文件


class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Define a list of server IPs and ports
SERVERS = [
    {'ip': 'http://localhost', 'ports': [8000]},
    #   {'ip': 'http://192.168.2.152', 'ports': [8000, 8001, 8002, 8003]},
    # Add more servers as needed
]

# Set a timeout for requests
REQUEST_TIMEOUT = 5  # seconds


async def submit_task_to_server(pdf_path: str, mode: str, server_url: str):
    """Submit a task to a specific server URL."""
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            data = {'mode': mode}
            response = await asyncio.to_thread(requests.post, f'{server_url}/predict', files=files, data=data,
                                               timeout=REQUEST_TIMEOUT)

            if response.status_code == 429:
                return None  # Queue is full
            elif response.status_code != 200:
                raise Exception(f"Error submitting task: {response.text}")

            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Server {server_url} is unresponsive: {e}")
        return None


async def check_task_status_on_server(server_url: str, task_id: str):
    """Check task status on a specific server URL."""
    try:
        response = await asyncio.to_thread(requests.get, f'{server_url}/task/{task_id}', timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            raise Exception(f"Error checking task status: {response.text}")
        result = response.json()
        # 添加结果路径检查
        if result.get('status') == TaskStatus.COMPLETED and not result.get('result_path'):
            print(f"Warning: Task {task_id} is marked as completed but result_path is missing")
            return None
        return result
    except requests.exceptions.RequestException as e:
        print(f"Server {server_url} is unresponsive: {e}")
        return None


async def distribute_tasks(pdf_list, servers, file_list_path):  # 添加 file_list_path 参数
    """Distribute tasks to available server processing units."""
    # Create processing units as a list of (ip, port) tuples
    processing_units = [(server['ip'], port) for server in servers for port in server['ports']]
    processing_units_cycle = itertools.cycle(processing_units)

    # Load existing task status
    try:
        with open(file_list_path, 'r') as f:
            task_status = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        task_status = {}

    pending_queue = [pdf for pdf in pdf_list if
                     pdf not in task_status or task_status[pdf]['status'] == TaskStatus.PENDING]
    batch_size = len(processing_units)

    while pending_queue:
        current_batch = pending_queue[:batch_size]
        pending_queue = pending_queue[batch_size:]

        assign_tasks = []
        for pdf in current_batch:
            server = next(processing_units_cycle)
            server_url = f"{server[0]}:{server[1]}"
            assign_tasks.append((pdf, server_url))
            print(f"Assigning task for {pdf} to server {server_url}")

        # Submit current batch of tasks
        batch_results = await asyncio.gather(*[
            submit_task_to_server(pdf, 'auto', server_url) for pdf, server_url in assign_tasks
        ], return_exceptions=True)

        # Count successful assignments
        successful_assignments = 0
        for (pdf, server_url), result in zip(assign_tasks, batch_results):
            if isinstance(result, Exception):
                print(f"Failed to assign task for {pdf} to server {server_url}: {result}")
                task_status[pdf] = {'status': TaskStatus.PENDING}
                pending_queue.append(pdf)
            elif result is None:
                print(f"Server queue is full for {server_url}. Task for {pdf} will be retried later.")
                task_status[pdf] = {'status': TaskStatus.PENDING}
                pending_queue.append(pdf)
                await asyncio.sleep(5)
            elif result:
                task_status[pdf] = {
                    'status': TaskStatus.PROCESSING,
                    'task_id': result.get('task_id'),
                    'server': server_url
                }
                successful_assignments += 1
            else:
                task_status[pdf] = {'status': TaskStatus.PENDING}
                pending_queue.append(pdf)
                print(f"Failed to assign task for {pdf} to server {server_url}")

        # Save task status to JSON
        with open(file_list_path, 'w') as f:
            json.dump(task_status, f, indent=4)

        print(f"Batch assigned. Pending queue size: {len(pending_queue)}")
        await asyncio.sleep(0.5)

    return task_status


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


# 修改 check_task_status 函数，使用共享的 JSON 文件来同步状态
async def check_task_status(file_list_path: str):
    """Check the status of tasks and update the JSON file."""
    processed_tasks = set()  # 用于追踪已经处理过的任务

    while True:
        try:
            # 读取当前状态
            with open(file_list_path, 'r') as f:
                task_status = json.load(f)

            # 检查是否有任务需要处理
            tasks_to_process = []

            # 计算当前状态
            current_completed = 0
            current_failed = 0
            current_processing = 0
            current_pending = 0

            # 首先统计当前状态
            for pdf, status in task_status.items():
                if isinstance(status, dict):
                    current_status = status.get('status')
                    if current_status == TaskStatus.COMPLETED:
                        current_completed += 1
                    elif current_status == TaskStatus.FAILED:
                        current_failed += 1
                    elif current_status == TaskStatus.PROCESSING:
                        current_processing += 1
                        # 只处理未处理过的任务
                        if pdf not in processed_tasks and status.get('server') and status.get('task_id'):
                            tasks_to_process.append((pdf, status))
                    elif current_status == TaskStatus.PENDING:
                        current_pending += 1

            # 打印当前状态
            print(f"\rStatus: {current_completed} completed, {current_failed} failed, "
                  f"{current_processing} processing, {current_pending} pending", end="", flush=True)

            # 如果没有正在处理或等待的任务，退出
            if current_processing == 0 and current_pending == 0:
                print(f"\nAll tasks have completed!")
                print(f"Final status: {current_completed} completed, {current_failed} failed")
                return

            # 处理收集到的任务
            for pdf, status in tasks_to_process:
                try:
                    result = await check_task_status_on_server(status['server'], status['task_id'])

                    if result:  # 确保服务器返回了结果
                        server_status = result.get('status')

                        if server_status == TaskStatus.COMPLETED:
                            # 确保输出目录存在
                            output_dir = os.path.normpath(os.path.join(os.path.dirname(pdf), 'auto'))
                            os.makedirs(output_dir, exist_ok=True)

                            # 下载结果
                            success = await download_and_save_results(
                                server_url=status['server'],
                                task_id=status['task_id'],
                                output_dir=output_dir
                            )

                            if success:
                                task_status[pdf]['status'] = TaskStatus.COMPLETED
                                processed_tasks.add(pdf)
                                print(f"\nSuccessfully downloaded results for {pdf}")

                                # 保存更新后的状态
                                with open(file_list_path, 'w') as f:
                                    json.dump(task_status, f, indent=4)
                            else:
                                print(f"\nFailed to download results for {pdf}, will retry later")

                        elif server_status == TaskStatus.FAILED:
                            task_status[pdf]['status'] = TaskStatus.FAILED
                            processed_tasks.add(pdf)
                            print(f"\nTask failed for {pdf}")

                            # 保存更新后的状态
                            with open(file_list_path, 'w') as f:
                                json.dump(task_status, f, indent=4)

                except Exception as e:
                    print(f"\nError checking status for {pdf}: {e}")
                    continue

            # 等待一段时间后继续检查
            await asyncio.sleep(2)

        except Exception as e:
            print(f"\nError in check_task_status: {e}")
            await asyncio.sleep(5)


def save_results(pdf_path, result):
    """Save the results of a completed task."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(os.path.dirname(pdf_path), 'auto')
    os.makedirs(output_dir, exist_ok=True)

    # Save markdown content
    md_path = os.path.join(output_dir, f"{pdf_name}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(result['md_content'])

    # Save images
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    for img_name, img_base64 in result['images'].items():
        img_path = os.path.join(images_dir, img_name)
        img_data = base64.b64decode(img_base64)
        with open(img_path, 'wb') as f:
            f.write(img_data)


# 修改 main 函数
async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process PDF files in a directory using MinerU service')
    parser.add_argument('directory', help='Path to the directory containing PDF files')

    args = parser.parse_args()

    try:
        # Collect all PDF files in the directory
        pdf_list = []
        for root, _, files in os.walk(args.directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_list.append(os.path.join(root, file))

        # Initialize task status for each PDF
        file_list_path = 'E:\\source_file\\file_list.json'

        # 确保目录存在
        os.makedirs(os.path.dirname(file_list_path), exist_ok=True)

        # 初始化或加载现有的任务状态
        if os.path.exists(file_list_path):
            with open(file_list_path, 'r') as f:
                task_status = json.load(f)
        else:
            task_status = {}

        # 更新任务状态
        for pdf in pdf_list:
            if pdf not in task_status:
                task_status[pdf] = {
                    'status': TaskStatus.PENDING,
                    'task_id': None,
                    'server': None
                }

        # 保存初始状态
        with open(file_list_path, 'w') as f:
            json.dump(task_status, f, indent=4)

        # 并行执行任务分发和状态检查
        distribute_task = asyncio.create_task(distribute_tasks(pdf_list, SERVERS, file_list_path))
        status_check_task = asyncio.create_task(check_task_status(file_list_path))

        # 等待两个任务都完成，添加超时处理
        try:
            await asyncio.wait_for(
                asyncio.gather(distribute_task, status_check_task),
                timeout=3600  # 设置1小时超时
            )
            print("All tasks have been completed successfully!")
        except asyncio.TimeoutError:
            print("Operation timed out after 1 hour")
        except Exception as e:
            print(f"An error occurred while processing tasks: {e}")

    except Exception as e:
        print(f"Processing failed: {e}")
    finally:
        # 确保程序能够正常退出
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()

        print("Program finished.")


if __name__ == '__main__':
    asyncio.run(main())
