import os
import subprocess

def generate_env_file(ip, gpu_id, index):
    env_content = f"""HOST_NAME=http://{ip}:800{index}
GPU_ID={gpu_id}

# Server host
SERVER_HOST=http://127.0.0.1:8080

#Worker host
WORKER_HOSTS=[http://{ip}:8000,http://{ip}:8001,http://{ip}:8002,http://{ip}:8003]

# Start Mode: SERVER | WORKER
START_MODE=WORKER

#redis
REDIS_HOST=r-uf6pgdnxgdu43ajndmpd.redis.rds.aliyuncs.com
REDIS_PWD=AweMinds2023

#output dir
OUTPUT_DIR=/tmp/minerU/results
"""
    file_name = f"env_{index}"
    with open(file_name, "w") as f:
        f.write(env_content)
    return file_name

def start_uvicorn(ip, num_gpus):
    processes = []
    for i in range(num_gpus):
        env_file = generate_env_file(ip, i, i)
        # 先执行source命令
        source_command = "source start_server.sh"
        subprocess.run(source_command, shell=True, executable='/bin/bash')
        
        # 然后启动uvicorn
        command = f"uvicorn main:app --host 0.0.0.0 --port 800{i} --env-file {env_file}"
        proc = subprocess.Popen(command, shell=True)
        processes.append(proc)
    return processes

if __name__ == "__main__":
    ip_address = input("请输入IP地址: ")
    num_gpus = int(input("请输入GPU数量: "))
    
    processes = start_uvicorn(ip_address, num_gpus)
    print(f"已启动 {num_gpus} 个Uvicorn实例。")
    
    # 等待所有进程完成
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("正在终止所有进程...")
        for proc in processes:
            proc.terminate()
