import redis
import json
import os
from typing import Any, Dict, Optional
from datetime import datetime


class RedisService:
    def __init__(self,
                 host='r-uf6pgdnxgdu43ajndmpd.redis.rds.aliyuncs.com',
                 port=6379,
                 db=0,
                 password=None):  # 添加密码参数
        """初始化Redis连接
        
        Args:
            host: Redis服务器地址
            port: Redis端口
            db: 数据库编号
            password: Redis密码
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,  # 添加密码
                decode_responses=True,  # 自动将响应解码为字符串
                socket_timeout=5,  # 添加超时设置
                socket_connect_timeout=5,
                retry_on_timeout=True  # 超时时自动重试
            )
            # 测试连接
            self.redis_client.ping()
            print("Successfully connected to Redis")
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while connecting to Redis: {e}")
            raise

        self.expiration_time = 3600000  # 过期时间（秒）

    def store_data(self, task_id: str, key: str, value: Any) -> bool:
        """
        存储与task_id关联的键值对数据
        
        Args:
            task_id: 任务ID
            key: 键名
            value: 值（将被转换为JSON字符串存储）
            
        Returns:
            bool: 存储是否成功
        """
        try:
            # 创建完整的键名（使用task_id作为前缀）
            full_key = f"{task_id}:{key}"

            # 将值转换为JSON字符串
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            else:
                value = str(value)

            # 存储数据
            self.redis_client.set(full_key, value, ex=self.expiration_time)

            # 将键添加到任务的键集合中
            task_keys_set = f"{task_id}:keys"
            self.redis_client.sadd(task_keys_set, key)
            self.redis_client.expire(task_keys_set, self.expiration_time)

            return True

        except Exception as e:
            print(f"Error storing data: {e}")
            return False

    def get_task_data(self, task_id: str) -> Optional[Dict]:
        """
        获取与特定task_id关联的所有数据
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict]: 包含所有相关数据的字典，如果出错则返回None
        """
        try:
            # 获取任务的所有键
            task_keys_set = f"{task_id}:keys"
            keys = self.redis_client.smembers(task_keys_set)

            if not keys:
                return None

            # 收集所有数据
            result = {}
            for key in keys:
                full_key = f"{task_id}:{key}"
                value = self.redis_client.get(full_key)

                # 尝试将值解析为JSON
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[key] = value

            return result

        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None

    def export_task_data(self, task_id: str, output_path: str = "E:\\source_file_test\\test\\data.json") -> bool:
        """
        将任务数据导出到JSON文件
        
        Args:
            task_id: 任务ID
            output_path: 输出文件路径
            
        Returns:
            bool: 导出是否成功
        """
        try:
            # 获取任务数据
            data = self.get_task_data(task_id)

            if data is None:
                print(f"No data found for task_id: {task_id}")
                return False

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 添加导出时间戳
            export_data = {
                "task_id": task_id,
                "export_time": datetime.now().isoformat(),
                "data": data
            }

            # 写入JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Error exporting data: {e}")
            return False

    def delete_task_data(self, task_id: str) -> bool:
        """
        删除与任务相关的所有数据
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            # 获取任务的所有键
            task_keys_set = f"{task_id}:keys"
            keys = self.redis_client.smembers(task_keys_set)

            # 删除所有相关数据
            for key in keys:
                full_key = f"{task_id}:{key}"
                self.redis_client.delete(full_key)

            # 删除键集合
            self.redis_client.delete(task_keys_set)

            return True

        except Exception as e:
            print(f"Error deleting data: {e}")
            return False

    def get_all_tasks(self) -> Dict[str, Dict]:
        """
        获取所有任务的数据
        
        Returns:
            Dict[str, Dict]: 包含所有任务数据的字典，key为task_id
        """
        try:
            # 使用scan_iter来获取所有键
            all_tasks = {}
            # 使用pattern匹配所有task_id:keys的键
            for key in self.redis_client.scan_iter(match="*:keys"):
                # 从key中提取task_id
                task_id = key.split(':')[0]
                # 获取该task_id的所有数据
                task_data = self.get_task_data(task_id)
                if task_data:
                    all_tasks[task_id] = task_data

            return all_tasks
        except Exception as e:
            print(f"Error getting all tasks: {e}")
            return {}

    def get_pending_tasks(self) -> Dict[str, Dict]:
        """
        获取所有pending状态的任务
        
        Returns:
            Dict[str, Dict]: 包含所有pending状态任务的字典
        """
        try:
            all_tasks = self.get_all_tasks()
            return {
                task_id: task_data
                for task_id, task_data in all_tasks.items()
                if task_data.get('status') == 'pending'
            }
        except Exception as e:
            print(f"Error getting pending tasks: {e}")
            return {}

    def get_task_count(self) -> Dict[str, int]:
        """
        获取各种状态的任务数量
        
        Returns:
            Dict[str, int]: 包含各状态任务数量的字典
        """
        try:
            all_tasks = self.get_all_tasks()
            counts = {
                'pending': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0,
                'total': len(all_tasks)
            }

            for task_data in all_tasks.values():
                status = task_data.get('status', 'unknown')
                if status in counts:
                    counts[status] += 1

            return counts
        except Exception as e:
            print(f"Error getting task counts: {e}")
            return {
                'pending': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0,
                'total': 0
            }


# 使用示例
def example_usage():
    # 创建Redis服务实例
    redis_service = RedisService()

    # 存储示例数据
    task_id = "example_task_001"
    redis_service.store_data(task_id, "name", "Test Task")
    redis_service.store_data(task_id, "status", "processing")
    redis_service.store_data(task_id, "metadata", {
        "created_at": "2024-03-20",
        "priority": "high"
    })

    # 导出数据到JSON文件
    redis_service.export_task_data(task_id)

    # 清理数据
    redis_service.delete_task_data(task_id)


if __name__ == "__main__":
    example_usage()
