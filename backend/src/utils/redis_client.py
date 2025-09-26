import redis
import json
import os
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            self.client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.client = None
    
    def set_cache(self, key: str, value: Any, expire: int = 3600):
        """Set cache with expiration"""
        if not self.client:
            return False
        
        try:
            serialized = json.dumps(value) if not isinstance(value, str) else value
            return self.client.setex(key, expire, serialized)
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    def delete_cache(self, key: str):
        """Delete cached value"""
        if not self.client:
            return False
        
        try:
            return self.client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False
    
    def set_task_result(self, task_id: str, result: dict, expire: int = 3600):
        """Store task result"""
        return self.set_cache(f"task:{task_id}", result, expire)
    
    def get_task_result(self, task_id: str) -> Optional[dict]:
        """Get task result"""
        return self.get_cache(f"task:{task_id}")

# Global Redis client instance
redis_client = RedisClient()