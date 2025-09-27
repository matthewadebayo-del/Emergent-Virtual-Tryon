"""
Production health monitoring for VirtualFit
"""
import asyncio
import time
import psutil
import logging
from datetime import datetime
from pathlib import Path

class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_health_check = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/health.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_system_stats(self):
        """Get system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available / (1024**3),  # GB
                "disk_usage": disk.percent,
                "disk_free": disk.free / (1024**3)  # GB
            }
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {}
    
    def get_application_stats(self):
        """Get application-specific stats"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": (self.error_count / max(self.request_count, 1)) * 100,
            "last_health_check": datetime.fromtimestamp(self.last_health_check).isoformat()
        }
    
    def increment_request(self):
        """Increment request counter"""
        self.request_count += 1
    
    def increment_error(self):
        """Increment error counter"""
        self.error_count += 1
    
    def check_health(self):
        """Perform comprehensive health check"""
        self.last_health_check = time.time()
        
        system_stats = self.get_system_stats()
        app_stats = self.get_application_stats()
        
        # Health thresholds
        health_status = "healthy"
        warnings = []
        
        if system_stats.get("cpu_usage", 0) > 80:
            warnings.append("High CPU usage")
            health_status = "warning"
        
        if system_stats.get("memory_usage", 0) > 85:
            warnings.append("High memory usage")
            health_status = "warning"
        
        if system_stats.get("disk_usage", 0) > 90:
            warnings.append("Low disk space")
            health_status = "critical"
        
        if app_stats.get("error_rate", 0) > 10:
            warnings.append("High error rate")
            health_status = "warning"
        
        health_report = {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_stats,
            "application": app_stats,
            "warnings": warnings
        }
        
        # Log health status
        if health_status == "critical":
            self.logger.critical(f"Health check CRITICAL: {warnings}")
        elif health_status == "warning":
            self.logger.warning(f"Health check WARNING: {warnings}")
        else:
            self.logger.info("Health check: All systems normal")
        
        return health_report
    
    async def start_monitoring(self, interval=300):  # 5 minutes
        """Start continuous health monitoring"""
        self.logger.info("Starting health monitoring...")
        
        while True:
            try:
                health_report = self.check_health()
                
                # Save health report
                health_file = Path("logs/health_report.json")
                import json
                with open(health_file, 'w') as f:
                    json.dump(health_report, f, indent=2)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute

# Global health monitor instance
health_monitor = HealthMonitor()