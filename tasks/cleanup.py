from celery import Celery
import logging
import os
import glob
from datetime import datetime
from services.task_manager import TaskManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Celery应用
celery_app = Celery('cleanup')
celery_app.config_from_object('celery_config')

@celery_app.task(name='tasks.cleanup.cleanup_expired_files')
def cleanup_expired_files(task_id: str):
    """
    清理指定任务的过期文件
    """
    try:
        logger.info(f"开始清理任务 {task_id} 的过期文件")
        
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        
        if not task:
            logger.warning(f"任务 {task_id} 不存在，跳过清理")
            return
        
        # 检查任务是否已过期
        if task.expires_at and task.expires_at > datetime.utcnow():
            logger.info(f"任务 {task_id} 尚未过期，跳过清理")
            return
        
        files_deleted = 0
        
        # 删除视频文件
        if task.video_url:
            try:
                filename = task.video_url.split('/')[-1]
                file_path = os.path.join(task_manager.video_storage_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    files_deleted += 1
                    logger.info(f"删除视频文件: {file_path}")
            except Exception as e:
                logger.error(f"删除视频文件失败: {e}")
        
        # 删除缩略图
        if task.thumbnail_url:
            try:
                filename = task.thumbnail_url.split('/')[-1]
                file_path = os.path.join(task_manager.video_storage_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    files_deleted += 1
                    logger.info(f"删除缩略图文件: {file_path}")
            except Exception as e:
                logger.error(f"删除缩略图文件失败: {e}")
        
        # 删除可能残留的临时文件
        temp_patterns = [
            f"/tmp/audio_{task_id}_*.wav",
            f"/tmp/combined_audio_{task_id}.wav",
            f"/tmp/adjusted_video_{task_id}.mp4",
            f"/tmp/final_video_{task_id}.mp4",
            f"/tmp/manim_{task_id}_*.py",
            f"/tmp/manim_output_{task_id}/*"
        ]
        
        for pattern in temp_patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                        files_deleted += 1
                    elif os.path.isdir(file):
                        import shutil
                        shutil.rmtree(file)
                        files_deleted += 1
                    logger.debug(f"删除临时文件: {file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败 {file}: {e}")
        
        logger.info(f"任务 {task_id} 文件清理完成，共删除 {files_deleted} 个文件")
        
        return {
            'task_id': task_id,
            'files_deleted': files_deleted,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"清理任务 {task_id} 失败: {e}")
        raise

@celery_app.task(name='tasks.cleanup.cleanup_all_expired_tasks')
def cleanup_all_expired_tasks():
    """
    清理所有过期任务（定期任务）
    """
    try:
        logger.info("开始清理所有过期任务")
        
        task_manager = TaskManager()
        task_manager.cleanup_expired_tasks()
        
        logger.info("所有过期任务清理完成")
        
        return {
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"清理所有过期任务失败: {e}")
        raise

@celery_app.task(name='tasks.cleanup.cleanup_temp_files')
def cleanup_temp_files():
    """
    清理临时目录中的孤立文件（定期任务）
    """
    try:
        logger.info("开始清理临时文件")
        
        temp_dir = "/tmp"
        files_deleted = 0
        
        # 清理超过1小时的临时文件
        import time
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1小时前
        
        # 匹配模式
        patterns = [
            "audio_*_*.wav",
            "combined_audio_*.wav",
            "adjusted_video_*.mp4",
            "final_video_*.mp4",
            "manim_*_*.py"
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(temp_dir, pattern))
            for file in files:
                try:
                    # 检查文件修改时间
                    if os.path.getmtime(file) < cutoff_time:
                        os.remove(file)
                        files_deleted += 1
                        logger.debug(f"删除过期临时文件: {file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败 {file}: {e}")
        
        # 清理空的临时目录
        temp_dirs = glob.glob(os.path.join(temp_dir, "manim_output_*"))
        for temp_dir_path in temp_dirs:
            try:
                if os.path.isdir(temp_dir_path) and not os.listdir(temp_dir_path):
                    os.rmdir(temp_dir_path)
                    logger.debug(f"删除空临时目录: {temp_dir_path}")
            except Exception as e:
                logger.warning(f"删除临时目录失败 {temp_dir_path}: {e}")
        
        logger.info(f"临时文件清理完成，共删除 {files_deleted} 个文件")
        
        return {
            'files_deleted': files_deleted,
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")
        raise

# 配置定期任务
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    # 每天凌晨2点清理过期任务
    'cleanup-expired-tasks': {
        'task': 'tasks.cleanup.cleanup_all_expired_tasks',
        'schedule': crontab(hour=2, minute=0),
    },
    # 每小时清理临时文件
    'cleanup-temp-files': {
        'task': 'tasks.cleanup.cleanup_temp_files',
        'schedule': crontab(minute=0),
    },
}

celery_app.conf.timezone = 'UTC'