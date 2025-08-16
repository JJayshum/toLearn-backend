import os
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from sqlalchemy.orm import Session
from database import get_db, VideoGenerationTask
import json

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self):
        self.video_storage_path = "/home/devbox/project/videos/final"
        self.cdn_base_url = os.getenv("CDN_BASE_URL", "https://crqrwmioxmmm.sealosgzg.site/videos")
    
    def create_task(self, task_id: str, question: str, solution_data: Dict, video_config: Dict) -> VideoGenerationTask:
        """
        创建新的视频生成任务
        """
        db = next(get_db())
        try:
            task = VideoGenerationTask(
                task_id=task_id,
                question=question,
                solution_data=json.dumps(solution_data, ensure_ascii=False),
                video_config=json.dumps(video_config, ensure_ascii=False),
                status="pending",
                progress=0,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=7)  # 7天后过期
            )
            
            db.add(task)
            db.commit()
            db.refresh(task)
            
            logger.info(f"任务 {task_id}: 创建成功")
            return task
            
        except Exception as e:
            db.rollback()
            logger.error(f"任务 {task_id}: 创建失败: {e}")
            raise
        finally:
            db.close()
    
    def update_task_status(self, task_id: str, status: str, progress: int = None, 
                          error_message: str = None, **kwargs) -> bool:
        """
        更新任务状态
        """
        db = next(get_db())
        try:
            task = db.query(VideoGenerationTask).filter(
                VideoGenerationTask.task_id == task_id
            ).first()
            
            if not task:
                logger.error(f"任务 {task_id}: 未找到")
                return False
            
            # 更新基本状态
            task.status = status
            if progress is not None:
                task.progress = progress
            if error_message:
                task.error_message = error_message
            
            # 更新时间戳
            if status == "processing" and not task.started_at:
                task.started_at = datetime.utcnow()
            elif status in ["completed", "failed"]:
                task.completed_at = datetime.utcnow()
            
            # 更新其他字段
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            db.commit()
            logger.info(f"任务 {task_id}: 状态更新为 {status}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"任务 {task_id}: 状态更新失败: {e}")
            return False
        finally:
            db.close()
    
    def get_task(self, task_id: str) -> Optional[VideoGenerationTask]:
        """
        获取任务信息
        """
        db = next(get_db())
        try:
            task = db.query(VideoGenerationTask).filter(
                VideoGenerationTask.task_id == task_id
            ).first()
            return task
        finally:
            db.close()
    
    def store_final_video(self, task_id: str, temp_video_path: str, video_info: Dict) -> Dict:
        """
        存储最终视频文件
        """
        try:
            # 确保存储目录存在
            os.makedirs(self.video_storage_path, exist_ok=True)
            
            # 生成最终文件名
            final_filename = f"video_{task_id}.mp4"
            final_path = os.path.join(self.video_storage_path, final_filename)
            
            # 移动文件到最终位置
            shutil.move(temp_video_path, final_path)
            
            # 生成访问URL
            video_url = f"{self.cdn_base_url}/{final_filename}"
            
            # 生成缩略图（可选）
            thumbnail_url = self._generate_thumbnail(final_path, task_id)
            
            # 更新数据库
            self.update_task_status(
                task_id=task_id,
                status="completed",
                progress=100,
                video_url=video_url,
                thumbnail_url=thumbnail_url,
                duration=video_info.get('duration', 0),
                file_size=video_info.get('size', 0)
            )
            
            # 安排文件清理
            self._schedule_file_cleanup(task_id)
            
            logger.info(f"任务 {task_id}: 视频存储完成: {video_url}")
            
            return {
                'video_url': video_url,
                'thumbnail_url': thumbnail_url,
                'duration': video_info.get('duration', 0),
                'file_size': video_info.get('size', 0)
            }
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 视频存储失败: {e}")
            raise
    
    def _generate_thumbnail(self, video_path: str, task_id: str) -> Optional[str]:
        """
        生成视频缩略图
        """
        try:
            import ffmpeg
            
            thumbnail_filename = f"thumb_{task_id}.jpg"
            thumbnail_path = os.path.join(self.video_storage_path, thumbnail_filename)
            
            # 在视频的第2秒生成缩略图
            (
                ffmpeg
                .input(video_path, ss=2)
                .output(thumbnail_path, vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(quiet=True)
            )
            
            thumbnail_url = f"{self.cdn_base_url}/{thumbnail_filename}"
            logger.info(f"任务 {task_id}: 缩略图生成完成: {thumbnail_url}")
            return thumbnail_url
            
        except Exception as e:
            logger.warning(f"任务 {task_id}: 缩略图生成失败: {e}")
            return None
    
    def _schedule_file_cleanup(self, task_id: str):
        """
        安排文件清理（通过Celery延时任务）
        """
        try:
            from tasks.cleanup import cleanup_expired_files
            
            # 7天后清理文件
            cleanup_time = datetime.utcnow() + timedelta(days=7)
            cleanup_expired_files.apply_async(
                args=[task_id],
                eta=cleanup_time
            )
            
            logger.info(f"任务 {task_id}: 已安排文件清理，时间: {cleanup_time}")
            
        except Exception as e:
            logger.warning(f"任务 {task_id}: 安排文件清理失败: {e}")
    
    def get_task_status_response(self, task_id: str) -> Dict:
        """
        获取任务状态响应
        """
        task = self.get_task(task_id)
        
        if not task:
            return {
                'task_id': task_id,
                'status': 'not_found',
                'message': '任务不存在'
            }
        
        response = {
            'task_id': task.task_id,
            'status': task.status,
            'progress': task.progress,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None
        }
        
        # 根据状态添加不同信息
        if task.status == 'completed':
            response.update({
                'video_url': task.video_url,
                'thumbnail_url': task.thumbnail_url,
                'duration': task.duration,
                'file_size': task.file_size
            })
        elif task.status == 'failed':
            response['error_message'] = task.error_message
        elif task.status == 'processing':
            response['message'] = '视频生成中，请稍候...'
        elif task.status == 'pending':
            response['message'] = '任务排队中...'
        
        return response
    
    def cleanup_expired_tasks(self):
        """
        清理过期任务
        """
        db = next(get_db())
        try:
            expired_tasks = db.query(VideoGenerationTask).filter(
                VideoGenerationTask.expires_at < datetime.utcnow()
            ).all()
            
            for task in expired_tasks:
                try:
                    # 删除视频文件
                    if task.video_url:
                        filename = task.video_url.split('/')[-1]
                        file_path = os.path.join(self.video_storage_path, filename)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    # 删除缩略图
                    if task.thumbnail_url:
                        filename = task.thumbnail_url.split('/')[-1]
                        file_path = os.path.join(self.video_storage_path, filename)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    # 删除数据库记录
                    db.delete(task)
                    
                    logger.info(f"清理过期任务: {task.task_id}")
                    
                except Exception as e:
                    logger.error(f"清理过期任务失败 {task.task_id}: {e}")
            
            db.commit()
            logger.info(f"清理了 {len(expired_tasks)} 个过期任务")
            
        except Exception as e:
            db.rollback()
            logger.error(f"清理过期任务失败: {e}")
        finally:
            db.close()