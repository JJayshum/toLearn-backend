from celery import Celery
import logging
import json
import os
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.ai_service import AIService
from services.manim_service import ManimService
from services.tts_service import TTSService
from services.video_composer import VideoComposer
from services.task_manager import TaskManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入Celery应用
from celery_config import celery_app

@celery_app.task(bind=True, name='tasks.video_generation.generate_video_task')
def generate_video_task(self, task_id: str, question: str, solution_data: Dict, video_config: Dict):
    """
    新的三步骤视频生成任务
    使用AIService的完整三步骤流程：
    步骤1: 生成分句旁白并TTS合成，记录时间点
    步骤2: 基于旁白内容生成连续动画
    步骤3: 精确同步音视频
    """
    task_manager = TaskManager()
    
    try:
        logger.info(f"任务 {task_id}: 开始新的三步骤视频生成")
        
        # 更新任务状态为处理中
        task_manager.update_task_status(task_id, "processing", 10)
        
        # 构造请求数据格式
        request_data = {
            'question': question,
            'solution_data': {
                'detailed_solution': solution_data.get('detailed_solution', str(solution_data))
            }
        }
        
        # 使用AIService的完整三步骤流程
        logger.info(f"任务 {task_id}: 执行AIService完整三步骤流程")
        ai_service = AIService()
        
        # 执行完整的三步骤视频生成流程
        ai_result = ai_service.generate_complete_video_with_narration(task_id, request_data)
        
        if not ai_result.get('success'):
            raise Exception(f"AIService三步骤流程失败: {ai_result.get('error')}")
        
        logger.info(f"任务 {task_id}: AIService三步骤流程完成")
        task_manager.update_task_status(task_id, "processing", 80)
        
        # 获取AI生成的最终视频路径
        final_video_path = ai_result.get('final_video')
        
        if not final_video_path or not os.path.exists(final_video_path):
            # 如果AI流程没有生成最终视频，使用传统方法进行后续处理
            logger.info(f"任务 {task_id}: 使用传统方法进行视频合成")
            
            step1_result = ai_result.get('step1_result', {})
            step2_result = ai_result.get('step2_result', {})
            
            # 使用ManimService和VideoComposer进行渲染和合成
            manim_service = ManimService()
            video_composer = VideoComposer()
            
            # 渲染动画
            manim_result = manim_service.render_continuous_animation(
                task_id,
                step2_result,
                step1_result.get('sentence_timings', [])
            )
            
            # 精确同步合成最终视频
            final_video_path = video_composer.compose_video_with_precise_sync(
                task_id=task_id,
                animation_segments=manim_result['animation_segments'],
                merged_audio_file=step1_result.get('audio_track_file'),
                sentence_timings=step1_result.get('sentence_timings', []),
                total_duration=step1_result.get('total_duration', 0)
            )
        
        logger.info(f"任务 {task_id}: 视频生成完成，路径: {final_video_path}")
        task_manager.update_task_status(task_id, "processing", 90)
        
        # 获取视频信息
        video_composer = VideoComposer()
        video_info = video_composer.get_video_info(final_video_path)
        
        # 存储最终视频
        logger.info(f"任务 {task_id}: 存储最终视频")
        storage_result = task_manager.store_final_video(
            task_id=task_id,
            temp_video_path=final_video_path,
            video_info=video_info
        )
        
        # 清理临时文件
        logger.info(f"任务 {task_id}: 开始清理临时文件")
        try:
            video_composer.cleanup_temp_files(task_id)
            if 'manim_service' in locals():
                manim_service.cleanup_temp_files(task_id)
            # 清理AI服务的临时文件
            ai_service.cleanup_temp_files(task_id)
            logger.info(f"任务 {task_id}: 临时文件清理完成")
        except Exception as e:
            logger.warning(f"任务 {task_id}: 清理临时文件失败: {e}")
        
        logger.info(f"任务 {task_id}: 视频生成完成")
        
        return {
            'status': 'completed',
            'video_url': storage_result['video_url'],
            'thumbnail_url': storage_result['thumbnail_url'],
            'duration': storage_result['duration'],
            'file_size': storage_result['file_size']
        }
        
    except Exception as e:
        error_message = f"视频生成失败: {str(e)}"
        logger.error(f"任务 {task_id}: {error_message}")
        
        # 更新任务状态为失败
        task_manager.update_task_status(
            task_id=task_id,
            status="failed",
            error_message=error_message
        )
        
        # 清理可能的临时文件
        logger.info(f"任务 {task_id}: 清理失败任务的临时文件")
        try:
            video_composer = VideoComposer()
            manim_service = ManimService()
            ai_service = AIService()
            video_composer.cleanup_temp_files(task_id)
            manim_service.cleanup_temp_files(task_id)
            ai_service.cleanup_temp_files(task_id)
        except:
            pass
        
        # 重新抛出异常以便Celery处理重试
        raise

@celery_app.task(name='tasks.video_generation.test_task')
def test_task(message: str):
    """
    测试任务
    """
    logger.info(f"测试任务执行: {message}")
    return f"任务完成: {message}"

# 任务重试配置
generate_video_task.retry_kwargs = {
    'max_retries': 3,
    'countdown': 60  # 重试间隔60秒
}