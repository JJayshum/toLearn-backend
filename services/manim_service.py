import tempfile
import os
import shutil
import logging
import subprocess
import json
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ManimService:
    def __init__(self):
        logger.info("使用本地Manim渲染方案")
    

    

    
    def render_video(self, task_id: str, manim_code: str, force_method: str = None) -> Dict:
        """
        渲染Manim视频并捕获同步标记
        使用本地渲染方案
        
        Args:
            task_id: 任务ID
            manim_code: Manim代码
            force_method: 强制指定渲染方法 (仅支持 'local')
        """
        # 如果强制指定了渲染方法
        if force_method:
            logger.info(f"任务 {task_id}: 强制使用 {force_method} 渲染方法")
            
            if force_method == 'local':
                return self._render_local(task_id, manim_code)
            else:
                raise Exception(f"不支持的渲染方法: {force_method}")
        
        # 统一使用本地渲染
        logger.info(f"任务 {task_id}: 使用本地渲染")
        return self._render_local(task_id, manim_code)
    
    def render_continuous_animation(self, task_id: str, animation_result: dict, sentence_timings: list) -> Dict:
        """
        渲染连续动画，为每个句子生成对应的动画片段
        
        Args:
            task_id: 任务ID
            animation_result: AI生成的动画结果，包含manim_code和animation_methods
            sentence_timings: 句子时间信息列表
            
        Returns:
            {
                "animation_segments": [...],  # 动画片段信息
                "total_duration": 总时长,
                "success": True/False
            }
        """
        try:
            manim_code = animation_result['manim_code']
            animation_methods = animation_result['animation_methods']
            
            logger.info(f"任务 {task_id}: 开始渲染连续动画，共 {len(animation_methods)} 个片段")
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix=f"manim_{task_id}_")
            
            # 为每个动画方法生成独立的动画片段
            animation_segments = []
            
            for i, (method_info, timing_info) in enumerate(zip(animation_methods, sentence_timings)):
                try:
                    # 生成单个动画片段
                    segment_result = self._render_animation_segment(
                        task_id, 
                        temp_dir, 
                        manim_code, 
                        method_info, 
                        timing_info, 
                        i
                    )
                    
                    if segment_result['success']:
                        animation_segments.append(segment_result)
                        logger.info(f"任务 {task_id}: 动画片段 {i+1}/{len(animation_methods)} 渲染完成")
                    else:
                        logger.error(f"任务 {task_id}: 动画片段 {i+1} 渲染失败")
                        
                except Exception as e:
                    logger.error(f"任务 {task_id}: 渲染动画片段 {i+1} 时发生错误: {e}")
                    continue
            
            # 计算总时长
            total_duration = sum(timing['duration'] for timing in sentence_timings)
            
            logger.info(f"任务 {task_id}: 连续动画渲染完成，共 {len(animation_segments)} 个有效片段")
            
            return {
                'animation_segments': animation_segments,
                'total_duration': total_duration,
                'success': len(animation_segments) > 0,
                'temp_dir': temp_dir
            }
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 连续动画渲染失败: {e}")
            return {
                'animation_segments': [],
                'total_duration': 0,
                'success': False,
                'error': str(e)
            }
    
    def _render_animation_segment(self, task_id: str, temp_dir: str, manim_code: str, method_info: dict, timing_info: dict, segment_index: int) -> Dict:
        """
        渲染单个动画片段
        """
        try:
            method_name = method_info['method_name']
            target_duration = timing_info['duration']
            
            # 修改Manim代码，只渲染指定的方法
            modified_code = self._modify_manim_code_for_segment(manim_code, method_name, target_duration)
            
            # 创建Python文件
            segment_file = os.path.join(temp_dir, f"segment_{segment_index}.py")
            with open(segment_file, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            
            # 渲染动画片段
            output_dir = os.path.join(temp_dir, f"output_{segment_index}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 从代码中提取类名
            class_name = self._extract_class_name(manim_code)
            if not class_name:
                class_name = 'MathScene'  # 默认类名
            
            # 执行Manim渲染命令
            cmd = [
                'manim', 
                segment_file, 
                class_name,
                '--output_file', f'segment_{segment_index}',
                '--media_dir', output_dir,
                '-q', 'm',  # 使用中等质量
                '--disable_caching'
            ]
            
            logger.info(f"任务 {task_id}: 执行渲染命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=temp_dir
            )
            
            if result.returncode == 0:
                # 查找输出的视频文件
                video_file = self._find_segment_output_file(output_dir, segment_index)
                
                if video_file and os.path.exists(video_file):
                    # 获取实际视频时长
                    actual_duration = self._get_video_duration(video_file)
                    
                    # 如果时长不匹配，进行调整
                    if abs(actual_duration - target_duration) > 0.1:  # 允许0.1秒误差
                        adjusted_file = self._adjust_video_duration(video_file, target_duration, segment_index, temp_dir)
                        if adjusted_file:
                            video_file = adjusted_file
                            actual_duration = target_duration
                    
                    return {
                        'success': True,
                        'video_file': video_file,
                        'method_name': method_name,
                        'target_duration': target_duration,
                        'actual_duration': actual_duration,
                        'segment_index': segment_index,
                        'start_time': timing_info['start_time'],
                        'end_time': timing_info['end_time']
                    }
                else:
                    logger.error(f"任务 {task_id}: 找不到片段 {segment_index} 的输出视频文件")
                    return {'success': False, 'error': '找不到输出视频文件'}
            else:
                logger.error(f"任务 {task_id}: 片段 {segment_index} 渲染失败: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"任务 {task_id}: 渲染片段 {segment_index} 时发生异常: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_class_name(self, manim_code: str) -> str:
        """
        从Manim代码中提取类名
        """
        import re
        
        # 查找类定义
        class_pattern = r'class\s+(\w+)\s*\([^)]*Scene[^)]*\):'
        match = re.search(class_pattern, manim_code)
        
        if match:
            return match.group(1)
        
        # 如果没有找到，尝试更宽泛的匹配
        class_pattern = r'class\s+(\w+)\s*\([^)]*\):'
        match = re.search(class_pattern, manim_code)
        
        if match:
            return match.group(1)
        
        return None
    
    def _modify_manim_code_for_segment(self, manim_code: str, method_name: str, target_duration: float) -> str:
        """
        修改Manim代码，只执行指定的动画方法
        """
        modified_code = manim_code
        
        # 直接调用方法，不传递任何模板参数
        method_call = f"self.{method_name}()"
        
        # 查找construct方法并替换
        construct_pattern = r'def construct\(self\):[\s\S]*?(?=\n    def|\n\n|\Z)'
        
        new_construct = f"""def construct(self):
        # 只执行指定的动画方法
        {method_call}
        
        # 确保动画时长匹配
        self.wait({target_duration})
"""
        
        if re.search(construct_pattern, modified_code):
            modified_code = re.sub(construct_pattern, new_construct, modified_code)
        else:
            # 如果没有找到construct方法，在类定义后添加
            # 查找类定义的结束位置
            lines = modified_code.split('\n')
            class_line_index = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('class ') and ':' in line:
                    class_line_index = i
                    break
            
            if class_line_index != -1:
                # 在类定义后插入construct方法，需要添加适当的缩进
                indented_construct = '\n'.join(['    ' + line if line.strip() else line for line in new_construct.split('\n')])
                lines.insert(class_line_index + 1, indented_construct)
                modified_code = '\n'.join(lines)
            else:
                # 如果找不到类定义，直接添加到末尾
                modified_code += '\n' + new_construct
        
        return modified_code
    
    def _find_segment_output_file(self, output_dir: str, segment_index: int) -> Optional[str]:
        """
        查找动画片段的输出文件
        """
        # 查找可能的输出文件位置
        possible_paths = [
            os.path.join(output_dir, 'videos', '1080p60', f'segment_{segment_index}.mp4'),
            os.path.join(output_dir, 'videos', 'MathScene', '1080p60', f'segment_{segment_index}.mp4'),
            os.path.join(output_dir, f'segment_{segment_index}.mp4'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果没有找到，尝试查找任何mp4文件
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.mp4'):
                    return os.path.join(root, file)
        
        return None
    
    def _adjust_video_duration(self, video_file: str, target_duration: float, segment_index: int, temp_dir: str) -> Optional[str]:
        """
        调整视频时长以匹配目标时长
        """
        try:
            output_file = os.path.join(temp_dir, f'adjusted_segment_{segment_index}.mp4')
            
            # 使用ffmpeg调整视频时长
            cmd = [
                'ffmpeg', '-i', video_file,
                '-t', str(target_duration),  # 截取到目标时长
                '-c', 'copy',  # 复制编码，避免重新编码
                '-y',  # 覆盖输出文件
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_file):
                return output_file
            else:
                logger.error(f"调整视频时长失败: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"调整视频时长时发生异常: {e}")
            return None

    def _find_output_file(self, temp_dir: str) -> str:
        """
        查找Manim输出的视频文件
        优先查找完整的最终视频，如果没有则合并partial片段
        """
        # 首先查找标准的完整输出文件
        standard_paths = [
            os.path.join(temp_dir, "videos", "scene", "1080p60", "MathVideoScene.mp4"),
            os.path.join(temp_dir, "media", "videos", "scene", "1080p60", "MathVideoScene.mp4"),
            os.path.join(temp_dir, "videos", "scene", "1080p60", "output.mp4"),
            os.path.join(temp_dir, "media", "videos", "scene", "1080p60", "output.mp4"),
            os.path.join(temp_dir, "output.mp4"),
            os.path.join(temp_dir, "MathVideoScene.mp4")
        ]
        
        for path in standard_paths:
            if os.path.exists(path):
                logger.info(f"找到标准输出文件: {path}")
                return path
        
        # 如果没有找到标准输出文件，查找partial目录中的片段
        partial_dir = os.path.join(temp_dir, "partial")
        if os.path.exists(partial_dir):
            partial_files = []
            for file in os.listdir(partial_dir):
                if file.endswith('.mp4'):
                    file_path = os.path.join(partial_dir, file)
                    # 检查文件是否有效（大小大于100字节）
                    if os.path.getsize(file_path) > 100:
                        partial_files.append(file_path)
            
            if partial_files:
                logger.info(f"找到 {len(partial_files)} 个partial片段，需要合并")
                # 按文件名排序以确保正确的顺序
                partial_files.sort()
                
                # 合并所有partial片段
                merged_file = self._merge_partial_videos(temp_dir, partial_files)
                if merged_file:
                    return merged_file
        
        # 最后的备选方案：递归搜索所有.mp4文件
        logger.warning("未找到标准输出文件和partial片段，进行递归搜索")
        for root, dirs, files in os.walk(temp_dir):
            # 跳过__pycache__目录
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for file in files:
                if file.endswith('.mp4'):
                    full_path = os.path.join(root, file)
                    # 检查文件是否有效
                    if os.path.getsize(full_path) > 100:
                        logger.info(f"找到视频文件: {full_path}")
                        return full_path
        
        return None
    
    def _merge_partial_videos(self, temp_dir: str, partial_files: List[str]) -> str:
        """
        合并partial视频片段为完整视频
        """
        try:
            import subprocess
            
            # 创建合并后的文件路径
            merged_file = os.path.join(temp_dir, "merged_output.mp4")
            
            # 如果只有一个文件，直接复制
            if len(partial_files) == 1:
                logger.info("只有一个partial文件，直接使用")
                shutil.copy2(partial_files[0], merged_file)
                return merged_file
            
            # 创建ffmpeg输入文件列表
            input_list_file = os.path.join(temp_dir, "input_list.txt")
            with open(input_list_file, 'w') as f:
                for file_path in partial_files:
                    # 使用相对路径避免路径问题
                    rel_path = os.path.relpath(file_path, temp_dir)
                    f.write(f"file '{rel_path}'\n")
            
            logger.info(f"准备合并 {len(partial_files)} 个视频片段")
            
            # 使用ffmpeg合并视频
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', input_list_file,
                '-c', 'copy',  # 直接复制流，不重新编码
                '-y',  # 覆盖输出文件
                merged_file
            ]
            
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"视频片段合并成功: {merged_file}")
                # 验证合并后的文件
                if os.path.exists(merged_file) and os.path.getsize(merged_file) > 1000:
                    return merged_file
                else:
                    logger.error("合并后的文件无效")
                    return None
            else:
                logger.error(f"视频合并失败 (返回码: {result.returncode}): {result.stderr}")
                if result.stdout:
                    logger.error(f"ffmpeg stdout: {result.stdout}")
                # 如果合并失败，尝试重新编码合并
                return self._merge_partial_videos_with_reencoding(temp_dir, partial_files)
                
        except Exception as e:
            logger.error(f"合并partial视频失败: {e}")
            # 如果合并失败，返回最长的片段
            return self._find_longest_partial(partial_files)
    
    def _merge_partial_videos_with_reencoding(self, temp_dir: str, partial_files: List[str]) -> str:
        """
        使用重新编码的方式合并视频片段（备选方案）
        """
        try:
            import subprocess
            
            merged_file = os.path.join(temp_dir, "merged_reencoded.mp4")
            
            # 构建ffmpeg命令，重新编码合并
            cmd = ['ffmpeg', '-y']
            
            # 添加所有输入文件
            for file_path in partial_files:
                cmd.extend(['-i', file_path])
            
            # 添加filter_complex来连接视频
            filter_parts = []
            for i in range(len(partial_files)):
                filter_parts.append(f"[{i}:v][{i}:a]")
            
            filter_complex = f"{''.join(filter_parts)}concat=n={len(partial_files)}:v=1:a=1[outv][outa]"
            
            cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[outv]', '-map', '[outa]',
                '-c:v', 'libx264', '-c:a', 'aac',
                merged_file
            ])
            
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0 and os.path.exists(merged_file):
                logger.info(f"重新编码合并成功: {merged_file}")
                return merged_file
            else:
                logger.error(f"重新编码合并失败 (返回码: {result.returncode}): {result.stderr}")
                if result.stdout:
                    logger.error(f"ffmpeg stdout: {result.stdout}")
                return self._find_longest_partial(partial_files)
                
        except Exception as e:
            logger.error(f"重新编码合并失败: {e}")
            return self._find_longest_partial(partial_files)
    
    def _find_longest_partial(self, partial_files: List[str]) -> str:
        """
        找到最长的partial视频片段作为备选
        """
        if not partial_files:
            return None
        
        longest_file = None
        max_duration = 0
        
        for file_path in partial_files:
            try:
                duration = self._get_video_duration(file_path)
                if duration > max_duration:
                    max_duration = duration
                    longest_file = file_path
            except Exception as e:
                logger.warning(f"无法获取文件时长 {file_path}: {e}")
        
        if longest_file:
            logger.info(f"使用最长的partial片段: {longest_file} (时长: {max_duration:.2f}s)")
            return longest_file
        else:
            logger.warning("无法确定最长片段，使用第一个")
            return partial_files[0]
    
    def _parse_markers_from_logs(self, logs: str) -> Dict[str, float]:
        """
        从容器日志中解析标记时间戳
        """
        markers = {}
        lines = logs.split('\n')
        
        # 首先尝试解析CAPTURED格式的标记
        for line in lines:
            if 'CAPTURED:' in line and 'at' in line:
                try:
                    # 解析格式: "CAPTURED: ##MARKER_1## at 2.345s"
                    parts = line.split('CAPTURED:')[1].strip()
                    marker_part, time_part = parts.split(' at ')
                    marker = marker_part.strip()
                    time_str = time_part.replace('s', '').strip()
                    timestamp = float(time_str)
                    
                    # 提取标记名称 (去掉##)
                    if marker.startswith('##') and marker.endswith('##'):
                        marker_name = marker[2:-2]  # 去掉前后的##
                        markers[marker_name] = timestamp
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析标记失败: {line}, 错误: {e}")
                    continue
        
        # 如果没有找到CAPTURED格式，尝试直接解析MARKER输出
        if not markers:
            logger.info("未找到CAPTURED格式标记，尝试解析原始MARKER输出")
            marker_lines = []
            for line in lines:
                if line.strip().startswith('##MARKER_') and line.strip().endswith('##'):
                    marker_lines.append(line.strip())
            
            # 为每个找到的MARKER分配时间戳（假设按顺序出现）
            for i, marker_line in enumerate(marker_lines):
                marker_name = marker_line[2:-2]  # 去掉前后的##
                # 使用8秒间隔作为默认时间戳
                markers[marker_name] = i * 8.0
                logger.info(f"解析原始标记: {marker_name} -> {i * 8.0}s")
        
        # 如果仍然没有找到标记，生成默认的
        if not markers:
            # 根据代码中的MARKER数量生成标记
            marker_count = max(8, len([line for line in logs.split('\n') if '##MARKER_' in line]))
            markers = {}
            for i in range(1, marker_count + 1):
                markers[f'MARKER_{i}'] = (i - 1) * 8.0
            logger.warning(f"未找到标记，生成{marker_count}个默认时间戳（8秒间隔）")
        
        logger.info(f"解析到的标记: {markers}")
        return markers
    
    def _get_video_duration(self, video_file: str) -> float:
        """
        获取视频时长
        """
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', video_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                return duration
            else:
                logger.warning(f"无法获取视频时长: {result.stderr}")
                return 30.0  # 最小默认时长
        except Exception as e:
            logger.warning(f"获取视频时长失败: {e}")
            return 30.0  # 最小默认时长
    
    def _extract_timestamps(self, manim_code: str) -> List[float]:
        """
        从Manim代码中提取时间戳（备用方法）
        """
        timestamps = []
        lines = manim_code.split('\n')
        
        for line in lines:
            if '# TIMESTAMP:' in line:
                try:
                    time_str = line.split('# TIMESTAMP:')[1].strip()
                    timestamp = float(time_str)
                    timestamps.append(timestamp)
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析时间戳失败: {line}, 错误: {e}")
                    continue
        
        # 如果没有找到时间戳，生成默认的
        if not timestamps:
            # 根据代码中的MARKER数量生成更多时间戳
            marker_count = manim_code.count('##MARKER_')
            if marker_count > 0:
                # 为每个MARKER生成时间戳，间隔约8秒
                timestamps = [i * 8.0 for i in range(marker_count)]
            else:
                timestamps = [0, 5, 10, 15, 20]  # 默认时间戳
        
        return sorted(timestamps)
    
    def _render_local(self, task_id: str, manim_code: str) -> Dict:
        """
        本地渲染方案：执行真正的Manim代码并捕获标记时间戳
        """
        logger.info(f"任务 {task_id}: 使用本地渲染方案")
        
        # 创建临时目录
        temp_dir = f"/tmp/manim_{task_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # 清理可能存在的__pycache__目录
            pycache_dir = os.path.join(temp_dir, "__pycache__")
            if os.path.exists(pycache_dir):
                shutil.rmtree(pycache_dir)
                logger.info(f"任务 {task_id}: 已清理__pycache__目录")
            
            # 创建场景文件
            scene_file = os.path.join(temp_dir, "scene.py")
            wrapper_script = os.path.join(temp_dir, "run_manim.py")
            
            # 写入Manim代码
            with open(scene_file, 'w', encoding='utf-8') as f:
                f.write(manim_code)
            
            # 创建包装脚本来捕获标记和时间戳
            wrapper_code = f'''
import sys
import time
import os
import subprocess
from io import StringIO
import contextlib
import builtins

# 重定向print输出以捕获标记
markers = {{}}
original_print = builtins.print
start_time = None

def capture_print(*args, **kwargs):
    global start_time
    if start_time is None:
        start_time = time.time()
    
    # 调用原始print函数
    original_print(*args, **kwargs)
    
    # 检查是否包含MARKER
    if args and isinstance(args[0], str) and "##MARKER_" in args[0]:
        marker = args[0].strip()
        current_time = time.time() - start_time
        markers[marker] = current_time
        original_print(f"CAPTURED: {{marker}} at {{current_time:.3f}}s")

# 替换print函数
builtins.print = capture_print

# 设置环境变量
os.environ["MANIMGL_LOG_LEVEL"] = "WARNING"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# 清理可能存在的__pycache__目录
import shutil
pycache_dir = "{temp_dir}/__pycache__"
if os.path.exists(pycache_dir):
    shutil.rmtree(pycache_dir)
    original_print(f"清理了__pycache__目录: {{pycache_dir}}")

try:
    # 导入Manim
    sys.path.insert(0, "{temp_dir}")
    from scene import MathVideoScene
    from manim import *
    
    # 创建场景并渲染
    config.media_dir = "{temp_dir}/media"
    config.video_dir = "{temp_dir}/videos"
    config.images_dir = "{temp_dir}/images"
    config.text_dir = "{temp_dir}/text"
    config.tex_dir = "{temp_dir}/tex"
    config.partial_movie_dir = "{temp_dir}/partial"
    
    scene = MathVideoScene()
    scene.construct()
    
    # 输出标记信息
    original_print("=== MARKERS ===")
    for marker, timestamp in markers.items():
        original_print(f"{{marker}}: {{timestamp:.3f}}")
    original_print("=== END MARKERS ===")
    
except Exception as e:
    original_print(f"渲染错误: {{e}}")
    import traceback
    traceback.print_exc()
'''
            
            with open(wrapper_script, 'w', encoding='utf-8') as f:
                f.write(wrapper_code)
            
            logger.info(f"任务 {task_id}: Manim代码和包装脚本已写入")
            
            # 执行Manim渲染
            try:
                import subprocess
                
                # 优先使用包装脚本来确保时间戳捕获
                logger.info(f"任务 {task_id}: 使用包装脚本执行Manim渲染")
                
                # 设置环境变量防止生成__pycache__目录
                env = os.environ.copy()
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                
                result = subprocess.run(
                    ['python3', wrapper_script],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env
                )
                logs = result.stdout + result.stderr
                logger.info(f"任务 {task_id}: 包装脚本执行日志: {logs}")  # 显示完整日志
                
                # 如果包装脚本失败，直接抛出错误
                if result.returncode != 0:
                    logger.error(f"任务 {task_id}: 包装脚本执行失败: {logs}")
                    raise Exception(f"Manim渲染失败: {logs}")
                
                # 再次清理可能生成的__pycache__目录
                pycache_dir = os.path.join(temp_dir, "__pycache__")
                if os.path.exists(pycache_dir):
                    shutil.rmtree(pycache_dir)
                    logger.info(f"任务 {task_id}: 执行后清理__pycache__目录")
                
                # 查找输出文件
                output_file = self._find_output_file(temp_dir)
                if not output_file:
                    logger.warning(f"未找到Manim输出文件，目录内容: {os.listdir(temp_dir)}")
                    # 递归查找所有mp4文件
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('.mp4'):
                                output_file = os.path.join(root, file)
                                logger.info(f"找到视频文件: {output_file}")
                                break
                        if output_file:
                            break
                
                if not output_file:
                    raise Exception(f"未找到Manim输出文件")
                
                # 从日志中解析标记时间戳
                timestamps = self._parse_markers_from_logs(logs)
                
                # 移动到临时位置
                final_output_path = f"/tmp/manim_video_{task_id}.mp4"
                shutil.move(output_file, final_output_path)
                
                # 获取实际视频时长
                actual_duration = self._get_video_duration(final_output_path)
                
                logger.info(f"任务 {task_id}: 本地视频生成成功: {final_output_path}")
                
                return {
                    "video_file": final_output_path,
                    "timestamps": timestamps,
                    "duration": actual_duration
                }
                
            except subprocess.TimeoutExpired:
                logger.error(f"任务 {task_id}: Manim执行超时")
                raise Exception("Manim渲染超时")
            except Exception as e:
                logger.error(f"任务 {task_id}: Manim执行失败: {e}")
                raise
                
        except ImportError as e:
            logger.error(f"缺少必要的库: {e}")
            raise Exception(f"本地渲染失败，缺少依赖库: {e}")
        except Exception as e:
            logger.error(f"任务 {task_id}: 本地渲染失败: {e}")
            raise
        finally:
            # 清理临时文件 - 暂时注释掉以便调试
            # try:
            #     shutil.rmtree(temp_dir, ignore_errors=True)
            # except Exception as e:
            #     logger.warning(f"清理临时目录失败: {e}")
            logger.info(f"任务 {task_id}: 保留临时目录用于调试: {temp_dir}")
    

    

    
    def cleanup_temp_files(self, task_id: str):
        """
        清理临时文件
        """
        logger.info(f"任务 {task_id}: 开始清理Manim临时文件")
        
        temp_patterns = [
            f"/tmp/manim_{task_id}_*",
            f"/tmp/manim_temp_{task_id}",
            f"/tmp/manim_video_{task_id}.mp4"
        ]
        
        import glob
        import shutil
        files_deleted = 0
        
        for pattern in temp_patterns:
            files = glob.glob(pattern)
            for file_or_dir in files:
                try:
                    if os.path.exists(file_or_dir):
                        if os.path.isdir(file_or_dir):
                            shutil.rmtree(file_or_dir)
                            files_deleted += 1
                            logger.debug(f"删除临时目录: {file_or_dir}")
                        else:
                            os.remove(file_or_dir)
                            files_deleted += 1
                            logger.debug(f"删除临时文件: {file_or_dir}")
                except Exception as e:
                    logger.warning(f"删除临时文件/目录失败 {file_or_dir}: {e}")
        
        logger.info(f"任务 {task_id}: Manim临时文件清理完成，共删除 {files_deleted} 个文件/目录")