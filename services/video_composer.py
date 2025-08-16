import ffmpeg
import os
import logging
from typing import List, Dict
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class VideoComposer:
    def __init__(self):
        self.temp_dir = "/tmp"
    

    
    def compose_video_with_precise_sync(self, task_id: str, animation_segments: List[Dict], merged_audio_file: str, sentence_timings: List[Dict], total_duration: float) -> str:
        """
        使用精确同步方式合成视频
        
        Args:
            task_id: 任务ID
            animation_segments: 动画片段列表
            merged_audio_file: 合并后的音频文件
            sentence_timings: 句子时间信息
            total_duration: 总时长
            
        Returns:
            最终视频文件路径
        """
        logger.info(f"任务 {task_id}: 开始精确同步视频合成")
        
        try:
            # 1. 验证动画片段和时间信息
            if len(animation_segments) != len(sentence_timings):
                logger.warning(f"任务 {task_id}: 动画片段数量({len(animation_segments)})与句子数量({len(sentence_timings)})不匹配")
            
            # 2. 处理动画片段时长匹配
            processed_segments = self._process_animation_segments(task_id, animation_segments, sentence_timings)
            
            # 3. 按时间顺序合并动画片段
            merged_video = self._merge_animation_segments_by_time(task_id, processed_segments, total_duration)
            
            # 4. 合成最终视频（音频+视频）
            final_video = self._merge_audio_video(task_id, merged_video, merged_audio_file)
            
            logger.info(f"任务 {task_id}: 精确同步视频合成完成: {final_video}")
            return final_video
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 精确同步视频合成失败: {e}")
            raise
    
    def _process_animation_segments(self, task_id: str, animation_segments: List[Dict], sentence_timings: List[Dict]) -> List[Dict]:
        """
        处理动画片段，确保时长匹配
        """
        processed_segments = []
        
        for i, (anim_segment, timing) in enumerate(zip(animation_segments, sentence_timings)):
            try:
                video_file = anim_segment['video_file']
                target_duration = timing['duration']
                actual_duration = anim_segment.get('actual_duration', target_duration)
                
                # 检查时长是否需要调整
                if abs(actual_duration - target_duration) > 0.1:  # 允许0.1秒误差
                    logger.info(f"任务 {task_id}: 调整片段 {i} 时长从 {actual_duration:.2f}s 到 {target_duration:.2f}s")
                    
                    if actual_duration > target_duration:
                        # 动画长于音频，压缩动画
                        adjusted_file = self._compress_video_duration(task_id, video_file, target_duration, i)
                    else:
                        # 动画短于音频，延长最后一帧
                        adjusted_file = self._extend_video_duration(task_id, video_file, target_duration, i)
                    
                    if adjusted_file:
                        video_file = adjusted_file
                        actual_duration = target_duration
                
                processed_segments.append({
                    'video_file': video_file,
                    'start_time': timing['start_time'],
                    'duration': actual_duration,
                    'end_time': timing['end_time'],
                    'segment_index': i
                })
                
            except Exception as e:
                logger.error(f"任务 {task_id}: 处理动画片段 {i} 失败: {e}")
                continue
        
        return processed_segments
    
    def _compress_video_duration(self, task_id: str, video_file: str, target_duration: float, segment_index: int) -> str:
        """
        压缩视频时长（加速播放）
        """
        try:
            output_file = f"/tmp/compressed_{task_id}_{segment_index}.mp4"
            
            # 获取原视频时长
            probe = ffmpeg.probe(video_file)
            original_duration = float(probe['streams'][0]['duration'])
            
            # 计算加速倍率
            speed_factor = original_duration / target_duration
            
            # 使用ffmpeg加速视频
            (
                ffmpeg
                .input(video_file)
                .filter('setpts', f'PTS/{speed_factor}')
                .output(output_file, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
            
            if os.path.exists(output_file):
                return output_file
            else:
                logger.error(f"任务 {task_id}: 压缩视频失败，输出文件不存在")
                return video_file
                
        except Exception as e:
            logger.error(f"任务 {task_id}: 压缩视频时长失败: {e}")
            return video_file
    
    def _extend_video_duration(self, task_id: str, video_file: str, target_duration: float, segment_index: int) -> str:
        """
        延长视频时长（保持最后一帧）
        """
        try:
            output_file = f"/tmp/extended_{task_id}_{segment_index}.mp4"
            
            # 获取原视频时长
            probe = ffmpeg.probe(video_file)
            original_duration = float(probe['streams'][0]['duration'])
            
            # 计算需要延长的时间
            extend_duration = target_duration - original_duration
            
            # 获取最后一帧作为静态图像
            last_frame_file = f"/tmp/last_frame_{task_id}_{segment_index}.png"
            (
                ffmpeg
                .input(video_file, ss=original_duration-0.1)  # 获取倒数第0.1秒的帧
                .output(last_frame_file, vframes=1)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 创建静态视频片段
            static_video_file = f"/tmp/static_{task_id}_{segment_index}.mp4"
            (
                ffmpeg
                .input(last_frame_file, loop=1, t=extend_duration)
                .output(static_video_file, vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 合并原视频和静态片段
            (
                ffmpeg
                .concat(
                    ffmpeg.input(video_file),
                    ffmpeg.input(static_video_file)
                )
                .output(output_file, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 清理临时文件
            for temp_file in [last_frame_file, static_video_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            if os.path.exists(output_file):
                return output_file
            else:
                logger.error(f"任务 {task_id}: 延长视频失败，输出文件不存在")
                return video_file
                
        except Exception as e:
            logger.error(f"任务 {task_id}: 延长视频时长失败: {e}")
            return video_file
    
    def _merge_animation_segments_by_time(self, task_id: str, processed_segments: List[Dict], total_duration: float) -> str:
        """
        按时间顺序合并动画片段
        """
        try:
            if not processed_segments:
                # 如果没有动画片段，创建黑屏视频
                return self._create_placeholder_video(task_id, total_duration)
            
            # 按开始时间排序
            processed_segments.sort(key=lambda x: x['start_time'])
            
            # 创建输入文件列表
            input_files = []
            current_time = 0.0
            
            for segment in processed_segments:
                start_time = segment['start_time']
                
                # 如果有时间间隔，添加黑屏
                if start_time > current_time:
                    gap_duration = start_time - current_time
                    if gap_duration > 0.1:  # 只有间隔大于0.1秒才添加黑屏
                        black_video = self._create_black_video(task_id, gap_duration, len(input_files))
                        input_files.append(ffmpeg.input(black_video))
                
                # 添加动画片段
                input_files.append(ffmpeg.input(segment['video_file']))
                current_time = segment['end_time']
            
            # 如果最后还有剩余时间，添加黑屏
            if current_time < total_duration:
                remaining_duration = total_duration - current_time
                if remaining_duration > 0.1:
                    black_video = self._create_black_video(task_id, remaining_duration, len(input_files))
                    input_files.append(ffmpeg.input(black_video))
            
            # 合并所有片段
            output_file = f"/tmp/merged_animation_{task_id}.mp4"
            
            if len(input_files) == 1:
                # 只有一个片段，标准化分辨率和帧率
                try:
                    (
                        input_files[0]
                        .output(output_file, 
                               vcodec='libx264', 
                               acodec='aac',
                               vf='scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2',
                               r=30)
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as e:
                    logger.error(f"任务 {task_id}: ffmpeg单片段处理失败: {e.stderr.decode() if e.stderr else str(e)}")
                    raise
            else:
                # 多个片段，先标准化每个片段的分辨率和帧率，再合并
                try:
                    # 标准化所有输入片段
                    normalized_inputs = []
                    for i, input_file in enumerate(input_files):
                        normalized_inputs.append(
                            input_file.video
                            .filter('scale', 1280, 720, force_original_aspect_ratio='decrease')
                            .filter('pad', 1280, 720, '(ow-iw)/2', '(oh-ih)/2')
                            .filter('fps', fps=30)
                        )
                    
                    (
                        ffmpeg
                        .concat(*normalized_inputs, v=1, a=0)
                        .output(output_file, vcodec='libx264', acodec='aac')
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as e:
                    logger.error(f"任务 {task_id}: ffmpeg多片段合并失败: {e.stderr.decode() if e.stderr else str(e)}")
                    raise
            
            return output_file
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 合并动画片段失败: {e}")
            # 返回备用黑屏视频
            return self._create_placeholder_video(task_id, total_duration)
    
    def _create_black_video(self, task_id: str, duration: float, index: int) -> str:
        """
        创建黑屏视频片段
        """
        output_file = f"/tmp/black_{task_id}_{index}.mp4"
        
        try:
            (
                ffmpeg
                .input('color=black:size=1280x720:duration={}:rate=30'.format(duration), f='lavfi')
                .output(output_file, vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            return output_file
        except ffmpeg.Error as e:
            logger.error(f"创建黑屏视频失败 (ffmpeg): {e.stderr.decode() if e.stderr else str(e)}")
            # 返回备用方案
            return self._create_placeholder_video(task_id, duration)
        except Exception as e:
            logger.error(f"创建黑屏视频失败 (其他): {e}")
            # 返回备用方案
            return self._create_placeholder_video(task_id, duration)
    
    def _calculate_actual_audio_duration(self, audio_segments: List[Dict]) -> float:
        """
        计算音频段的实际总时长
        """
        if not audio_segments:
            return 10.0  # 最小默认时长
        
        # 找到最后一个音频段的结束时间
        max_end_time = 0
        total_actual_duration = 0
        
        for segment in audio_segments:
            try:
                # 获取音频文件的实际时长
                audio_segment = AudioSegment.from_wav(segment['file_path'])
                actual_duration = len(audio_segment) / 1000.0  # 转换为秒
                
                # 累加实际音频时长
                total_actual_duration += actual_duration
                
                # 更新最大结束时间
                max_end_time = max(max_end_time, segment['end_time'])
                
            except Exception as e:
                logger.warning(f"无法读取音频文件 {segment['file_path']}: {e}")
                # 使用预设时间作为备用
                segment_duration = segment['end_time'] - segment['start_time']
                total_actual_duration += segment_duration
        
        # 返回实际音频总时长，确保视频时长与音频完全匹配
        return total_actual_duration
    
    def _combine_audio_segments(self, task_id: str, audio_segments: List[Dict], total_duration: float) -> str:
        """
        合并音频段为完整音频（按顺序连接）
        """
        logger.info(f"任务 {task_id}: 合并 {len(audio_segments)} 个音频段")
        
        if not audio_segments:
            # 如果没有音频段，创建静音音频
            combined_audio = AudioSegment.silent(duration=int(total_duration * 1000))
        else:
            # 按start_time排序音频段
            sorted_segments = sorted(audio_segments, key=lambda x: x['start_time'])
            
            combined_audio = AudioSegment.empty()
            current_time = 0
            
            for i, segment in enumerate(sorted_segments):
                try:
                    # 加载音频段
                    audio_segment = AudioSegment.from_wav(segment['file_path'])
                    
                    # 如果当前音频段的开始时间晚于当前时间，添加静音间隔
                    if segment['start_time'] > current_time:
                        silence_duration = (segment['start_time'] - current_time) * 1000
                        silence = AudioSegment.silent(duration=int(silence_duration))
                        combined_audio += silence
                        logger.debug(f"任务 {task_id}: 添加静音间隔 {silence_duration}ms")
                    
                    # 添加音频段
                    combined_audio += audio_segment
                    current_time = segment['start_time'] + len(audio_segment) / 1000.0
                    
                    logger.debug(f"任务 {task_id}: 音频段 {i+1} 添加完成，时长 {len(audio_segment)}ms")
                    
                except Exception as e:
                    logger.error(f"任务 {task_id}: 处理音频段失败: {e}")
                    # 添加静音作为备用
                    segment_duration = segment['end_time'] - segment['start_time']
                    silence = AudioSegment.silent(duration=int(segment_duration * 1000))
                    combined_audio += silence
                    current_time = segment['end_time']
        
        # 导出合并后的音频
        combined_audio_file = f"{self.temp_dir}/combined_audio_{task_id}.wav"
        combined_audio.export(combined_audio_file, format="wav")
        
        actual_duration = len(combined_audio) / 1000.0
        logger.info(f"任务 {task_id}: 音频合并完成: {combined_audio_file}，实际时长: {actual_duration:.2f}s")
        return combined_audio_file
    
    def _combine_audio_with_timestamps(self, task_id: str, audio_segments: List[Dict], timestamps: Dict, total_duration: float) -> str:
        """
        根据时间戳精确同步音频段，确保音频片段之间有合适的间隔
        timestamps: {'MARKER_1': 2.5, 'MARKER_2': 5.0, ...}
        """
        logger.info(f"任务 {task_id}: 使用时间戳同步合并 {len(audio_segments)} 个音频段")
        
        # 预处理音频段，计算实际需要的时长和位置
        processed_segments = []
        max_end_time = total_duration
        
        for i, segment in enumerate(audio_segments):
            marker = segment.get('marker', '')
            try:
                # 预加载音频段以获取其长度
                temp_audio = AudioSegment.from_file(segment['file_path'])
                audio_duration = len(temp_audio) / 1000.0
                
                if marker in timestamps:
                    # 使用时间戳位置，但确保不会与前一个音频段重叠
                    base_start_time = timestamps[marker]
                    
                    # 检查是否与前一个音频段重叠
                    if processed_segments:
                        prev_segment = processed_segments[-1]
                        min_start_time = prev_segment['end_time'] + 0.3  # 至少0.3秒间隔
                        if base_start_time < min_start_time:
                            logger.warning(f"任务 {task_id}: 音频段 {marker} 时间戳 {base_start_time:.2f}s 与前一段重叠，调整为 {min_start_time:.2f}s")
                            start_time = min_start_time
                        else:
                            start_time = base_start_time
                    else:
                        start_time = base_start_time
                else:
                    # 如果没有时间戳，按顺序排列，保持0.5秒间隔
                    if processed_segments:
                        start_time = processed_segments[-1]['end_time'] + 0.5
                    else:
                        start_time = 0
                    logger.warning(f"任务 {task_id}: 音频段 {marker} 未找到时间戳，使用顺序时间 {start_time:.2f}s")
                
                end_time = start_time + audio_duration
                max_end_time = max(max_end_time, end_time)
                
                processed_segments.append({
                    'file_path': segment['file_path'],
                    'marker': marker,
                    'start_time': start_time,
                    'end_time': end_time,
                    'audio_duration': audio_duration
                })
                
                logger.info(f"任务 {task_id}: 音频段 {marker} 计划在 {start_time:.2f}s-{end_time:.2f}s 播放")
                
            except Exception as e:
                logger.warning(f"任务 {task_id}: 无法预加载音频段 {marker}: {e}")
        
        # 使用计算出的最大时长创建音频轨道
        actual_duration = max_end_time + 1.0  # 额外添加1秒缓冲
        logger.info(f"任务 {task_id}: 调整音频轨道时长为 {actual_duration:.2f}s")
        combined_audio = AudioSegment.silent(duration=int(actual_duration * 1000))
        
        # 按处理后的时间顺序插入音频段
        for segment in processed_segments:
            try:
                # 加载音频段
                audio_segment = AudioSegment.from_file(segment['file_path'])
                
                # 计算插入位置（毫秒）
                insert_position = int(segment['start_time'] * 1000)
                
                # 确保音频轨道足够长
                required_length = insert_position + len(audio_segment)
                if required_length > len(combined_audio):
                    extension_duration = required_length - len(combined_audio)
                    extension = AudioSegment.silent(duration=extension_duration)
                    combined_audio = combined_audio + extension
                    logger.info(f"任务 {task_id}: 扩展音频轨道以容纳音频段 {segment['marker']}")
                
                # 使用叠加模式而不是覆盖模式，避免截断音频
                # 先提取前后部分
                before_audio = combined_audio[:insert_position]
                after_position = insert_position + len(audio_segment)
                after_audio = combined_audio[after_position:]
                
                # 重新组合音频
                combined_audio = before_audio + audio_segment + after_audio
                
                logger.info(f"任务 {task_id}: 音频段 {segment['marker']} 已插入到 {segment['start_time']:.2f}s 位置")
                    
            except Exception as e:
                logger.error(f"任务 {task_id}: 处理音频段 {segment.get('marker', 'unknown')} 失败: {e}")
                continue
        
        # 导出合并后的音频
        combined_audio_file = f"{self.temp_dir}/combined_audio_synced_{task_id}.wav"
        combined_audio.export(combined_audio_file, format="wav")
        
        final_duration = len(combined_audio) / 1000.0
        logger.info(f"任务 {task_id}: 时间戳同步音频合并完成: {combined_audio_file}，实际时长: {final_duration:.2f}s")
        
        # 将实际时长存储在实例变量中，供后续使用
        self._last_audio_duration = final_duration
        
        return combined_audio_file
    
    def _adjust_video_duration(self, task_id: str, video_file: str, target_duration: float) -> str:
        """
        调整视频时长以匹配音频
        """
        logger.info(f"任务 {task_id}: 调整视频时长至 {target_duration:.2f}s")
        
        # 检查是否为占位文件
        try:
            with open(video_file, 'r') as f:
                content = f.read()
                if "# 占位视频文件" in content:
                    logger.info(f"任务 {task_id}: 检测到占位文件，生成与音频时长匹配的视频")
                    return self._create_placeholder_video(task_id, target_duration)
        except:
            # 如果不是文本文件，继续正常处理
            pass
        
        try:
            # 获取视频信息
            probe = ffmpeg.probe(video_file)
            video_duration = float(probe['streams'][0]['duration'])
            
            logger.info(f"任务 {task_id}: 原视频时长 {video_duration:.2f}s")
            
            adjusted_video_file = f"{self.temp_dir}/adjusted_video_{task_id}.mp4"
            
            # 计算时长差异百分比
            duration_diff_percent = abs(video_duration - target_duration) / target_duration * 100
            
            if abs(video_duration - target_duration) > 0.5:  # 如果差异超过0.5秒
                if video_duration > target_duration:
                    # 截断视频
                    (
                        ffmpeg
                        .input(video_file)
                        .output(adjusted_video_file, t=target_duration, vcodec='libx264', acodec='copy')
                        .overwrite_output()
                        .run(quiet=True)
                    )
                else:
                    # 🎯 新的处理策略：不延长视频，保持原始时长
                    # 让动画自然结束，音频继续播放
                    logger.info(f"任务 {task_id}: 视频时长({video_duration:.2f}s)短于音频时长({target_duration:.2f}s)")
                    logger.info(f"任务 {task_id}: 采用新策略：保持视频原始时长，让动画自然结束")
                    logger.info(f"任务 {task_id}: 视频结束后将停留在最后一帧，音频继续播放")
                    
                    # 不进行任何时长调整，直接使用原视频
                    # 在后续的音视频合并阶段，视频会自动停留在最后一帧
                    return video_file
                
                logger.info(f"任务 {task_id}: 视频时长调整完成")
                return adjusted_video_file
            else:
                # 时长差异不大，直接使用原视频
                return video_file
                
        except Exception as e:
            logger.error(f"任务 {task_id}: 调整视频时长失败: {e}")
            # 如果调整失败，尝试创建占位视频
            logger.info(f"任务 {task_id}: 创建占位视频作为备用")
            return self._create_placeholder_video(task_id, target_duration)
    
    def _create_placeholder_video(self, task_id: str, target_duration: float) -> str:
        """
        创建与音频时长匹配的占位视频
        """
        placeholder_video_file = f"{self.temp_dir}/placeholder_video_{task_id}.mp4"
        
        try:
            import subprocess
            
            # 创建一个与目标时长匹配的渐变视频
            cmd = [
                'ffmpeg', '-y',  # -y 覆盖输出文件
                '-f', 'lavfi',
                '-i', f'color=blue:size=1280x720:duration={target_duration}:rate=30',
                '-vf', 'fade=in:0:30,fade=out:{}:30'.format(max(0, int((target_duration - 1) * 30))),  # 添加淡入淡出效果
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                placeholder_video_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"任务 {task_id}: 占位视频生成成功，时长: {target_duration:.2f}s")
                return placeholder_video_file
            else:
                logger.error(f"任务 {task_id}: 占位视频生成失败: {result.stderr}")
                raise Exception(f"占位视频生成失败: {result.stderr}")
                
        except Exception as e:
            logger.error(f"任务 {task_id}: 创建占位视频失败: {e}")
            # 如果无法创建视频，返回原文件路径（让后续处理决定如何处理）
            raise Exception(f"无法创建占位视频: {e}")
    
    def _merge_audio_video(self, task_id: str, video_file: str, audio_file: str) -> str:
        """
        合并音频和视频
        支持视频短于音频的情况，视频结束后停留在最后一帧
        """
        logger.info(f"任务 {task_id}: 合并音频和视频")
        
        final_video_file = f"{self.temp_dir}/final_video_{task_id}.mp4"
        
        try:
            # 获取音频和视频的时长信息
            video_probe = ffmpeg.probe(video_file)
            audio_probe = ffmpeg.probe(audio_file)
            
            video_duration = float(video_probe['streams'][0]['duration'])
            audio_duration = float(audio_probe['streams'][0]['duration'])
            
            logger.info(f"任务 {task_id}: 视频时长 {video_duration:.2f}s, 音频时长 {audio_duration:.2f}s")
            
            video_input = ffmpeg.input(video_file)
            audio_input = ffmpeg.input(audio_file)
            
            if video_duration < audio_duration:
                # 视频短于音频：循环最后一帧直到音频结束
                logger.info(f"任务 {task_id}: 视频短于音频，将停留在最后一帧")
                try:
                    (
                        ffmpeg
                        .output(
                            video_input['v'],
                            audio_input['a'],
                            final_video_file,
                            vcodec='libx264',
                            acodec='aac',
                            video_bitrate='2M',
                            audio_bitrate='128k',
                            preset='medium',
                            crf=23,
                            shortest=None,  # 不使用shortest，让音频完整播放
                            **{'filter:v': f'tpad=stop_mode=clone:stop_duration={audio_duration-video_duration}'}
                        )
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as e:
                    logger.error(f"任务 {task_id}: ffmpeg视频扩展合并失败: {e.stderr.decode() if e.stderr else str(e)}")
                    raise
            else:
                # 正常合并
                try:
                    (
                        ffmpeg
                        .output(
                            video_input['v'],
                            audio_input['a'],
                            final_video_file,
                            vcodec='libx264',
                            acodec='aac',
                            video_bitrate='2M',
                            audio_bitrate='128k',
                            preset='medium',
                            crf=23
                        )
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as e:
                    logger.error(f"任务 {task_id}: ffmpeg正常合并失败: {e.stderr.decode() if e.stderr else str(e)}")
                    raise
            
            logger.info(f"任务 {task_id}: 音视频合并完成: {final_video_file}")
            return final_video_file
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 音视频合并失败: {e}")
            raise
    
    def get_video_info(self, video_file: str) -> Dict:
        """
        获取视频信息
        """
        try:
            probe = ffmpeg.probe(video_file)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            info = {
                'duration': float(probe['format']['duration']),
                'size': int(probe['format']['size']),
                'format': probe['format']['format_name']
            }
            
            if video_stream:
                info.update({
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height']),
                    'fps': eval(video_stream['r_frame_rate'])
                })
            
            if audio_stream:
                info.update({
                    'audio_codec': audio_stream['codec_name'],
                    'sample_rate': int(audio_stream['sample_rate'])
                })
            
            return info
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return {}
    
    def cleanup_temp_files(self, task_id: str):
        """
        清理临时文件
        """
        logger.info(f"任务 {task_id}: 开始清理VideoComposer临时文件")
        
        temp_patterns = [
            f"audio_{task_id}_*.wav",
            f"combined_audio_{task_id}.wav",
            f"adjusted_video_{task_id}.mp4",
            f"final_video_{task_id}.mp4",
            f"manim_video_{task_id}.mp4"
        ]
        
        import glob
        files_deleted = 0
        
        # 清理temp_dir中的文件
        for pattern in temp_patterns:
            files = glob.glob(os.path.join(self.temp_dir, pattern))
            for file in files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                        files_deleted += 1
                        logger.debug(f"删除临时文件: {file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败 {file}: {e}")
        
        # 清理/tmp目录中的相关文件
        tmp_patterns = [
            f"/tmp/audio_{task_id}_*.wav",
            f"/tmp/combined_audio_{task_id}.wav",
            f"/tmp/adjusted_video_{task_id}.mp4",
            f"/tmp/final_video_{task_id}.mp4",
            f"/tmp/manim_video_{task_id}.mp4",
            f"/tmp/sentence_{task_id}_*.wav",
            f"/tmp/merged_audio_{task_id}.wav",
            f"/tmp/black_{task_id}_*.mp4",
            f"/tmp/merged_animation_{task_id}.mp4"
        ]
        
        for pattern in tmp_patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                        files_deleted += 1
                        logger.debug(f"删除临时文件: {file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败 {file}: {e}")
        
        logger.info(f"任务 {task_id}: VideoComposer临时文件清理完成，共删除 {files_deleted} 个文件")