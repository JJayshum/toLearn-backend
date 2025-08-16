from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tts.v20190823 import tts_client, models
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from pydub import AudioSegment
import os
import logging
import time
import base64
import json
from typing import Dict, List

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.region = os.getenv("TENCENT_REGION", "ap-beijing")
        
        if not self.secret_id or not self.secret_key:
            logger.warning("腾讯云密钥未配置，将使用模拟TTS服务")
            self.use_mock = True
        else:
            self.use_mock = False
            # 实例化一个认证对象
            cred = credential.Credential(self.secret_id, self.secret_key)
            # 实例化一个http选项
            httpProfile = HttpProfile()
            httpProfile.endpoint = "tts.tencentcloudapi.com"
            # 实例化一个client选项
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            # 实例化要请求产品的client对象
            self.client = tts_client.TtsClient(cred, self.region, clientProfile)


    
    def generate_sentence_audio_with_timing(self, task_id: str, sentence_narration: list) -> Dict:
        """
        为分句旁白生成音频并记录精确时间点
        
        Args:
            task_id: 任务ID
            sentence_narration: 分句旁白列表，格式：[{"sentence": "句子内容"}, ...]
            
        Returns:
            {
                "audio_segments": [...],  # 音频文件信息
                "sentence_timings": [...],  # 句子时间信息
                "merged_audio_file": "...",  # 合并后的完整音频文件
                "total_duration": 总时长
            }
        """
        audio_files = []
        sentence_timings = []
        current_time = 0.0
        
        logger.info(f"任务 {task_id}: 开始生成 {len(sentence_narration)} 个句子的音频")
        
        for i, sentence_data in enumerate(sentence_narration):
            try:
                sentence_text = sentence_data['sentence']
                
                # 生成单句音频
                audio_file = f"/tmp/sentence_{task_id}_{i}.wav"
                
                if self.use_mock:
                    self._create_improved_mock_audio(sentence_text, audio_file)
                else:
                    self._synthesize_text(sentence_text, audio_file)
                
                # 获取音频实际时长
                actual_duration = self._get_audio_duration(audio_file)
                
                # 记录句子时间信息
                sentence_timing = {
                    'sentence_index': i,
                    'sentence': sentence_text,
                    'start_time': current_time,
                    'duration': actual_duration,
                    'end_time': current_time + actual_duration
                }
                
                sentence_timings.append(sentence_timing)
                
                # 记录音频文件信息
                audio_files.append({
                    'file_path': audio_file,
                    'start_time': current_time,
                    'end_time': current_time + actual_duration,
                    'duration': actual_duration,
                    'sentence_index': i
                })
                
                # 更新当前时间（添加小间隔）
                current_time += actual_duration + 0.2  # 句子间0.2秒间隔
                
                logger.info(f"任务 {task_id}: 句子 {i+1}/{len(sentence_narration)} 音频生成完成，时长: {actual_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"任务 {task_id}: 生成句子 {i} 音频失败: {e}")
                # 使用模拟音频作为备用
                fallback_audio = f"/tmp/sentence_{task_id}_{i}_fallback.wav"
                try:
                    self._create_improved_mock_audio(sentence_data.get('sentence', '备用音频'), fallback_audio)
                    fallback_duration = self._get_audio_duration(fallback_audio)
                    
                    sentence_timing = {
                        'sentence_index': i,
                        'sentence': sentence_data.get('sentence', '备用音频'),
                        'start_time': current_time,
                        'duration': fallback_duration,
                        'end_time': current_time + fallback_duration
                    }
                    
                    sentence_timings.append(sentence_timing)
                    
                    audio_files.append({
                        'file_path': fallback_audio,
                        'start_time': current_time,
                        'end_time': current_time + fallback_duration,
                        'duration': fallback_duration,
                        'sentence_index': i
                    })
                    
                    current_time += fallback_duration + 0.2
                    
                except Exception as fallback_e:
                    logger.error(f"任务 {task_id}: 生成备用音频也失败: {fallback_e}")
                    continue
        
        # 合并所有音频文件
        merged_audio_file = f"/tmp/merged_audio_{task_id}.wav"
        total_duration = self._merge_audio_files(audio_files, merged_audio_file)
        
        logger.info(f"任务 {task_id}: 所有句子音频生成完成，总时长: {total_duration:.2f}s")
        
        return {
            'audio_segments': audio_files,
            'sentence_timings': sentence_timings,
            'merged_audio_file': merged_audio_file,
            'total_duration': total_duration
        }
    
    def _merge_audio_files(self, audio_files: list, output_file: str) -> float:
        """
        合并音频文件，在句子间添加间隔
        """
        try:
            if not audio_files:
                # 创建空音频文件
                silence = AudioSegment.silent(duration=1000)  # 1秒静音
                silence.export(output_file, format="wav")
                return 1.0
            
            # 加载第一个音频文件
            merged_audio = AudioSegment.from_wav(audio_files[0]['file_path'])
            
            # 逐个添加后续音频文件，在每个文件间添加间隔
            for i in range(1, len(audio_files)):
                # 添加0.2秒间隔
                silence = AudioSegment.silent(duration=200)  # 200ms
                merged_audio += silence
                
                # 添加下一个音频文件
                next_audio = AudioSegment.from_wav(audio_files[i]['file_path'])
                merged_audio += next_audio
            
            # 导出合并后的音频
            merged_audio.export(output_file, format="wav")
            
            # 返回总时长（秒）
            return len(merged_audio) / 1000.0
            
        except Exception as e:
            logger.error(f"合并音频文件失败: {e}")
            # 创建备用静音文件
            silence = AudioSegment.silent(duration=5000)  # 5秒静音
            silence.export(output_file, format="wav")
            return 5.0
    
    def _synthesize_text(self, text: str, output_file: str):
        """
        使用腾讯云TTS合成单段文本为音频
        """
        try:
            # 清理文本，移除或替换可能导致TTS失败的特殊字符
            cleaned_text = self._clean_text_for_tts(text)
            logger.info(f"原始文本: {text}")
            logger.info(f"清理后文本: {cleaned_text}")
            
            # 检查文本长度，如果过长则分割处理
            if len(cleaned_text) > 200:  # 腾讯云TTS单次请求建议不超过200字符
                logger.info(f"文本长度 {len(cleaned_text)} 超过限制，进行分割处理")
                self._synthesize_long_text(cleaned_text, output_file)
                return
            
            # 实例化一个请求对象
            req = models.TextToVoiceRequest()
            
            # 设置请求参数
            req.Text = cleaned_text
            req.SessionId = f"session_{int(time.time())}"
            req.Volume = 1  # 音量 0-10
            req.Speed = 1   # 语速 0.6-1.5
            req.ProjectId = 0
            req.ModelType = 1  # 模型类型：1-默认模型
            req.VoiceType = 101002  # 音色：智聆通用女声（精品音色）
            req.PrimaryLanguage = 1  # 主语言类型：1-中文（默认）
            req.SampleRate = 16000  # 采样率：16000（腾讯云TTS支持的标准采样率）
            req.Codec = "wav"  # 返回音频格式
            req.EnableSubtitle = False  # 是否开启时间戳功能
            
            # 发起请求
            resp = self.client.TextToVoice(req)
            
            # 检查响应
            if not resp.Audio:
                raise Exception("TTS合成失败：返回音频数据为空")
            
            # 解码base64音频数据并保存
            audio_data = base64.b64decode(resp.Audio)
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"TTS合成成功: {output_file}")
            
        except TencentCloudSDKException as e:
            error_code = getattr(e, 'code', 'Unknown')
            if error_code == 'UnsupportedOperation.PkgExhausted':
                logger.warning(f"腾讯云TTS资源包配额已用尽，请检查账户配额或开通后付费模式: {e}")
                logger.warning("解决方案：")
                logger.warning("1. 登录腾讯云控制台 -> 语音合成 -> 资源包管理，领取免费资源包")
                logger.warning("2. 或购买预付费资源包")
                logger.warning("3. 或开通后付费模式")
                logger.warning("详情请参考：https://cloud.tencent.com/document/product/1073/34112")
            else:
                logger.error(f"腾讯云TTS API调用失败: {e}")
            raise Exception(f"TTS合成失败: {e}")
        except Exception as e:
            logger.error(f"TTS合成过程中发生错误: {e}")
            raise Exception(f"TTS合成失败: {e}")
    
    def _synthesize_long_text(self, text: str, output_file: str):
        """
        处理长文本，分割后分别合成再合并
        """
        try:
            # 智能分割文本
            segments = self._split_text_intelligently(text, max_length=200)
            logger.info(f"文本分割为 {len(segments)} 个片段")
            
            # 为每个片段生成音频
            segment_files = []
            for i, segment in enumerate(segments):
                segment_file = f"{output_file}_segment_{i}.wav"
                logger.info(f"合成片段 {i+1}/{len(segments)}: {segment[:50]}...")
                
                # 递归调用，但确保片段长度不会再次超限
                if len(segment) <= 200:
                    self._synthesize_single_segment(segment, segment_file)
                else:
                    # 如果片段仍然过长，进行强制分割
                    logger.warning(f"片段 {i} 仍然过长，进行强制分割")
                    sub_segments = [segment[j:j+180] for j in range(0, len(segment), 180)]
                    sub_files = []
                    for j, sub_segment in enumerate(sub_segments):
                        sub_file = f"{output_file}_segment_{i}_{j}.wav"
                        self._synthesize_single_segment(sub_segment, sub_file)
                        sub_files.append(sub_file)
                    # 合并子片段
                    self._merge_audio_segments(sub_files, segment_file)
                    # 清理子片段文件
                    for sub_file in sub_files:
                        if os.path.exists(sub_file):
                            os.remove(sub_file)
                
                segment_files.append(segment_file)
            
            # 合并所有片段
            self._merge_audio_segments(segment_files, output_file)
            
            # 清理片段文件
            for segment_file in segment_files:
                if os.path.exists(segment_file):
                    os.remove(segment_file)
            
            logger.info(f"长文本TTS合成完成: {output_file}")
            
        except Exception as e:
            logger.error(f"长文本TTS合成失败: {e}")
            raise Exception(f"长文本TTS合成失败: {e}")
    
    def _split_text_intelligently(self, text: str, max_length: int = 200) -> List[str]:
        """
        智能分割文本，优先在句号、逗号等标点符号处分割
        """
        if len(text) <= max_length:
            return [text]
        
        segments = []
        current_segment = ""
        
        # 按句号分割
        sentences = text.split('。')
        
        for i, sentence in enumerate(sentences):
            # 重新添加句号（除了最后一个）
            if i < len(sentences) - 1:
                sentence += '。'
            
            # 如果当前片段加上这个句子不会超长
            if len(current_segment + sentence) <= max_length:
                current_segment += sentence
            else:
                # 如果当前片段不为空，先保存
                if current_segment:
                    segments.append(current_segment.strip())
                    current_segment = ""
                
                # 如果单个句子就超长，需要进一步分割
                if len(sentence) > max_length:
                    # 按逗号分割
                    sub_parts = sentence.split('，')
                    for j, part in enumerate(sub_parts):
                        if j < len(sub_parts) - 1:
                            part += '，'
                        
                        if len(current_segment + part) <= max_length:
                            current_segment += part
                        else:
                            if current_segment:
                                segments.append(current_segment.strip())
                            current_segment = part
                else:
                    current_segment = sentence
        
        # 添加最后一个片段
        if current_segment:
            segments.append(current_segment.strip())
        
        return [seg for seg in segments if seg.strip()]
    
    def _synthesize_single_segment(self, text: str, output_file: str):
        """
        合成单个文本片段（确保不超过长度限制）
        """
        # 实例化一个请求对象
        req = models.TextToVoiceRequest()
        
        # 设置请求参数
        req.Text = text
        req.SessionId = f"session_{int(time.time())}"
        req.Volume = 1  # 音量 0-10
        req.Speed = 1   # 语速 0.6-1.5
        req.ProjectId = 0
        req.ModelType = 1  # 模型类型：1-默认模型
        req.VoiceType = 101002  # 音色：智聆通用女声（精品音色）
        req.PrimaryLanguage = 1  # 主语言类型：1-中文（默认）
        req.SampleRate = 16000  # 采样率：16000（腾讯云TTS支持的标准采样率）
        req.Codec = "wav"  # 返回音频格式
        req.EnableSubtitle = False  # 是否开启时间戳功能
        
        # 发起请求
        resp = self.client.TextToVoice(req)
        
        # 检查响应
        if not resp.Audio:
            raise Exception("TTS合成失败：返回音频数据为空")
        
        # 解码base64音频数据并保存
        audio_data = base64.b64decode(resp.Audio)
        with open(output_file, 'wb') as f:
            f.write(audio_data)
    
    def _merge_audio_segments(self, segment_files: List[str], output_file: str):
        """
        合并多个音频片段
        """
        if not segment_files:
            raise Exception("没有音频片段可合并")
        
        if len(segment_files) == 1:
            # 只有一个片段，直接复制
            import shutil
            shutil.copy2(segment_files[0], output_file)
            return
        
        # 合并多个片段
        combined = AudioSegment.empty()
        
        for segment_file in segment_files:
            if os.path.exists(segment_file):
                segment = AudioSegment.from_wav(segment_file)
                combined += segment
                # 添加短暂间隔
                combined += AudioSegment.silent(duration=100)  # 100ms间隔
        
        # 导出合并后的音频
        combined.export(output_file, format="wav")
    
    def _clean_text_for_tts(self, text: str) -> str:
        """
        清理文本，移除或替换可能导致腾讯云TTS API失败的特殊字符
        """
        if not text or not text.strip():
            return "空白内容"
        
        # 移除或替换数学符号和特殊字符
        cleaned = text
        
        # 替换数学符号
        replacements = {
            '=': '等于',
            '+': '加',
            '-': '减',
            '×': '乘以',
            '÷': '除以',
            '/': '除以',
            '*': '乘以',
            '²': '的平方',
            '³': '的立方',
            '√': '根号',
            '≤': '小于等于',
            '≥': '大于等于',
            '<': '小于',
            '>': '大于',
            '≠': '不等于',
            '∞': '无穷大',
            'π': '派',
            '°': '度',
            '%': '百分之',
        }
        
        for symbol, replacement in replacements.items():
            cleaned = cleaned.replace(symbol, replacement)
        
        # 移除其他可能有问题的字符，保留中文、英文、数字、基本标点符号
        import re
        # 移除不常见的特殊字符，但保留基本的中文标点
        cleaned = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z0-9\s，。！？；：（）【】《》、\.,!?;:"\(\)\[\]<>\-_]', '', cleaned)
        
        # 如果清理后文本为空或只有空白字符，返回默认内容
        if not cleaned or not cleaned.strip():
            return "无法识别的内容"
        
        return cleaned.strip()
        """
        创建模拟音频（用于测试）
        """
        try:
            # 尝试使用pyttsx3进行本地TTS
            import pyttsx3
            engine = pyttsx3.init()
            
            # 设置语音参数
            engine.setProperty('rate', 150)  # 语速
            engine.setProperty('volume', 0.8)  # 音量
            
            # 尝试设置中文语音
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            # 生成音频文件
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            
            logger.info(f"使用pyttsx3生成音频: {output_file}")
            
        except ImportError:
            logger.warning("pyttsx3未安装，使用改进的模拟音频")
            self._create_improved_mock_audio(text, output_file)
        except Exception as e:
            logger.warning(f"pyttsx3生成失败: {e}，使用改进的模拟音频")
            self._create_improved_mock_audio(text, output_file)
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """
        获取音频文件的时长（秒）
        """
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_file)
            return len(audio) / 1000.0  # 转换为秒
        except Exception as e:
            logger.error(f"获取音频时长失败: {e}")
            # 返回默认时长
            return 2.0
    
    def _create_improved_mock_audio(self, text: str, output_file: str):
        """
        创建高质量的模拟音频（使用静音音频避免杂音）
        """
        # 根据文本长度估算音频时长（假设每个字符0.15秒）
        duration_ms = len(text) * 150
        
        try:
            # 直接创建静音音频，避免任何杂音
            audio = AudioSegment.silent(duration=duration_ms)
            
            # 设置正确的音频参数
            audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
            
            # 导出为高质量WAV文件
            audio.export(output_file, format="wav")
            
            logger.info(f"创建静音音频: {output_file}, 时长: {duration_ms}ms")
            
        except Exception as e:
            logger.error(f"创建模拟音频失败: {e}")
            # 如果连静音音频都失败，创建最基本的空文件
            try:
                import wave
                with wave.open(output_file, 'w') as wav_file:
                    wav_file.setnchannels(1)  # 单声道
                    wav_file.setsampwidth(2)  # 16位
                    wav_file.setframerate(44100)  # 44.1kHz
                    # 写入静音数据
                    silence_frames = b'\x00\x00' * int(44100 * duration_ms / 1000)
                    wav_file.writeframes(silence_frames)
                logger.info(f"使用wave库创建静音音频: {output_file}")
            except Exception as wave_e:
                logger.error(f"使用wave库也失败: {wave_e}")
                raise Exception(f"无法创建音频文件: {e}")
    
    def _adjust_audio_duration(self, audio_file: str, target_duration: float) -> str:
        """
        调整音频时长
        """
        try:
            audio = AudioSegment.from_wav(audio_file)
            current_duration = len(audio) / 1000.0  # 转换为秒
            
            logger.info(f"调整音频时长: 当前 {current_duration:.2f}s -> 目标 {target_duration:.2f}s")
            
            if abs(current_duration - target_duration) > 0.5:  # 如果差异超过0.5秒
                # 调整播放速度
                speed_ratio = current_duration / target_duration
                if 0.5 <= speed_ratio <= 2.0:  # 合理的速度范围
                    # 使用speedup方法调整速度
                    if speed_ratio > 1:
                        audio = audio.speedup(playback_speed=speed_ratio)
                    else:
                        # 对于减速，我们需要用不同的方法
                        new_frame_rate = int(audio.frame_rate * speed_ratio)
                        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
                        audio = audio.set_frame_rate(audio.frame_rate)
                else:
                    # 如果速度调整超出合理范围，则截断或填充
                    target_length = int(target_duration * 1000)
                    if len(audio) > target_length:
                        audio = audio[:target_length]
                    else:
                        silence = AudioSegment.silent(duration=target_length - len(audio))
                        audio = audio + silence
            
            adjusted_file = audio_file.replace('.wav', '_adjusted.wav')
            audio.export(adjusted_file, format="wav")
            
            # 删除原文件
            os.remove(audio_file)
            
            return adjusted_file
            
        except Exception as e:
            logger.error(f"调整音频时长失败: {e}")
            # 如果调整失败，返回原文件
            return audio_file
    
    def _create_silence_audio(self, task_id: str, segment_index: int, duration: float) -> str:
        """
        创建静音音频作为备用
        """
        silence_file = f"/tmp/audio_{task_id}_{segment_index}_silence.wav"
        silence = AudioSegment.silent(duration=int(duration * 1000))
        silence.export(silence_file, format="wav")
        
        logger.info(f"创建静音音频: {silence_file}, 时长: {duration}s")
        return silence_file