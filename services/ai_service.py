import os
import re
import ast
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from .tts_service import TTSService

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        load_dotenv()
        # 优先使用 DeepSeek API
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.deepseek_api_key:
            self.use_deepseek = True
            self.use_mock = False
            logger.info("使用 DeepSeek API 进行动画生成")
        elif self.openai_api_key:
            self.use_deepseek = False
            self.use_mock = False
            logger.info("使用 OpenAI API 进行动画生成")
        else:
            logger.warning("未找到 DeepSeek 或 OpenAI API 密钥，将使用模拟数据")
            self.use_deepseek = False
            self.use_mock = True
        
        # 初始化TTS服务
        self.tts_service = TTSService()
    
    def generate_sentence_narration(self, request_data: Dict) -> List[Dict]:
        """
        步骤一：生成分段旁白脚本
        
        通过AI提示词生成旁白，每个完整的意群作为一个分段
        
        Args:
            request_data: 包含问题和解答数据的字典
            
        Returns:
            List[Dict]: 分句旁白列表，每个元素包含sentence字段
        """
        try:
            question = request_data.get('question', '')
            solution = request_data.get('solution_data', {}).get('detailed_solution', '')
            
            # 构建AI提示词
            prompt = f"""
请为以下数学问题生成详细的解题旁白，要求：
1. 将解题过程分解为多个完整的意群段落
2. 每个段落应该是一个意义：例如一个公式的完整变形，一个图像的生成和变化等。
3. 语言要清晰易懂，适合语音播报
4. 每个段落长度适中（30-80字），便于TTS合成
5. 可以多用图像表述，增强直观理解

问题：{question}
解答：{solution}

请按照以下格式返回JSON数组：
[
  {{"sentence": "第一段旁白内容"}},
  {{"sentence": "第二段旁白内容"}},
  ...
]
"""
            
            # 强制使用AI API生成分句旁白
            sentences = self._call_ai_for_narration_generation(prompt, question, solution)
            
            logger.info(f"生成了 {len(sentences)} 句旁白")
            return sentences
            
        except Exception as e:
            logger.error(f"生成分句旁白时出错: {e}")
            raise Exception(f"AI API调用失败: {e}")
    

    
    def generate_narration_with_audio_track(self, task_id: str, request_data: Dict) -> Dict:
        """
        步骤一完整流程：生成分段旁白并合成音轨，记录每句话的时间点
        
        Args:
            task_id: 任务ID
            request_data: 包含问题和解答数据的字典
            
        Returns:
            Dict: 包含旁白、音轨文件路径和时间信息的字典
            {
                "sentence_narration": [...],  # 分句旁白列表
                "audio_track_file": "path/to/audio.wav",  # 完整音轨文件路径
                "sentence_timings": [...],  # 每句话的时间信息
                "total_duration": 10.5  # 总时长（秒）
            }
        """
        try:
            # 1. 生成分段旁白
            sentence_narration = self.generate_sentence_narration(request_data)
            logger.info(f"生成了 {len(sentence_narration)} 句旁白")
            
            # 2. 通过TTS合成音轨并记录时间点
            audio_result = self.tts_service.generate_sentence_audio_with_timing(
                task_id, sentence_narration
            )
            
            # 3. 整理返回结果
            result = {
                "sentence_narration": sentence_narration,
                "audio_track_file": audio_result.get("merged_audio_file"),
                "sentence_timings": audio_result.get("sentence_timings", []),
                "total_duration": audio_result.get("total_duration", 0)
            }
            
            logger.info(f"步骤一完成：生成音轨文件 {result['audio_track_file']}，总时长 {result['total_duration']}秒")
            return result
            
        except Exception as e:
            logger.error(f"生成旁白音轨时出错: {e}")
            return {
                "sentence_narration": [{'sentence': '生成旁白时出现错误'}],
                "audio_track_file": None,
                "sentence_timings": [],
                "total_duration": 0
            }
    
    def generate_continuous_animation(self, request_data: Dict, sentence_narration: List[Dict], sentence_timings: List[Dict]) -> Dict:
        """
        步骤二：基于旁白内容生成对应的manim动画
        
        将旁白的每一段话组合而成的内容传入AI提示词，生成对应的manim动画
        
        Args:
            request_data: 包含问题和解答数据的字典
            sentence_narration: 分句旁白列表
            sentence_timings: 每句话的时间信息
            
        Returns:
            Dict: 包含动画代码和相关信息的字典
            {
                'manim_code': str,  # 完整的Manim动画代码
                'animation_methods': List[Dict],  # 每句话对应的动画方法信息
                'method_count': int  # 动画方法数量
            }
        """
        try:
            task_id = request_data.get('task_id', 'unknown')
            logger.info(f"任务 {task_id}: 开始执行步骤二 - 生成连续动画")
            
            question = request_data.get('question', '')
            solution = request_data.get('solution_data', {}).get('detailed_solution', '')
            
            # 构建AI提示词，包含所有旁白内容
            narration_content = '\n'.join([item['sentence'] for item in sentence_narration])
            
            # 计算旁白句子数量
            sentence_count = len(sentence_narration)
            
            prompt = f"""
请根据以下旁白内容生成对应的Manim动画代码，要求：

**核心要求：严格一对一映射**
1. **必须且只能**为每句旁白生成一个独立的动画方法，总共生成 {sentence_count} 个方法
2. 方法名必须严格按照格式：animate_sentence_1(self), animate_sentence_2(self), ..., animate_sentence_{sentence_count}(self)
3. **禁止生成超过 {sentence_count} 个动画方法！**
4. **禁止生成少于 {sentence_count} 个动画方法！**

**动画内容要求：**
5. 动画要与旁白内容紧密配合，形象生动
6. 使用适当的数学可视化元素（如图形、公式、文本等），当同一行内容过多时及时换行，避免显示不完全
7. 确保动画流畅自然，符合教学逻辑
8. 每个动画方法要生动形象且**完全独立**，不依赖其他方法或实例变量
9. 在construct方法中按顺序调用这些animate_sentence方法
10. **每个animate_sentence_X方法必须完全独立，不能引用self的任何属性或其他方法的变量**
11. **重要：每个animate_sentence_X方法内部必须创建自己的所有对象，不能依赖类的其他属性**
12. **动画时长要求：每个动画方法中的文本、公式等元素要在画面中停留足够长的时间，不要一闪而过。建议每个元素至少停留2-3秒，使用self.wait(2)或self.wait(3)来控制停留时间**

**技术要求：**
13. **必须在文件开头定义ctex_template，并严格按照以下规则使用：**
    - **Tex对象：用于显示包含中文的文本，必须使用tex_template=ctex_template参数**
    - **MathTex对象：仅用于纯数学公式（不含中文），必须使用tex_template=ctex_template参数**
14. **文本类型严格区分：**
    - 包含中文的文本（如"求解方程"、"这是一个二次函数"）→ 使用Tex
    - 纯数学公式（如"x^2 + 5x - 6 = 0"、"\\frac{{a}}{{b}}"）→ 使用MathTex
    - 混合内容（如"方程 x^2 + 5x - 6 = 0 的解"）→ 使用Tex
15. **重要：对于中文文本显示，禁止使用Write、AddTextLetterByLetter等逐字显示动画，必须使用FadeIn、DrawBorderThenFill或直接显示，以确保完整文本同时出现，避免波浪效果**
16. **颜色设置要求：中文文本必须使用WHITE颜色，数学公式和MathTex对象必须使用YELLOW颜色，确保在默认背景下可见。禁止使用BLACK颜色！**
17. **背景设置：在construct方法开始时设置背景颜色为深色，使用 self.camera.background_color = "#1e1e1e" 或类似深色**
18. **动画布局要求：所有文本、公式和图形元素都应该居中显示，使用适当的位置参数如UP、DOWN、LEFT、RIGHT来调整布局，确保整体视觉效果居中对称**

原始问题：{question}
解答过程：{solution}

旁白内容（共 {sentence_count} 句，请为每句生成对应的animate_sentence方法）：
{narration_content}

请生成完整的Manim类代码，包含所有必要的导入和方法定义。
**必须生成的方法列表：**
{chr(10).join([f'- def animate_sentence_{i+1}(self): # 对应第{i+1}句旁白: {sentence_narration[i]["sentence"][:50]}...' for i in range(sentence_count)])}

**正确的文本和公式显示方法示例：**
```python
def animate_sentence_1(self):
    # 在方法内部创建所有需要的对象
    
    # 示例1：包含中文的文本 → 使用Tex
    chinese_text = Tex(r"我们来求解这个二次方程", tex_template=ctex_template, color=WHITE)
    
    # 示例2：纯数学公式 → 使用MathTex
    pure_formula = MathTex(r"x^2 + 5x - 6 = 0", tex_template=ctex_template, color=YELLOW)
    
    # 示例3：混合内容（中文+公式）→ 使用Tex
    mixed_content = Tex(r"方程 $x^2 + 5x - 6 = 0$ 的解为", tex_template=ctex_template, color=WHITE)
    
    # 示例4：纯数学表达式 → 使用MathTex
    solution_formula = MathTex(r"x^2 + y^2 = z^2", tex_template=ctex_template, color=YELLOW)
    
    # 确保元素居中显示
    chinese_text.move_to(ORIGIN + 2*UP)
    pure_formula.move_to(ORIGIN + UP)
    mixed_content.move_to(ORIGIN)
    solution_formula.move_to(ORIGIN + DOWN)
    
    self.play(FadeIn(chinese_text))  # 使用FadeIn而不是Write，确保文本完整显示
    self.wait(3)  # 让文本在画面中停留3秒
    self.play(FadeIn(pure_formula))
    self.wait(2)
    self.play(FadeIn(mixed_content))
    self.wait(2)
    self.play(FadeIn(solution_formula))
    self.wait(3)
    self.play(FadeOut(chinese_text), FadeOut(pure_formula), FadeOut(mixed_content), FadeOut(solution_formula))
```

**错误示例（禁止使用）：**
```python
def animate_sentence_2(self):
    # 错误1：引用了不存在的属性
    self.play(Transform(self.solution_group, new_text))  # 禁止！
    
    # 错误2：对中文文本使用Write动画（会产生波浪效果）
    text = Tex(r"中文文本", tex_template=ctex_template)
    self.play(Write(text))  # 禁止！应该使用FadeIn
    
    # 错误3：使用MathTex显示包含中文的内容
    wrong_text = MathTex(r"求解方程", tex_template=ctex_template)  # 禁止！中文应该用Tex
    
    # 错误4：使用Tex显示纯数学公式（虽然可以工作，但不规范）
    formula = Tex(r"x^2 + 5x - 6 = 0", tex_template=ctex_template)  # 不推荐！纯公式应该用MathTex
    
    # 错误5：混合内容错误地拆分为多个对象
    text_part = MathTex(r"方程", tex_template=ctex_template)  # 错误！
    formula_part = MathTex(r"x^2 + 5x - 6 = 0", tex_template=ctex_template)  # 错误！
    text_part2 = MathTex(r"的解为", tex_template=ctex_template)  # 错误！
    # 正确做法：mixed_content = Tex(r"方程 $x^2 + 5x - 6 = 0$ 的解为", tex_template=ctex_template)
```

请直接返回Python代码，不要包含任何解释文字。
"""
            
            # 调用AI API生成动画代码
            if self.use_mock:
                # 模拟模式：使用基于旁白内容的智能动画结构
                animation_code = self._generate_narration_based_animation(question, solution, sentence_narration, sentence_timings)
            else:
                # 真实模式：调用AI API生成动画代码
                animation_code = self._call_ai_for_animation_generation(prompt)
                # 如果AI生成失败，回退到模板模式
                if not animation_code or len(animation_code) < 100:
                    logger.warning("AI生成的动画代码过短或失败，回退到模板模式")
                    animation_code = self._generate_narration_based_animation(question, solution, sentence_narration, sentence_timings)
            
            # 提取动画方法列表
            animation_methods = self._extract_animation_methods_with_timing(animation_code, sentence_timings)
            
            # 验证动画方法数量与旁白句子数量是否一致
            expected_count = len(sentence_narration)
            actual_count = len(animation_methods)
            
            if actual_count != expected_count:
                logger.warning(f"动画方法数量不匹配：期望 {expected_count} 个，实际生成 {actual_count} 个")
                
                # 如果方法数量不匹配，尝试修复
                if actual_count > expected_count:
                    # 如果生成的方法过多，只保留前N个
                    animation_methods = animation_methods[:expected_count]
                    logger.info(f"截取前 {expected_count} 个动画方法")
                elif actual_count < expected_count:
                    # 如果生成的方法过少，使用模板模式重新生成
                    logger.warning("动画方法数量不足，使用模板模式重新生成")
                    animation_code = self._generate_narration_based_animation(question, solution, sentence_narration, sentence_timings)
                    animation_methods = self._extract_animation_methods_with_timing(animation_code, sentence_timings)
            
            # 调试模式：保存生成的Manim代码到临时文件
            import tempfile
            temp_code_file = f"/tmp/animation_code_{task_id}.py"
            try:
                with open(temp_code_file, 'w', encoding='utf-8') as f:
                    f.write(animation_code)
                logger.info(f"任务 {task_id}: 调试：Manim代码已保存到 {temp_code_file}")
                logger.info(f"任务 {task_id}: 生成的代码长度: {len(animation_code)} 字符")
            except Exception as e:
                logger.warning(f"任务 {task_id}: 保存Manim代码文件失败: {e}")
            
            result = {
                'manim_code': animation_code,
                'animation_methods': animation_methods,
                'method_count': len(animation_methods),
                'sentence_count': expected_count,
                'mapping_correct': len(animation_methods) == expected_count,
                'debug_code_file': temp_code_file
            }
            
            logger.info(f"任务 {task_id}: 步骤二完成：生成了包含 {len(animation_methods)} 个动画方法的Manim代码")
            logger.info(f"任务 {task_id}: 返回结果包含 manim_code: {bool(result.get('manim_code'))}")
            return result
            
        except Exception as e:
            task_id = request_data.get('task_id', 'unknown')
            logger.error(f"任务 {task_id}: 生成连续动画时出错: {e}")
            import traceback
            logger.error(f"任务 {task_id}: 错误详情: {traceback.format_exc()}")
            return {
                'manim_code': '# 生成动画代码时出现错误',
                'animation_methods': [],
                'method_count': 0
            }
    
    def _generate_narration_based_animation(self, question: str, solution: str, sentence_narration: List[Dict], sentence_timings: List[Dict]) -> str:
        """
        基于旁白内容生成对应的动画代码
        """
        methods_code = ""
        animation_methods = []
        
        for i, (narration, timing) in enumerate(zip(sentence_narration, sentence_timings)):
            method_name = f"animate_sentence_{i+1}"
            sentence = narration['sentence']
            duration = timing.get('duration', 3.0)
            
            animation_methods.append({
                'method_name': method_name,
                'sentence': sentence,
                'start_time': timing.get('start_time', 0.0),
                'duration': duration,
                'audio_file': timing.get('audio_file', '')
            })
            
            # 根据旁白内容生成相应的动画
            # 优化动画时长：确保最小0.5秒，最大不超过音频时长的80%
            animation_duration = max(0.5, min(duration * 0.8, duration - 0.2))
            wait_duration = duration - animation_duration
            
            # 为每个句子生成一个占位方法，具体实现由AI生成
            methods_code += f"""
    def {method_name}(self):
        # {sentence}
        # 此方法的具体实现将由AI生成
        pass
"""
        
        # 生成完整的Manim类代码（包含中文模板配置）
        # 添加场景转换效果，提高连贯性
        animation_calls_with_transitions = []
        for i, method in enumerate(animation_methods):
            animation_calls_with_transitions.append(f'        self.{method["method_name"]}()')
            # 在动画之间添加短暂的过渡效果（除了最后一个）
            if i < len(animation_methods) - 1:
                animation_calls_with_transitions.append('        self.wait(0.3)  # 场景过渡')
        
        animation_calls = chr(10).join(animation_calls_with_transitions)
        
        # 让AI完全自主生成Manim代码，不使用任何预定义模板
        manim_code = f"""from manim import *

class MathExplanation(Scene):
    def construct(self):
        # 按顺序播放所有动画
{animation_calls}
{methods_code}
"""
        
        return manim_code
    
    def _call_ai_for_narration_generation(self, prompt: str, question: str = "", solution: str = "") -> List[Dict]:
        """
        调用AI API根据提示词生成分句旁白
        """
        try:
            import openai
            import json
            
            # 初始化AI客户端
            if self.use_deepseek:
                client = openai.OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
            else:
                client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="deepseek-chat" if self.use_deepseek else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的数学教学助手，擅长将复杂的数学解题过程分解为清晰易懂的意群段落。请严格按照要求的JSON格式返回结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # 尝试解析JSON响应
            try:
                # 提取JSON部分（可能包含在代码块中）
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                elif "[" in content and "]" in content:
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    content = content[start:end]
                
                sentences = json.loads(content)
                
                # 验证格式并确保长度控制
                validated_sentences = []
                for item in sentences:
                    if isinstance(item, dict) and "sentence" in item:
                        sentence = item["sentence"].strip()
                        # 确保每个句子长度在合理范围内
                        if 10 <= len(sentence) <= 100:
                            validated_sentences.append({"sentence": sentence})
                        elif len(sentence) > 100:
                            # 如果句子过长，尝试分割
                            parts = sentence.split("。")
                            current_part = ""
                            for part in parts:
                                if part.strip():
                                    test_part = current_part + part.strip() + "。"
                                    if len(test_part) <= 80:
                                        current_part = test_part
                                    else:
                                        if current_part and len(current_part) >= 30:
                                            validated_sentences.append({"sentence": current_part})
                                        current_part = part.strip() + "。"
                            if current_part and len(current_part) >= 30:
                                validated_sentences.append({"sentence": current_part})
                
                if validated_sentences:
                    logger.info(f"AI生成了 {len(validated_sentences)} 句旁白")
                    return validated_sentences
                else:
                    logger.error("AI返回的旁白格式不正确")
                    raise Exception("AI返回的旁白格式不正确，无法解析")
                    
            except json.JSONDecodeError as e:
                logger.error(f"解析AI返回的JSON时出错: {e}, 内容: {content}")
                raise Exception(f"AI返回的JSON格式错误: {e}")
                
        except Exception as e:
            logger.error(f"调用AI API生成旁白时出错: {e}")
            raise Exception(f"AI API调用失败: {e}")
    
    def _call_ai_for_animation_generation(self, prompt: str) -> str:
        """
        调用AI API根据提示词生成Manim动画代码
        """
        return self._call_deepseek_for_animation_generation(prompt)
    
    def _call_deepseek_for_animation_generation(self, prompt: str) -> str:
        """
        调用DeepSeek API根据提示词生成Manim动画代码
        """
        try:
            import openai
            
            # 初始化DeepSeek客户端（使用OpenAI兼容接口）
            client = openai.OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            
            # 中文模板配置（避免f-string转义问题）
            template_config = r'''# 中文模板配置（显式指定字体）
from manim import TexTemplate

ctex_template = TexTemplate(
    tex_compiler="xelatex",
    output_format=".pdf",
    preamble=r"""
    \usepackage{ctex}
    \usepackage{amsmath}
    \setCJKmainfont{Noto Sans CJK SC}
    \setmainfont{DejaVu Sans}
    """
)'''
            
            template_usage = r'''# 正确的使用方式：
# 1. 包含中文的文本 → 使用Tex
chinese_text = Tex(r"这是一个示例文本", tex_template=ctex_template, color=WHITE)
# 2. 纯数学公式 → 使用MathTex
math_formula = MathTex(r"x^2 + y^2 = z^2", tex_template=ctex_template, color=YELLOW)
# 3. 混合内容 → 使用Tex
mixed_text = Tex(r"公式 $x^2 + y^2 = z^2$ 表示", tex_template=ctex_template, color=WHITE)

# 动画显示（中文文本禁用Write）
self.play(FadeIn(chinese_text))  # 正确：使用FadeIn
self.play(FadeIn(math_formula))
self.play(FadeIn(mixed_text))'''
            
            # 构建完整的提示词，要求包含中文模板配置
            full_prompt = f"""{prompt}

补充技术要求：
1. 导入：from manim import *
2. 包含完整的类定义和方法
3. 代码可以直接运行
4. **中文模板配置代码：**

```python
{template_config}
```

5. **模板使用示例：**
```python
{template_usage}
```
"""
            
            # 调用DeepSeek API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的Manim动画代码生成专家，能够根据数学内容生成高质量的动画代码。"},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            generated_code = response.choices[0].message.content.strip()
            
            # 清理代码（移除可能的markdown标记）
            if generated_code.startswith('```python'):
                generated_code = generated_code[9:]
            if generated_code.startswith('```'):
                generated_code = generated_code[3:]
            if generated_code.endswith('```'):
                generated_code = generated_code[:-3]
            
            logger.info(f"DeepSeek API成功生成了 {len(generated_code)} 字符的动画代码")
            return generated_code.strip()
            
        except Exception as e:
            logger.error(f"调用DeepSeek API生成动画代码时出错: {e}")
            return ""
    

    
    def _extract_animation_methods_with_timing(self, animation_code: str, sentence_timings: List[Dict]) -> List[Dict]:
        """
        从动画代码中提取动画方法信息并添加时间信息
        """
        animation_methods = []
        
        # 使用正则表达式提取方法名（支持带参数的方法）
        method_pattern = r'def (animate_sentence_\d+)\(self[^)]*\):'
        methods = re.findall(method_pattern, animation_code)
        
        for i, method_name in enumerate(methods):
            timing = sentence_timings[i] if i < len(sentence_timings) else {}
            animation_methods.append({
                'method_name': method_name,
                'start_time': timing.get('start_time', 0.0),
                'duration': timing.get('duration', 3.0),
                'end_time': timing.get('end_time', 3.0),
                'audio_file': timing.get('audio_file', '')
            })
        
        return animation_methods
    
    def synchronize_audio_video(self, task_id: str, audio_track_file: str, video_file: str, sentence_timings: List[Dict]) -> Dict:
        """
        步骤三：音视频同步处理
        
        将音轨与视频进行同步，确保旁白与动画完美配合
        
        Args:
            task_id: 任务ID
            audio_track_file: 音轨文件路径
            video_file: 视频文件路径
            sentence_timings: 每句话的时间信息
            
        Returns:
            Dict: 同步后的结果信息
            {
                "synchronized_video": "path/to/final_video.mp4",
                "sync_quality": "good",  # good/fair/poor
                "timing_adjustments": [...],  # 时间调整记录
                "final_duration": 15.5
            }
        """
        try:
            # 这里应该实现音视频同步逻辑
            # 暂时返回模拟结果
            
            import os
            output_dir = f"output/{task_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            synchronized_video = os.path.join(output_dir, "synchronized_video.mp4")
            
            # 模拟同步处理
            timing_adjustments = []
            for i, timing in enumerate(sentence_timings):
                # 检查是否需要时间调整
                if timing.get('duration', 0) > 5.0:  # 如果单句时间过长
                    timing_adjustments.append({
                        'sentence_index': i,
                        'original_duration': timing.get('duration'),
                        'adjusted_duration': 5.0,
                        'reason': '单句时间过长，进行压缩'
                    })
            
            # 计算最终时长
            final_duration = sum([timing.get('duration', 0) for timing in sentence_timings])
            
            result = {
                "synchronized_video": synchronized_video,
                "sync_quality": "good" if len(timing_adjustments) == 0 else "fair",
                "timing_adjustments": timing_adjustments,
                "final_duration": final_duration
            }
            
            logger.info(f"步骤三完成：音视频同步，最终视频: {synchronized_video}")
            return result
            
        except Exception as e:
            logger.error(f"音视频同步时出错: {e}")
            return {
                "synchronized_video": None,
                "sync_quality": "poor",
                "timing_adjustments": [],
                "final_duration": 0
            }
    
    def generate_complete_video_with_narration(self, task_id: str, request_data: Dict) -> Dict:
        """
        完整的三步骤视频生成流程
        
        步骤一：生成旁白并合成音轨
        步骤二：基于旁白生成manim动画
        步骤三：音视频同步
        
        Args:
            task_id: 任务ID
            request_data: 包含问题和解答数据的字典
            
        Returns:
            Dict: 完整的视频生成结果
            {
                "step1_result": {...},  # 步骤一结果
                "step2_result": {...},  # 步骤二结果
                "step3_result": {...},  # 步骤三结果
                "final_video": "path/to/video.mp4",
                "success": True/False
            }
        """
        try:
            logger.info(f"开始为任务 {task_id} 生成完整视频")
            
            # 步骤一：生成旁白并合成音轨
            logger.info("执行步骤一：生成旁白并合成音轨")
            step1_result = self.generate_narration_with_audio_track(task_id, request_data)
            
            if not step1_result.get("audio_track_file"):
                raise Exception("步骤一失败：无法生成音轨文件")
            
            # 步骤二：基于旁白生成manim动画
            logger.info("执行步骤二：基于旁白生成manim动画")
            # 将task_id添加到request_data中
            request_data_with_task_id = request_data.copy()
            request_data_with_task_id['task_id'] = task_id
            step2_result = self.generate_continuous_animation(
                request_data_with_task_id,
                step1_result["sentence_narration"],
                step1_result["sentence_timings"]
            )
            
            if not step2_result.get("manim_code"):
                raise Exception("步骤二失败：无法生成动画代码")
            
            # 这里应该渲染manim动画为视频文件
            # 暂时模拟视频文件路径
            video_file = f"output/{task_id}/animation.mp4"
            
            # 步骤三：音视频同步
            logger.info("执行步骤三：音视频同步")
            step3_result = self.synchronize_audio_video(
                task_id,
                step1_result["audio_track_file"],
                video_file,
                step1_result["sentence_timings"]
            )
            
            # 整合最终结果
            result = {
                "step1_result": step1_result,
                "step2_result": step2_result,
                "step3_result": step3_result,
                "final_video": step3_result.get("synchronized_video"),
                "success": True
            }
            
            logger.info(f"任务 {task_id} 完整视频生成成功")
            return result
            
        except Exception as e:
            logger.error(f"生成完整视频时出错: {e}")
            return {
                "step1_result": {},
                "step2_result": {},
                "step3_result": {},
                "final_video": None,
                "success": False,
                "error": str(e)
            }
    
    # 保留原有的代码检测和修复功能
    def _detect_code_issues(self, code: str) -> List[str]:
        """
        检测代码中的问题
        """
        issues = []
        
        # 检测VGroup自引用错误
        vgroup_issues = self._detect_vgroup_self_reference(code)
        issues.extend(vgroup_issues)
        
        # 检测语法错误
        syntax_issues = self._detect_syntax_errors(code)
        issues.extend(syntax_issues)
        
        # 检测未定义变量
        undefined_issues = self._detect_undefined_variables(code)
        issues.extend(undefined_issues)
        
        return issues
    
    def _detect_vgroup_self_reference(self, code: str) -> List[str]:
        """
        检测VGroup自引用错误
        """
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # 检测VGroup定义行
            if 'VGroup(' in line and '=' in line:
                var_name = line.split('=')[0].strip()
                
                # 提取VGroup内容（可能跨行）
                vgroup_content = self._extract_vgroup_content(lines, i)
                
                # 检查是否在VGroup定义中引用了自身
                self_ref_pattern = f'{var_name}\\[\\d+\\]'
                if re.search(self_ref_pattern, vgroup_content):
                    issues.append(f"VGroup自引用错误：变量 '{var_name}' 在定义时引用了自身")
        
        return issues
    
    def _extract_vgroup_content(self, lines: List[str], start_line: int) -> str:
        """
        提取VGroup的完整内容（处理跨行情况）
        """
        content = lines[start_line]
        
        # 如果当前行没有闭合括号，继续读取后续行
        open_parens = content.count('(') - content.count(')')
        line_idx = start_line + 1
        
        while open_parens > 0 and line_idx < len(lines):
            content += lines[line_idx]
            open_parens += lines[line_idx].count('(') - lines[line_idx].count(')')
            line_idx += 1
        
        return content
    
    def _detect_syntax_errors(self, code: str) -> List[str]:
        """
        检测语法错误
        """
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"语法错误：第{e.lineno}行 - {e.msg}")
        except Exception as e:
            issues.append(f"代码解析错误：{str(e)}")
        
        return issues
    
    def _detect_undefined_variables(self, code: str) -> List[str]:
        """
        检测可能的未定义变量
        """
        issues = []
        
        # 简单的未定义变量检测
        common_undefined = ['points', 'results', 'summary']
        for var in common_undefined:
            if f'{var}[' in code and f'{var} =' not in code:
                issues.append(f"可能的未定义变量：{var}")
        
        return issues
    
    def _ai_code_review_and_fix(self, code: str, question: str, solution: str) -> str:
        """
        AI代码检测和修复
        """
        try:
            logger.info("开始AI代码检测和修复")
            
            # 检测代码问题
            issues = self._detect_code_issues(code)
            
            if not issues:
                logger.info("代码检测未发现问题")
                return code
             
            logger.info(f"检测到 {len(issues)} 个问题，尝试修复")
             
            # 应用修复
            fixed_code = self._apply_code_fixes(code, issues)
             
            return fixed_code
        except Exception as e:
            logger.error(f"AI代码修复时出错: {e}")
            return code
    
    def cleanup_temp_files(self, task_id: str):
        """
        清理AI服务产生的临时文件
        """
        logger.info(f"任务 {task_id}: 开始AIService临时文件清理")
        
        import glob
        import shutil
        files_deleted = 0
        
        # AI服务可能产生的临时文件模式
        temp_patterns = [
            f"/tmp/ai_temp_{task_id}_*",
            f"/tmp/narration_{task_id}_*",
            f"/tmp/animation_code_{task_id}*",
            f"output/{task_id}/*"
        ]
        
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
        
        # 清理output目录下的任务目录
        task_output_dir = f"output/{task_id}"
        if os.path.exists(task_output_dir):
            try:
                shutil.rmtree(task_output_dir)
                files_deleted += 1
                logger.debug(f"删除任务输出目录: {task_output_dir}")
            except Exception as e:
                logger.warning(f"删除任务输出目录失败 {task_output_dir}: {e}")
        
        logger.info(f"任务 {task_id}: AIService临时文件清理完成，共删除 {files_deleted} 个文件/目录")
    
    def _apply_code_fixes(self, code: str, issues: List[str]) -> str:
        """
        应用代码修复
        """
        fixed_code = code
        
        for issue in issues:
            if "VGroup自引用错误" in issue:
                fixed_code = self._fix_vgroup_self_reference(fixed_code)
            elif "未定义变量" in issue:
                fixed_code = self._fix_undefined_variables(fixed_code)
            elif "语法错误" in issue:
                fixed_code = self._fix_syntax_errors(fixed_code, issue)
        
        return fixed_code
    
    def _fix_vgroup_self_reference(self, code: str) -> str:
        """
        修复VGroup自引用错误
        """
        # 修复VGroup自引用的基本策略：
        # 1. 先创建单个元素
        # 2. 再组合成VGroup
        
        # 查找并修复points自引用
        if 'points = VGroup(' in code and 'points[0]' in code:
            # 替换为分步创建
            fixed_code = re.sub(
                r'points = VGroup\(([^,]+),([^)]+points\[0\][^)]+)\)',
                r'point1 = \1\npoint2 = \2.replace("points[0]", "point1")\npoints = VGroup(point1, point2)',
                code
            )
            return fixed_code
        
        # 类似地处理其他变量
        for var_name in ['results', 'summary']:
            if f'{var_name} = VGroup(' in code and f'{var_name}[0]' in code:
                pattern = f'{var_name} = VGroup\\(([^,]+),([^)]+{var_name}\\[0\\][^)]+)\\)'
                replacement = f'{var_name}_1 = \\1\\n{var_name}_2 = \\2\\n{var_name} = VGroup({var_name}_1, {var_name}_2)'
                code = re.sub(pattern, replacement, code)
        
        return code
    
    def _fix_undefined_variables(self, code: str) -> str:
        """
        修复未定义变量错误
        """
        # 这里可以添加更复杂的变量定义修复逻辑
        return code
    
    def _fix_syntax_errors(self, code: str, issue: str) -> str:
        """
        修复语法错误
        """
        try:
            lines = code.split('\n')
            
            # 修复常见的语法错误
            for i, line in enumerate(lines):
                # 修复未闭合的括号
                if 'Tex("Hello"' in line and not line.strip().endswith(')'):
                    lines[i] = line.replace('Tex("Hello"', 'Tex("Hello")')
                
                # 修复缩进错误
                if line.strip().startswith('def ') and not line.startswith('    '):
                    lines[i] = '    ' + line.strip()
            
            fixed_code = '\n'.join(lines)
            
            # 验证修复后的代码语法
            try:
                compile(fixed_code, '<string>', 'exec')
                logger.info("语法错误修复成功")
                return fixed_code
            except (SyntaxError, NameError) as e:
                logger.warning(f"代码编译检查失败: {e}，返回原代码")
                return code
                
        except Exception as e:
            logger.error(f"修复语法错误时出错: {e}")
            return code