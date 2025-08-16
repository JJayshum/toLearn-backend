from fastapi import FastAPI, HTTPException, Request, Response, Depends, status
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
from database import get_db, Feedback, AuthCode, UserAccount, create_tables
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from pydantic import BaseModel, validator, Field
from typing import Optional, Dict, Any, Tuple, List
import os
import hashlib
import json
import asyncio
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
import time
import httpx
import logging
import re
import uuid
import random
from dotenv import load_dotenv
import requests
from openai import OpenAI
from PIL import Image
import io
import base64
import hmac
from base64 import b64decode
from celery import Celery
from services.task_manager import TaskManager

# Load environment variables
load_dotenv()

# LangChain 相关导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MathChains:
    def __init__(self, api_key: str):
        # 初始化 OpenAI 客户端，但指向 DeepSeek 的 API 端点
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 使用 ChatOpenAI 作为 LangChain 的接口，增加token限制和超时设置
        self.reasoner = ChatOpenAI(
            openai_api_key=api_key,
            model="deepseek-reasoner",
            openai_api_base="https://api.deepseek.com",
            temperature=0.3,
            max_tokens=4096,  # 修正为DeepSeek官方最大token限制范围内
            request_timeout=600,  # 增加超时时间到600秒
            max_retries=3  # 增加重试次数
        )
        self.chat = ChatOpenAI(
            openai_api_key=api_key,
            model="deepseek-chat",
            openai_api_base="https://api.deepseek.com",
            temperature=0.3,
            max_tokens=2048,  # 修正为DeepSeek官方最大token限制范围内
            request_timeout=600,  # 增加超时时间到600秒
            max_retries=3  # 增加重试次数
        )
        self.output_parser = StrOutputParser()
        
        # 解题提示词 - 详细版本
        self.solver_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的数学解题助手。请按照以下要求解答数学问题：
            1. 仔细阅读并理解题目
            2. 分步骤展示解题过程
            3. 使用LaTeX格式表示所有数学公式
            4. 最终答案用【答案】标记
            6. 保持解答的准确性和专业性
            """),
            ("human", "请解决以下数学问题：\n{question}")
        ])
        
        # 验证提示词 - 优化版
        self.verifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严格的数学验证者，负责检查数学解答的正确性。
            
            请按照以下步骤进行验证:
            1. 检查解题思路是否合理
            2. 验证每一步的数学推导是否正确
            3. 检查计算过程是否有误
            4. 确认最终答案是否准确
            
            验证标准:
            - 数学概念使用是否正确
            - 公式应用是否恰当
            - 计算过程是否准确
            - 推理逻辑是否严密
            
            如果解答完全正确，请只回复:"没有错误"
            
            如果发现错误，请按以下格式回复:
            
            [错误类型]
            具体描述错误类型(如: 计算错误, 公式使用错误, 逻辑错误等)
            
            [错误位置]
            指出错误发生的具体位置
            
            [修正建议]
            提供具体的修改建议或正确解法
            """),
            ("human", """请验证以下数学问题的解答是否正确:
            
            [问题]
            {question}
            
            [解答]
            {solution}
            
            请仔细检查并给出验证结果:""")
        ])
        
        # 格式化提示词 - 增强版
        formatter_system_message = """你是一个专业的数学解答格式化助手。请将以下解答按照要求格式化为三部分，严格遵循以下格式要求：

1. [基础知识点]
要求：
- 列出解决本题最关键的1-5个核心知识点
- 每个知识点必须包含详细解释，说明其定义、性质和在本题中的应用
- 每个知识点应该解释的尽可能详细，使读者阅读后可以完全理解这个知识点
- 知识点的表述应该像老师在讲解一样
- 格式：
  - 知识点名称 详细解释...（不需要标注第几点，知识点名称与解释之间用空格分开，“详细解释”仅仅作为占位符来展示输出格式，并不需要输出“详细解释”，‘- ’需要保留）

2. [解题思路]
要求：
- 详细分析解题思路，仔细分析这个解答中思维链，并且将其以步骤的形式分离
- 对于每一个步骤，都需要结合具体题目，详细解释为什么会想到这样做，哪些情况下可以用这种方法
- 每个步骤都要详细解释思维过程，就像老师在讲解一样
- 读者阅读完你的解题思路后，需要对这道题目有更深层次的理解，而不是仅仅记忆答案
- 格式：
  - 思路概述 详细解释...（不需要标注第几点，思路概述与解释之间用空格分开，“详细解释”仅仅作为占位符来展示输出格式，并不需要输出“详细解释”，‘- ’需要保留）

3. [完整解答]
要求：
- 只包含必要的解题步骤，不添加解释性文字
- 最终答案用[答案]标出
- 格式要像学生的标准解题过程，整洁规范

重要说明：
1. 前两个部分（基础知识点和解题思路）必须非常详细，这是理解题目的关键
2. 第三部分（完整解答）要保持简洁，只包含必要的解题步骤
3. 行内公式使用单个$符号，例如 $E=mc^2$
4. 独立公式使用双$符号，例如 $$\frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{{2a}}$$
5. 严格遵循上述格式要求，不要添加额外内容
6. 保持内容的准确性和专业性
7. 请严格按照要求格式化以上内容，确保格式完全符合规范："""
        
        self.formatter_prompt = ChatPromptTemplate.from_messages([
            ("system", formatter_system_message),
            ("human", "问题: {question}\\n\\n原始解答:\\n{solution}\\n\\n请严格按照要求格式化以上内容，确保格式完全符合规范：")
        ])
        
        # 创建链并添加日志记录
        async def log_solver_io(inputs):
            logger.info(f"Solver input: {inputs}")
            try:
                # 使用 chat 模型替代 reasoner 模型
                result = await self.chat.ainvoke(inputs)
                logger.info(f"Solver raw output: {result}")
                return result
            except Exception as e:
                logger.error(f"调用模型时发生错误: {str(e)}")
                raise
                
        self.solver_chain = self.solver_prompt | log_solver_io | self.output_parser
        self.verifier_chain = self.verifier_prompt | self.chat | self.output_parser
        self.formatter_chain = self.formatter_prompt | self.chat | self.output_parser

def parse_formatted_response(text: str) -> Dict[str, str]:
    """
    解析格式化后的响应，将所有内容整合到 detailed_solution 字段中
    
    Args:
        text: 格式化后的响应文本
        
    Returns:
        Dict[str, str]: 包含 detailed_solution 的字典
    """
    # 清理文本中的多余空白字符
    text = text.strip()
    
    # 使用正则表达式提取各部分
    knowledge_section = ""
    steps_section = ""
    solution_section = ""
    
    # 尝试提取基础知识点
    knowledge_pattern = r'(?:1\. )?\[基础知识点\]([\s\S]*?)(?=(?:2\. )?\[解题思路\]|(?:3\. )?\[完整解答\]|$)'
    knowledge_match = re.search(knowledge_pattern, text)
    if knowledge_match and knowledge_match.group(1).strip():
        knowledge_section = knowledge_match.group(1).strip()
    
    # 尝试提取解题思路
    steps_pattern = r'(?:2\. )?\[解题思路\]([\s\S]*?)(?=(?:3\. )?\[完整解答\]|$)'
    steps_match = re.search(steps_pattern, text)
    if steps_match and steps_match.group(1).strip():
        steps_section = steps_match.group(1).strip()
    
    # 尝试提取完整解答
    solution_pattern = r'(?:3\. )?\[完整解答\]([\s\S]*?)$'
    solution_match = re.search(solution_pattern, text)
    if solution_match and solution_match.group(1).strip():
        solution_section = solution_match.group(1).strip()
    
    # 构建最终的详细解答
    markdown_parts = []
    
    if knowledge_section:
        # 处理基础知识点部分
        knowledge_section = re.sub(r'\n\s*-\s*', '\n- ', knowledge_section)  # 标准化列表格式
        markdown_parts.append(f"## 基础知识点\n{knowledge_section}")
    
    if steps_section:
        # 处理解题思路部分
        steps_section = re.sub(r'\n\s*-\s*', '\n- ', steps_section)  # 标准化列表格式
        markdown_parts.append(f"## 解题思路\n{steps_section}")
    
    if solution_section:
        # 处理完整解答部分
        markdown_parts.append(f"## 完整解答\n{solution_section}")
    
    # 如果没有找到任何匹配，将整个文本作为完整解答
    if not markdown_parts:
        markdown_parts.append(f"## 完整解答\n{text}")
    
    # 合并所有部分
    detailed_solution = '\n\n'.join(markdown_parts)
    
    # 处理可能的格式问题
    detailed_solution = re.sub(r'\n{3,}', '\n\n', detailed_solution)  # 将多个空行减少为一个
    
    return {
        'detailed_solution': detailed_solution if detailed_solution else "未提供解答"
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# 性能监控
class PerformanceMonitor:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0
        self.start_time = time.time()
    
    def record_request(self, processing_time):
        self.request_count += 1
        self.total_processing_time += processing_time
    
    def record_error(self):
        self.error_count += 1
    
    def get_stats(self):
        uptime = time.time() - self.start_time
        avg_time = self.total_processing_time / self.request_count if self.request_count > 0 else 0
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate": (1 - (self.error_count / self.request_count)) * 100 if self.request_count > 0 else 100,
            "average_processing_time": avg_time,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0
        }

# 全局性能监控器
performance_monitor = PerformanceMonitor()

# Initialize FastAPI app
app = FastAPI(
    title="Math Solver API",
    description="API for solving math questions with OCR and AI",
    version="1.0.0",
    openapi_url=None,  # Disable OpenAPI schema generation
    docs_url=None,     # Disable Swagger UI
    redoc_url=None     # Disable ReDoc UI
)

# 挂载静态文件服务
app.mount("/videos", StaticFiles(directory="/home/devbox/project/videos/final"), name="videos")

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()
    logger.info("Database tables created")

# Add middleware
# Add GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global timeout settings
REQUEST_TIMEOUT = 300  # 5 minutes
MODEL_TIMEOUT = 240    # 4 minutes

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "https://tolearn.top"  # 添加前端域名
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # 添加OPTIONS方法支持预检请求
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Authorization",
        "Content-Language",
        "Content-Type",
        "Date",
        "Digest",
        "Host",
        "User-Agent",
        "X-Requested-With"
    ],
)

# 确保 LangChain 模型已正确初始化
try:
    # 测试模型连接
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    test_chains = MathChains(deepseek_api_key)
    logger.info("LangChain 模型初始化成功")
except Exception as e:
    logger.error(f"LangChain 模型初始化失败: {str(e)}")
    raise

# 缓存配置
CACHE_TTL = 3600  # 1小时缓存过期
cache = {}

def get_from_cache(key: str) -> Optional[Dict]:
    """从缓存中获取数据"""
    entry = cache.get(key)
    if entry and not entry.is_expired():
        return entry.data
    return None

def set_in_cache(key: str, data: Dict, ttl: int = 3600):
    """设置缓存数据"""
    cache[key] = CacheEntry(data=data, expiry=time.time() + ttl)

class CacheEntry:
    def __init__(self, data: dict, expiry: float):
        self.data = data
        self.expiry = expiry

    def is_expired(self) -> bool:
        return time.time() > self.expiry

def cache_response(ttl: int = 3600):
    """缓存响应装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            request = None
            solve_request = None
            
            # 查找 FastAPI Request 对象和 SolveQuestionRequest 对象
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                elif isinstance(arg, SolveQuestionRequest):
                    solve_request = arg
            
            # 从 kwargs 中获取
            if not request:
                request = kwargs.get('request')
            if not solve_request and 'solve_request' in kwargs:
                solve_request = kwargs['solve_request']
            
            # 生成缓存键
            if solve_request:
                # 如果是 SolveQuestionRequest，使用问题内容作为缓存键
                cache_key = f"solve_question:{hashlib.md5(solve_request.question.encode()).hexdigest()}"
            elif request:
                # 否则使用请求方法和路径
                cache_key = f"{request.method}:{request.url.path}"
                if request.method == "POST":
                    body = await request.body()
                    cache_key += f":{hashlib.md5(body).hexdigest()}"
            else:
                # 如果没有可用的请求对象，直接调用函数
                return await func(*args, **kwargs)
            
            # 检查缓存
            cached_data = get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"使用缓存响应: {cache_key}")
                return JSONResponse(content=cached_data)
            
            # 调用原始函数
            response = await func(*args, **kwargs)
            
            # 更新缓存
            if hasattr(response, 'body'):
                try:
                    # 解析响应体
                    response_data = json.loads(response.body)
                    # 只保留 detailed_solution 和其他元数据
                    filtered_response = {
                        'detailed_solution': response_data.get('detailed_solution', ''),
                        'request_id': response_data.get('request_id', ''),
                        'timestamp': response_data.get('timestamp', ''),
                        'processing_time': response_data.get('processing_time', 0)
                    }
                    # 更新缓存
                    set_in_cache(cache_key, filtered_response, ttl=ttl)
                except Exception as e:
                    logger.error(f"处理缓存时出错: {str(e)}")
            
            return response
        return wrapper
    return decorator

# Request models
class OCRRequest(BaseModel):
    common: dict
    business: dict
    data: dict

    @validator('common')
    def validate_common(cls, v):
        if 'app_id' not in v:
            raise ValueError("app_id is required in common")
        return v

    @validator('business')
    def validate_business(cls, v):
        if 'ent' not in v or v['ent'] != 'teach-photo-print':
            raise ValueError("ent must be 'teach-photo-print'")
        if 'aue' not in v or v['aue'] != 'raw':
            raise ValueError("aue must be 'raw'")
        return v

    @validator('data')
    def validate_data(cls, v):
        if 'image' not in v:
            raise ValueError("image is required in data")
        try:
            import base64
            base64.b64decode(v['image'])
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 encoded image: {str(e)}")

class OCRResponse(BaseModel):
    extractedText: str = Field(..., description="Extracted text from the image")

    class Config:
        schema_extra = {
            "example": {
                "extractedText": "Extracted text from the image"
            }
        }

class QuestionRequest(BaseModel):
    inputText: str

class SolveQuestionRequest(BaseModel):
    question: str
    
    def dict(self, **kwargs):
        return {"question": self.question}

class SolutionResponse(BaseModel):
    knowledge_points: str
    solution_steps: str
    detailed_solution: str
    request_id: str
    timestamp: str
    processing_time: float

class FeedbackRequest(BaseModel):
    inputFeedback: str

class AuthCodeRequest(BaseModel):
    phoneNumber: str

class AuthCodeResponse(BaseModel):
    success: bool
    message: str
    request_id: str
    timestamp: str

class VerifyAuthCodeRequest(BaseModel):
    phoneNumber: str
    authcode: str

class VerifyAuthCodeResponse(BaseModel):
    success: bool
    phoneNumber: str = None
    pro_time: int = None
    message: str = None
    request_id: str
    timestamp: str

class ProRestTimeRequest(BaseModel):
    phoneNumber: str

class ProRestTimeResponse(BaseModel):
    success: bool
    phoneNumber: str = None
    restTime: int = None  # 剩余秒数
    message: str = None
    request_id: str
    timestamp: str

# 视频生成相关模型
class VideoConfig(BaseModel):
    resolution: str = Field(default="1080p", description="视频分辨率")
    fps: int = Field(default=30, description="帧率")
    duration: Optional[int] = Field(default=None, description="视频时长（秒）")
    voice_type: str = Field(default="female", description="语音类型")
    animation_style: str = Field(default="standard", description="动画风格")

class VideoGenerationRequest(BaseModel):
    question: str = Field(..., description="数学问题")
    solution_data: Dict = Field(..., description="解题数据")
    video_config: VideoConfig = Field(default_factory=VideoConfig, description="视频配置")

class VideoGenerationResponse(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="状态消息")
    request_id: str = Field(..., description="请求ID")
    timestamp: str = Field(..., description="时间戳")

class VideoStatusResponse(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    progress: int = Field(..., description="进度百分比")
    video_url: Optional[str] = Field(None, description="视频URL")
    thumbnail_url: Optional[str] = Field(None, description="缩略图URL")
    duration: Optional[float] = Field(None, description="视频时长")
    file_size: Optional[int] = Field(None, description="文件大小")
    error_message: Optional[str] = Field(None, description="错误信息")
    created_at: Optional[str] = Field(None, description="创建时间")
    started_at: Optional[str] = Field(None, description="开始时间")
    completed_at: Optional[str] = Field(None, description="完成时间")
    message: Optional[str] = Field(None, description="状态消息")

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Question Detection API is running"}

# Feedback endpoint
@app.post("/api/store_feedback")
async def store_feedback(
    feedback_data: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Store user feedback in the database.
    
    Args:
        feedback_data: The feedback data containing 'inputFeedback' field
        db: Database session dependency
        
    Returns:
        dict: Empty response with 201 status code on success
    """
    try:
        # 使用中国时区（UTC+8）
        china_tz = timezone(timedelta(hours=8))
        feedback = Feedback(
            feedback=feedback_data.inputFeedback,
            created_at=datetime.now(china_tz)
        )
        db.add(feedback)
        db.commit()
        logger.info(f"Feedback stored successfully: {feedback_data.inputFeedback[:50]}...")
        return Response(status_code=status.HTTP_201_CREATED)
    except Exception as e:
        db.rollback()
        logger.error(f"Error storing feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store feedback"
        )

# Question detection endpoint
@app.post("/api/judge_question")
async def judge_question(request: QuestionRequest):
    """
    判断输入文本是否是一个题目或问题
    
    Args:
        request (QuestionRequest): 包含输入文本的请求
        
    Returns:
        dict: 包含判断结果的字典
        
    Raises:
        HTTPException: 如果请求处理失败或超时
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] 收到判断题目请求: {request.inputText[:100]}...")
    
    # 检查输入长度
    if not request.inputText or len(request.inputText.strip()) < 3:
        return {
            "is_question": False,
            "request_id": request_id,
            "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "reason": "输入文本过短"
        }
    
    # 设置超时
    timeout = 600.0  # 增加到600秒
    
    try:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            error_msg = "DEEPSEEK_API_KEY 环境变量未设置"
            logger.error(f"[{request_id}] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        # 准备模型的提示词
        prompt = f"""请判断以下文本是否是一个题目或问题。请只回答 'True' 或 'False'。

文本："{request.inputText[:1000]}"

答案（仅回答 True 或 False）："""
        
        logger.info(f"[{request_id}] 准备调用 DeepSeek API")
        
        # 使用带重试的异步客户端
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            transport=httpx.AsyncHTTPTransport(retries=2)  # 重试2次
        ) as client:
            request_data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个帮助判断文本是否为题目的助手。请只回答 'True' 或 'False'。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 5,
                "timeout": timeout - 1  # 确保比客户端超时短
            }
            
            try:
                # 使用asyncio.wait_for添加额外超时保护
                response = await asyncio.wait_for(
                    client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {deepseek_api_key}",
                            "Content-Type": "application/json",
                            "X-Request-ID": request_id
                        },
                        json=request_data
                    ),
                    timeout=timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                # 提取模型响应
                answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
                logger.info(f"[{request_id}] 模型原始响应: {answer}")
                
                # 判断响应是否表示一个问题
                is_question = "true" in answer.lower()
                
                # 构建响应体
                response_data = {
                    "is_question": is_question,
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
                
                logger.info(f"[{request_id}] 返回响应: {response_data}")
                return response_data
                
            except asyncio.TimeoutError:
                error_msg = f"DeepSeek API 请求超时（{timeout}秒）"
                logger.error(f"[{request_id}] {error_msg}")
                raise HTTPException(status_code=504, detail=error_msg)
            except httpx.HTTPStatusError as e:
                error_msg = f"DeepSeek API 调用失败: {e.response.status_code} {e.response.text}"
                logger.error(f"[{request_id}] {error_msg}")
                raise HTTPException(status_code=502, detail=error_msg)
            except json.JSONDecodeError as e:
                error_msg = f"API 响应解析失败: {str(e)}"
                logger.error(f"[{request_id}] {error_msg}", exc_info=True)
                raise HTTPException(status_code=502, detail=error_msg)
            
    except asyncio.TimeoutError:
        error_msg = f"请求处理超时（{timeout}秒）"
        logger.error(f"[{request_id}] {error_msg}")
        raise HTTPException(status_code=504, detail=error_msg)
        
    except Exception as e:
        error_msg = f"处理请求时发生错误: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

# Generate auth code endpoint
@app.post("/api/generate_authcode", response_model=AuthCodeResponse)
async def generate_authcode(
    request: AuthCodeRequest,
    db: Session = Depends(get_db)
):
    """
    生成验证码
    
    Args:
        request: 包含手机号码的请求
        db: 数据库会话依赖
        
    Returns:
        AuthCodeResponse: 包含生成结果的响应
        
    Raises:
        HTTPException: 如果生成验证码失败
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] 收到验证码生成请求: {request.phoneNumber}")
    
    try:
        # 1. 清理过期的验证码记录
        # 使用中国时区（UTC+8）
        china_tz = timezone(timedelta(hours=8))
        current_time = datetime.now(china_tz)
        expired_codes = db.query(AuthCode).filter(
            AuthCode.expires_at <= current_time
        ).all()
        
        for expired_code in expired_codes:
            db.delete(expired_code)
        
        # 2. 删除该手机号的所有旧验证码记录（确保一个手机号最多只有一条记录）
        old_codes = db.query(AuthCode).filter(
            AuthCode.phone_number == request.phoneNumber
        ).all()
        
        for old_code in old_codes:
            db.delete(old_code)
        
        # 3. 生成6位随机验证码
        auth_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # 4. 设置过期时间（5分钟后）
        created_at = current_time
        expires_at = created_at + timedelta(minutes=5)
        
        # 5. 保存新验证码到数据库
        new_auth_code = AuthCode(
            phone_number=request.phoneNumber,
            code=auth_code,
            created_at=created_at,
            expires_at=expires_at,
            is_used=0
        )
        
        db.add(new_auth_code)
        db.commit()
        
        # 6. 记录日志
        logger.info(f"[{request_id}] 验证码生成成功: {request.phoneNumber} -> {auth_code} (有效期至: {expires_at})")
        
        # 7. 构建响应
        processing_time = round(time.time() - start_time, 2)
        response = AuthCodeResponse(
            success=True,
            message=f"验证码已生成并发送到 {request.phoneNumber}",
            request_id=request_id,
            timestamp=datetime.now(china_tz).isoformat()
        )
        
        logger.info(f"[{request_id}] 验证码生成完成，耗时 {processing_time} 秒")
        return response
        
    except Exception as e:
        db.rollback()
        error_msg = f"生成验证码时发生错误: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.post("/api/verify_authcode", response_model=VerifyAuthCodeResponse)
async def verify_authcode(
    request: VerifyAuthCodeRequest,
    db: Session = Depends(get_db)
):
    """
    校对验证码并保存账户信息
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    china_tz = timezone(timedelta(hours=8))
    
    try:
        logger.info(f"[{request_id}] 开始验证码校验: {request.phoneNumber}")
        
        # 1. 查找数据库中该手机号对应的验证码
        current_time = datetime.now(china_tz)
        auth_code_record = db.query(AuthCode).filter(
            AuthCode.phone_number == request.phoneNumber,
            AuthCode.expires_at > current_time,
            AuthCode.is_used == 0
        ).order_by(AuthCode.created_at.desc()).first()
        
        # 2. 验证码校验
        if not auth_code_record:
            logger.warning(f"[{request_id}] 验证码不存在或已过期: {request.phoneNumber}")
            return VerifyAuthCodeResponse(
                success=False,
                message="验证码不存在或已过期",
                request_id=request_id,
                timestamp=current_time.isoformat()
            )
        
        if auth_code_record.code != request.authcode:
            logger.warning(f"[{request_id}] 验证码错误: {request.phoneNumber}")
            return VerifyAuthCodeResponse(
                success=False,
                message="验证码错误",
                request_id=request_id,
                timestamp=current_time.isoformat()
            )
        
        # 3. 验证码正确，标记为已使用
        auth_code_record.is_used = 1
        db.commit()
        
        logger.info(f"[{request_id}] 验证码校验成功: {request.phoneNumber}")
        
        # 4. 检查用户账户是否存在
        user_account = db.query(UserAccount).filter(
            UserAccount.phone_number == request.phoneNumber
        ).first()
        
        if user_account:
            # 用户已存在，返回现有账户信息
            logger.info(f"[{request_id}] 用户已存在: {request.phoneNumber}, pro_time: {user_account.pro_time}")
            return VerifyAuthCodeResponse(
                success=True,
                phoneNumber=user_account.phone_number,
                pro_time=user_account.pro_time,
                message="验证成功",
                request_id=request_id,
                timestamp=current_time.isoformat()
            )
        else:
            # 新用户，创建账户（使用数据库默认值）
            new_user = UserAccount(
                phone_number=request.phoneNumber,
                register_time=current_time
            )
            db.add(new_user)
            db.commit()
            
            logger.info(f"[{request_id}] 新用户注册成功: {request.phoneNumber}")
            return VerifyAuthCodeResponse(
                success=True,
                phoneNumber=new_user.phone_number,
                pro_time=new_user.pro_time,
                message="验证成功，新用户注册完成",
                request_id=request_id,
                timestamp=current_time.isoformat()
            )
            
    except Exception as e:
        db.rollback()
        logger.error(f"[{request_id}] 验证码校验失败: {str(e)}")
        return VerifyAuthCodeResponse(
             success=False,
             message="服务器内部错误",
             request_id=request_id,
             timestamp=datetime.now(china_tz).isoformat()
         )

@app.get("/api/pro_resttime", response_model=ProRestTimeResponse)
async def get_pro_resttime(
    phoneNumber: str,
    db: Session = Depends(get_db)
):
    """
    获取用户会员权益的剩余天数
    """
    request_id = str(uuid.uuid4())
    china_tz = timezone(timedelta(hours=8))
    current_time = datetime.now(china_tz)
    
    try:
        logger.info(f"[{request_id}] 查询会员剩余时间，手机号: {phoneNumber}")
        
        # 查询用户账户
        user_account = db.query(UserAccount).filter(UserAccount.phone_number == phoneNumber).first()
        
        if not user_account:
            logger.warning(f"[{request_id}] 用户不存在: {phoneNumber}")
            return ProRestTimeResponse(
                success=False,
                message="用户不存在",
                request_id=request_id,
                timestamp=current_time.isoformat()
            )
        
        # 获取当前时间的Unix时间戳
        current_timestamp = int(current_time.timestamp())
        pro_time = user_account.pro_time
        
        # 计算剩余时间
        if pro_time > current_timestamp:
            rest_time = pro_time - current_timestamp
            logger.info(f"[{request_id}] 会员剩余时间: {rest_time}秒")
        else:
            rest_time = 0
            logger.info(f"[{request_id}] 会员已过期")
        
        return ProRestTimeResponse(
            success=True,
            phoneNumber=phoneNumber,
            restTime=rest_time,
            message="查询成功",
            request_id=request_id,
            timestamp=current_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] 查询会员剩余时间失败: {str(e)}")
        return ProRestTimeResponse(
            success=False,
            message="服务器内部错误",
            request_id=request_id,
            timestamp=datetime.now(china_tz).isoformat()
        )

# Xunfei OCR API configuration
XUNFEI_API_URL = "https://rest-api.xfyun.cn/v2/itr"

# Helper functions for Xunfei API
def httpdate(dt):
    """
    Return a string representation of a date according to RFC 1123
    (HTTP/1.1).
    """
    weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
             "Oct", "Nov", "Dec"][dt.month - 1]
    return "%s, %02d %s %04d %02d:%02d:%02d GMT" % (weekday, dt.day, month,
                                                    dt.year, dt.hour, dt.minute, dt.second)

def hashlib_256(res: str):
    """Generate SHA-256 digest with base64 encoding."""
    m = hashlib.sha256(bytes(res.encode(encoding='utf-8'))).digest()
    result = "SHA-256=" + base64.b64encode(m).decode(encoding='utf-8')
    return result

def generate_signature(host: str, date: str, digest: str, api_secret: str):
    """Generate HMAC signature for Xunfei API authentication."""
    signatureStr = "host: " + host + "\n"
    signatureStr += "date: " + date + "\n"
    signatureStr += "POST /v2/itr HTTP/1.1\n"
    signatureStr += "digest: " + digest
    
    signature = hmac.new(
        bytes(api_secret.encode(encoding='utf-8')),
        bytes(signatureStr.encode(encoding='utf-8')),
        digestmod=hashlib.sha256
    ).digest()
    
    return base64.b64encode(signature).decode(encoding='utf-8')

def prepare_request_body(image_base64: str):
    """Prepare the request body for Xunfei OCR API with fixed parameters."""
    return {
        "common": {
            "app_id": os.getenv('XUNFEI_APPID')
        },
        "business": {
            "ent": "teach-photo-print",
            "aue": "raw"
        },
        "data": {
            "image": image_base64
        }
    }

# Add CORS middleware
# 重复的CORS中间件已删除
# 添加 GZip 压缩中间件
app.add_middleware(GZipMiddleware)

# Extract question endpoint
@app.post(
    "/api/extract_question",
    response_model=OCRResponse,
    summary="Extract text from image using Xunfei OCR API",
    description="""
    Extract text from an image using the Xunfei OCR API. The frontend only needs to provide the image data,
    all other parameters are handled internally by the backend.
    
    Required Request Body Format:
    ```json
    {
        "data": {
            "image": "base64_encoded_image_string"
        }
    }
    ```
    
    Response Format:
    ```json
    {
        "extractedText": "Extracted text from the image"
    }
    ```
    
    Error Responses:
    - 422: Validation Error - Invalid request format
    - 400: Bad Request - Invalid image format or OCR API error
    - 500: Internal Server Error - Failed to process request
    """
)
async def extract_question(request: Request):
    """
    Extract question text from an image using Xunfei OCR API
    
    Args:
        request (Request): FastAPI request containing OCRRequest body
        
    Returns:
        OCRResponse: Response containing extracted question text
        
    Raises:
        HTTPException: If image processing fails or OCR API returns error
    """
    try:
        # Get request body
        try:
            body = await request.json()
            
            # Validate required fields
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
                
            if 'data' not in body or 'image' not in body['data']:
                raise ValueError("Missing image data")
                
            # Decode base64 and check image format
            image_data = b64decode(body['data']['image'])
            image = Image.open(io.BytesIO(image_data))
            if image.format not in ['JPEG', 'PNG', 'BMP']:
                raise ValueError(f"Unsupported image format: {image.format}")
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request data: {str(e)}"
            )

        # Get API credentials from environment
        app_id = os.getenv('XUNFEI_APPID')
        app_key = os.getenv('XUNFEI_APPKEY')
        app_secret = os.getenv('XUNFEI_APPSECRET')
        
        if not all([app_id, app_key, app_secret]):
            raise HTTPException(
                status_code=500,
                detail="Xunfei API credentials not configured"
            )

        # Get current time in GMT format
        curTime_utc = datetime.utcnow()
        date = httpdate(curTime_utc)
        
        # Prepare request body with fixed parameters
        xunfei_body = prepare_request_body(body['data']['image'])
        body_str = json.dumps(xunfei_body)
        
        # Generate signature
        digest = hashlib_256(body_str)
        signature = generate_signature(
            host="rest-api.xfyun.cn",
            date=date,
            digest=digest,
            api_secret=app_secret
        )
        
        # Prepare headers
        authHeader = 'api_key="%s", algorithm="%s", ' \
                     'headers="host date request-line digest", ' \
                     'signature="%s"' % (app_key, "hmac-sha256", signature)
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Method": "POST",
            "Host": "rest-api.xfyun.cn",
            "Date": date,
            "Digest": digest,
            "Authorization": authHeader
        }

        # Make request to Xunfei API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                XUNFEI_API_URL,
                data=body_str,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Xunfei API Error: {response.text}"
                )
                
            result = response.json()
            
            # Process response
            if result.get('code') != 0:
                error_msg = result.get('message', 'Unknown error')
                raise HTTPException(
                    status_code=400,
                    detail=f"Xunfei API Error: {error_msg}"
                )
                
            # Extract text from response
            extracted_text = ""
            if result.get('data') and 'region' in result['data']:
                for region in result['data']['region']:
                    if region.get('type') == 'text':
                        content = region.get('recog', {}).get('content', '')
                        if content:
                            extracted_text += content + " "

            # Clean up extracted text
            extracted_text = extracted_text.strip()
            
            return OCRResponse(
                extractedText=extracted_text
            )
            
    except HTTPException as e:
        raise
    except Exception as e:
        logging.error(f"Error extracting question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract question text: {str(e)}"
        )

# Solve question endpoint
@app.post("/api/solve_question")
@cache_response(ttl=3600)  # 缓存1小时
async def solve_question(solve_request: SolveQuestionRequest, request: Request):
    """
    使用 LangChain 和 DeepSeek 模型解决数学问题
    
    Args:
        solve_request (SolveQuestionRequest): 包含待解决问题的请求
        
    Returns:
        dict: 包含解答详情的字典
        
    Raises:
        HTTPException: 如果请求处理失败或超时
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] 开始处理请求: {solve_request.question[:100]}...")
    
    # 设置各阶段超时（秒）
    timeouts = {
        "solve": 600.0,      # 解题阶段：10分钟
        "verify": 300.0,     # 验证阶段：5分钟
        "format": 60.0       # 格式化阶段：1分钟
    }
    
    try:
        # 1. 输入验证
        if not solve_request.question or len(solve_request.question.strip()) < 3:
            error_msg = "问题不能为空或过短"
            logger.warning(f"[{request_id}] {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
            
        # 2. 检查缓存（测试期间禁用）
        cache_key = f"solve_question:{hashlib.md5(solve_request.question.encode()).hexdigest()}"
        # 测试期间禁用缓存
        # cached_response = get_from_cache(cache_key)
        # if cached_response:
        #     logger.info(f"[{request_id}] 使用缓存响应: {cache_key}")
        #     return cached_response
        
        # 3. 初始化 MathChains
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            error_msg = "DEEPSEEK_API_KEY 环境变量未设置"
            logger.error(f"[{request_id}] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
            
        # 4. 创建链式处理任务
        async def process_question():
            nonlocal solve_request, request_id, timeouts
            
            # 初始化链
            chains = MathChains(deepseek_api_key)
            
            # 4.1 解题阶段
            try:
                logger.info(f"[{request_id}] 开始解题阶段...")
                logger.info(f"[{request_id}] 问题: {solve_request.question}")
                
                # 确保问题被正确传递
                input_data = {"question": str(solve_request.question).strip()}
                logger.info(f"[{request_id}] 传递给solver_chain的数据: {input_data}")
                
                # 获取并记录完整的提示词
                prompt_value = chains.solver_prompt.format_messages(**input_data)
                logger.info(f"[{request_id}] 完整提示词: {prompt_value}")
                
                try:
                    # 调用solver_chain
                    solution = await asyncio.wait_for(
                        chains.solver_chain.ainvoke(input_data),
                        timeout=timeouts["solve"]
                    )
                    
                    logger.info(f"[{request_id}] 解题阶段完成，获取到初步解答")
                    logger.info(f"[{request_id}] 原始解答: {solution}")
                    
                    # 确保solution是字符串类型
                    if not isinstance(solution, str):
                        solution = str(solution)
                    
                    # 如果solution为空或太短，可能是API调用问题
                    if not solution or len(solution.strip()) < 10:
                        error_msg = f"获取到的解答过短或无效: {solution}"
                        logger.error(f"[{request_id}] {error_msg}")
                        raise HTTPException(status_code=500, detail=error_msg)
                        
                except Exception as e:
                    logger.error(f"[{request_id}] 调用solver_chain时发生错误: {str(e)}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"调用解题模型时发生错误: {str(e)}")
                    
            except asyncio.TimeoutError:
                error_msg = f"解题阶段超时（{timeouts['solve']}秒）"
                logger.error(f"[{request_id}] {error_msg}")
                raise HTTPException(status_code=504, detail=error_msg)
            except Exception as e:
                error_msg = f"解题阶段出错: {str(e)}"
                logger.error(f"[{request_id}] {error_msg}", exc_info=True)
                raise HTTPException(status_code=500, detail=error_msg)
            
            # 4.2 验证阶段（最多重试2次）
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    logger.info(f"[{request_id}] 验证尝试 {attempt + 1}/{max_attempts}")
                    verification = await asyncio.wait_for(
                        chains.verifier_chain.ainvoke({
                            "question": solve_request.question,
                            "solution": solution
                        }),
                        timeout=timeouts["verify"]
                    )
                    
                    if "没有错误" in verification:
                        logger.info(f"[{request_id}] 验证通过，没有发现错误")
                        break
                        
                    logger.warning(f"[{request_id}] 验证发现错误: {verification}")
                    if attempt < max_attempts - 1:
                        # 使用更短的超时进行重试
                        solution = await asyncio.wait_for(
                            chains.solver_chain.ainvoke({
                                "question": f"修正以下解答中的错误：{solution}。错误：{verification}"
                            }),
                            timeout=timeouts["solve"] / 2  # 重试时使用更短的超时
                        )
                except asyncio.TimeoutError:
                    logger.warning(f"[{request_id}] 验证阶段超时，继续处理")
                    break
                except Exception as e:
                    logger.warning(f"[{request_id}] 验证阶段出错，继续处理: {str(e)}")
                    break
            
            # 4.3 格式化阶段
            try:
                logger.info(f"[{request_id}] 开始格式化阶段...")
                formatted = await asyncio.wait_for(
                    chains.formatter_chain.ainvoke({
                        "question": solve_request.question,
                        "solution": solution
                    }),
                    timeout=timeouts["format"]
                )
                
                # 解析格式化后的响应
                formatted_response = parse_formatted_response(formatted)
                
                # 构建响应，只包含 detailed_solution 和其他元数据
                result = {
                    "detailed_solution": formatted_response['detailed_solution'],
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    "processing_time": round(time.time() - start_time, 2)
                }
                
                # 缓存结果
                set_in_cache(cache_key, result)
                
                logger.info(f"[{request_id}] 请求处理完成，耗时 {result['processing_time']} 秒")
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"格式化阶段超时（{timeouts['format']}秒）"
                logger.error(f"[{request_id}] {error_msg}")
                raise HTTPException(status_code=504, detail=error_msg)
            except Exception as e:
                error_msg = f"格式化阶段出错: {str(e)}"
                logger.error(f"[{request_id}] {error_msg}", exc_info=True)
                raise HTTPException(status_code=500, detail=error_msg)
        
        # 5. 执行处理任务，带全局超时
        try:
            return await asyncio.wait_for(
                process_question(),
                timeout=timeouts["solve"] + timeouts["verify"] + timeouts["format"] + 30  # 各阶段超时之和 + 30秒缓冲
            )
        except asyncio.TimeoutError:
            error_msg = f"请求处理总时间超过限制"
            logger.error(f"[{request_id}] {error_msg}")
            raise HTTPException(status_code=504, detail=error_msg)
            
    except HTTPException as he:
        # 记录性能指标
        processing_time = round(time.time() - start_time, 2)
        performance_monitor.record_request(processing_time)
        if he.status_code >= 500:
            performance_monitor.record_error()
        logger.error(f"[{request_id}] 请求处理失败: {he.detail} (耗时: {processing_time}秒)")
        raise
        
    except Exception as e:
        # 记录未捕获的异常
        error_msg = f"处理请求时发生意外错误: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}", exc_info=True)
        performance_monitor.record_error()
        
        # 返回标准化的错误响应
        return JSONResponse(
            status_code=500,
            content={
                "error": "服务器内部错误",
                "request_id": request_id,
                "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "details": str(e) if os.getenv("ENV") == "development" else None
            }
        )

@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "performance": performance_monitor.get_stats()
    }

# 视频生成API端点
@app.post("/api/generate_video", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    db: Session = Depends(get_db)
):
    """
    生成数学问题解答视频
    """
    request_id = str(uuid.uuid4())[:8]
    task_id = str(uuid.uuid4())
    
    try:
        logger.info(f"[{request_id}] 收到视频生成请求，任务ID: {task_id}")
        
        # 初始化任务管理器
        task_manager = TaskManager()
        
        # 创建任务记录
        task = task_manager.create_task(
            task_id=task_id,
            question=request.question,
            solution_data=request.solution_data,
            video_config=request.video_config.dict()
        )
        
        # 导入Celery应用并提交任务
        from celery_config import celery_app
        
        # 异步提交视频生成任务
        celery_app.send_task(
            'tasks.video_generation.generate_video_task',
            args=[
                task_id,
                request.question,
                request.solution_data,
                request.video_config.dict()
            ],
            task_id=task_id
        )
        
        logger.info(f"[{request_id}] 视频生成任务已提交到队列")
        
        return VideoGenerationResponse(
            task_id=task_id,
            status="pending",
            message="视频生成任务已提交，请使用task_id查询进度",
            request_id=request_id,
            timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] 提交视频生成任务失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"提交视频生成任务失败: {str(e)}"
        )

@app.get("/api/video_status/{task_id}", response_model=VideoStatusResponse)
async def get_video_status(task_id: str):
    """
    查询视频生成任务状态
    """
    try:
        logger.info(f"查询任务状态: {task_id}")
        
        # 初始化任务管理器
        task_manager = TaskManager()
        
        # 获取任务状态
        status_data = task_manager.get_task_status_response(task_id)
        
        return VideoStatusResponse(**status_data)
        
    except Exception as e:
        logger.error(f"查询任务状态失败 {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"查询任务状态失败: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='启动Math AI后端服务')
    parser.add_argument('--host', type=str, default=os.getenv("HOST", "0.0.0.0"),
                      help='服务器监听地址')
    parser.add_argument('--port', type=int, default=int(os.getenv("PORT", 8001)),
                      help='服务器监听端口')
    args = parser.parse_args()
    
    # 配置日志
    logger.info("启动服务器...")
    logger.info(f"服务器地址: {args.host}:{args.port}")
    logger.info(f"DeepSeek API 端点: {os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions')}")
    
    try:
        # 启动服务器
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            timeout_keep_alive=300  # 5分钟保持连接
        )
    except Exception as e:
        logger.error(f"启动服务器失败: {str(e)}")
        raise
