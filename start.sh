#!/bin/bash

# 视频生成API启动脚本

echo "=== 视频生成API启动脚本 ==="

# 检查环境变量文件
if [ ! -f ".env" ]; then
    echo "错误: .env 文件不存在，请复制 .env.example 并配置相应的环境变量"
    exit 1
fi

# 加载环境变量
source .env

# 检查必要的环境变量
required_vars=("DEEPSEEK_API_KEY" "DATABASE_URL" "REDIS_URL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "错误: 环境变量 $var 未设置"
        exit 1
    fi
done

echo "✓ 环境变量检查通过"

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "错误: Docker未运行，请启动Docker服务"
    exit 1
fi

echo "✓ Docker服务检查通过"

# 拉取Manim Docker镜像
echo "正在拉取Manim Docker镜像..."
docker pull manimcommunity/manim:latest

if [ $? -eq 0 ]; then
    echo "✓ Manim Docker镜像拉取成功"
else
    echo "警告: Manim Docker镜像拉取失败，将在运行时自动拉取"
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p videos/final
mkdir -p logs
echo "✓ 目录创建完成"

# 检查数据库连接
echo "检查数据库连接..."
python3 -c "
import os
from sqlalchemy import create_engine
from database import create_tables
try:
    create_tables()
    print('✓ 数据库连接成功，表创建完成')
except Exception as e:
    print(f'错误: 数据库连接失败: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# 检查Redis连接
echo "检查Redis连接..."
python3 -c "
import redis
import os
try:
    r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    r.ping()
    print('✓ Redis连接成功')
except Exception as e:
    print(f'错误: Redis连接失败: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo "=== 所有检查通过，准备启动服务 ==="

# 询问启动模式
echo "请选择启动模式:"
echo "1. 开发模式 (单进程)"
echo "2. 生产模式 (Docker Compose)"
read -p "请输入选择 (1 或 2): " choice

case $choice in
    1)
        echo "启动开发模式..."
        echo "注意: 开发模式下需要手动启动Celery Worker"
        echo "在另一个终端运行: celery -A tasks.video_generation worker --loglevel=info"
        python3 main.py
        ;;
    2)
        echo "启动生产模式..."
        docker-compose up -d
        echo "✓ 服务已启动"
        echo "Web服务: http://localhost:8000"
        echo "Flower监控: http://localhost:5555"
        echo "查看日志: docker-compose logs -f"
        echo "停止服务: docker-compose down"
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

echo "=== 启动完成 ==="