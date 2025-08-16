#!/usr/bin/env python3
"""
启动验证码API服务器
"""

import uvicorn
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """
    启动FastAPI服务器
    """
    print("启动验证码API服务器...")
    print("=" * 50)
    
    # 服务器配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    
    print(f"服务器地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs (如果启用)")
    print(f"健康检查: http://{host}:{port}/")
    print(f"验证码API: http://{host}:{port}/api/generate_authcode")
    print("=" * 50)
    
    try:
        # 启动服务器
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,  # 开发模式，文件变化时自动重载
            log_level="info",
            timeout_keep_alive=300
        )
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动服务器失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()