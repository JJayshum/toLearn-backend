#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频生成任务状态监控脚本
"""

import requests
import time
import json

def monitor_task_status(task_id: str, max_wait_time: int = 300):
    """
    监控视频生成任务状态
    
    Args:
        task_id: 任务ID
        max_wait_time: 最大等待时间（秒）
    """
    base_url = "http://localhost:8001"
    status_url = f"{base_url}/api/video_status/{task_id}"
    
    print(f"🔍 开始监控任务: {task_id}")
    print(f"📊 状态查询URL: {status_url}")
    print("=" * 60)
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(status_url)
            
            if response.status_code == 200:
                data = response.json()
                
                status = data.get('status', 'unknown')
                progress = data.get('progress', 0)
                
                print(f"⏰ {time.strftime('%H:%M:%S')} | 状态: {status} | 进度: {progress}%")
                
                if status == 'completed':
                    print("\n🎉 视频生成完成!")
                    print(f"📹 视频URL: {data.get('video_url', 'N/A')}")
                    print(f"🖼️ 缩略图URL: {data.get('thumbnail_url', 'N/A')}")
                    print(f"⏱️ 视频时长: {data.get('duration', 'N/A')}秒")
                    print(f"📦 文件大小: {data.get('file_size', 'N/A')}字节")
                    return data
                    
                elif status == 'failed':
                    print(f"\n❌ 视频生成失败: {data.get('error_message', '未知错误')}")
                    return data
                    
                elif status in ['pending', 'processing']:
                    # 继续等待
                    time.sleep(5)
                    continue
                    
            else:
                print(f"❌ 状态查询失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络请求失败: {e}")
            
        time.sleep(5)
    
    print(f"\n⏰ 监控超时（{max_wait_time}秒）")
    return None

if __name__ == "__main__":
    # 使用最新的任务ID
    task_id = "240abb66-f007-4fce-9e7f-3e30e8110b3e"
    
    print("🎬 视频生成任务监控")
    print("=" * 60)
    
    result = monitor_task_status(task_id, max_wait_time=600)  # 10分钟超时
    
    if result:
        print("\n📋 最终结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n😞 监控结束，未获得最终结果")