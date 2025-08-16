#!/usr/bin/env python3
import requests
import json
import time

# 测试数据
test_data = {
    "question": "求函数 f(x) = x³ - 6x² + 9x + 1 在区间 [0, 4] 上的最大值和最小值",
    "solution_data": {
        "knowledge_points": "导数的应用，函数的极值，闭区间上连续函数的最值",
        "solution_steps": "1. 求导数 f'(x) = 3x² - 12x + 9\n2. 令 f'(x) = 0，解得 x = 1, x = 3\n3. 计算端点和驻点的函数值\n4. 比较得出最值",
        "detailed_solution": "首先求导数：f'(x) = 3x² - 12x + 9 = 3(x² - 4x + 3) = 3(x-1)(x-3)\n\n令 f'(x) = 0，得到驻点 x = 1 和 x = 3\n\n计算关键点的函数值：\nf(0) = 0³ - 6(0)² + 9(0) + 1 = 1\nf(1) = 1³ - 6(1)² + 9(1) + 1 = 1 - 6 + 9 + 1 = 5\nf(3) = 3³ - 6(3)² + 9(3) + 1 = 27 - 54 + 27 + 1 = 1\nf(4) = 4³ - 6(4)² + 9(4) + 1 = 64 - 96 + 36 + 1 = 5\n\n比较各点函数值：f(0) = 1, f(1) = 5, f(3) = 1, f(4) = 5\n\n因此，函数在区间 [0, 4] 上的最大值为 5（在 x = 1 和 x = 4 处取得），最小值为 1（在 x = 0 和 x = 3 处取得）。"
    },
    "video_config": {
        "resolution": "1080p",
        "fps": 30,
        "voice_type": "female",
        "animation_style": "standard"
    }
}

print("发送视频生成请求...")
response = requests.post(
    "http://localhost:8001/api/generate_video",
    json=test_data,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    task_id = result['task_id']
    print(f"任务已提交，任务ID: {task_id}")
    print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # 监控任务进度
    print("\n开始监控任务进度...")
    while True:
        status_response = requests.get(f"http://localhost:8001/api/video_status/{task_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"状态: {status_data['status']}, 进度: {status_data['progress']}%")
            
            if status_data['status'] in ['completed', 'failed']:
                print(f"\n最终结果: {json.dumps(status_data, indent=2, ensure_ascii=False)}")
                break
        
        time.sleep(5)
else:
    print(f"请求失败: {response.status_code}")
    print(response.text)