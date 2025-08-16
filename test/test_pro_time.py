#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 pro_time 默认值设置
"""

import requests
import json
import time
from datetime import datetime, timezone, timedelta

# API基础URL
BASE_URL = "http://localhost:8001"

def test_generate_authcode(phone_number):
    """测试生成验证码"""
    url = f"{BASE_URL}/api/generate_authcode"
    data = {
        "phoneNumber": phone_number
    }
    
    print(f"\n=== 测试生成验证码: {phone_number} ===")
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"请求失败: {e}")
        return False

def test_verify_authcode(phone_number, authcode):
    """测试验证码校验"""
    url = f"{BASE_URL}/api/verify_authcode"
    data = {
        "phoneNumber": phone_number,
        "authcode": authcode
    }
    
    print(f"\n=== 测试验证码校验: {phone_number}, 验证码: {authcode} ===")
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return response.status_code == 200, result
    except Exception as e:
        print(f"请求失败: {e}")
        return False, None

def check_pro_time(pro_time_timestamp):
    """检查 pro_time 是否正确设置为当前时间加一天"""
    if not pro_time_timestamp:
        print("❌ pro_time 为空")
        return False
    
    # 转换时间戳为北京时间
    china_tz = timezone(timedelta(hours=8))
    pro_time = datetime.fromtimestamp(pro_time_timestamp, tz=china_tz)
    current_time = datetime.now(china_tz)
    
    print(f"当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Pro到期时间: {pro_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # 计算时间差
    time_diff = pro_time - current_time
    hours_diff = time_diff.total_seconds() / 3600
    
    print(f"时间差: {hours_diff:.2f} 小时")
    
    # 检查是否在23-25小时之间（允许一些误差）
    if 23 <= hours_diff <= 25:
        print("✅ pro_time 设置正确（约24小时后到期）")
        return True
    else:
        print(f"❌ pro_time 设置错误，应该是24小时后，实际是 {hours_diff:.2f} 小时后")
        return False

def main():
    """主测试函数"""
    print("开始测试 pro_time 默认值设置...")
    
    # 使用一个新的测试手机号
    test_phone = "13777777777"
    
    # 1. 生成验证码
    print("\n步骤1: 生成验证码")
    if not test_generate_authcode(test_phone):
        print("生成验证码失败，测试终止")
        return
    
    # 等待一秒
    time.sleep(1)
    
    # 2. 提示用户输入验证码
    print("\n步骤2: 验证码校验")
    print("请查看服务器日志获取生成的验证码")
    authcode = input(f"请输入 {test_phone} 的验证码: ").strip()
    
    if not authcode:
        print("未输入验证码，测试终止")
        return
    
    # 3. 验证码校验
    success, result = test_verify_authcode(test_phone, authcode)
    
    if not success or not result or not result.get('success'):
        print("验证码校验失败，测试终止")
        return
    
    # 4. 检查 pro_time
    print("\n步骤3: 检查 pro_time 设置")
    pro_time = result.get('pro_time')
    
    if check_pro_time(pro_time):
        print("\n🎉 测试通过！新用户的 pro_time 已正确设置为当前时间加一天")
    else:
        print("\n❌ 测试失败！pro_time 设置不正确")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()