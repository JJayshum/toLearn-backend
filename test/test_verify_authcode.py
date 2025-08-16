#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证码校验API测试脚本
"""

import requests
import json
import time

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

def main():
    """主测试函数"""
    print("开始测试验证码校验API...")
    
    # 测试手机号
    test_phone = "13888888888"
    
    # 1. 先生成验证码
    print("\n步骤1: 生成验证码")
    if not test_generate_authcode(test_phone):
        print("生成验证码失败，测试终止")
        return
    
    # 等待一秒
    time.sleep(1)
    
    # 2. 测试错误的验证码
    print("\n步骤2: 测试错误验证码")
    success, result = test_verify_authcode(test_phone, "000000")
    if success and result and not result.get('success'):
        print("✓ 错误验证码测试通过")
    else:
        print("✗ 错误验证码测试失败")
    
    # 3. 测试正确的验证码（需要手动输入）
    print("\n步骤3: 测试正确验证码")
    print("请查看服务器日志获取生成的验证码，然后输入:")
    correct_code = input("请输入验证码: ").strip()
    
    if correct_code:
        success, result = test_verify_authcode(test_phone, correct_code)
        if success and result and result.get('success'):
            print("✓ 正确验证码测试通过")
            print(f"用户信息: 手机号={result.get('phoneNumber')}, Pro时间={result.get('pro_time')}")
        else:
            print("✗ 正确验证码测试失败")
    
    # 4. 测试重复验证（应该失败，因为验证码已被使用）
    print("\n步骤4: 测试重复验证")
    if correct_code:
        success, result = test_verify_authcode(test_phone, correct_code)
        if success and result and not result.get('success'):
            print("✓ 重复验证测试通过（正确拒绝已使用的验证码）")
        else:
            print("✗ 重复验证测试失败")
    
    # 5. 测试另一个手机号（新用户注册）
    print("\n步骤5: 测试新用户注册")
    new_phone = "13999999999"
    if test_generate_authcode(new_phone):
        time.sleep(1)
        print("请查看服务器日志获取新手机号的验证码:")
        new_code = input(f"请输入{new_phone}的验证码: ").strip()
        if new_code:
            success, result = test_verify_authcode(new_phone, new_code)
            if success and result and result.get('success'):
                print("✓ 新用户注册测试通过")
                print(f"新用户信息: 手机号={result.get('phoneNumber')}, Pro时间={result.get('pro_time')}")
            else:
                print("✗ 新用户注册测试失败")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()