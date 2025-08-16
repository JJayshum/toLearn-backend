#!/usr/bin/env python3
"""
验证码API测试脚本
"""

import requests
import json

def test_generate_authcode():
    """
    测试验证码生成API
    """
    url = "http://localhost:8002/api/generate_authcode"
    
    # 测试不同的手机号码
    test_phones = [
        "13800138000",
        "15912345678",
        "18888888888"
    ]
    
    print("测试验证码生成API...")
    print(f"请求URL: {url}")
    
    for phone in test_phones:
        test_data = {
            "phoneNumber": phone
        }
        
        print(f"\n测试手机号: {phone}")
        print(f"请求数据: {json.dumps(test_data, ensure_ascii=False, indent=2)}")
        
        try:
            response = requests.post(url, json=test_data)
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            
            if response.status_code == 200:
                print("✅ 验证码生成成功！")
            else:
                print("❌ 验证码生成失败！")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ 响应解析失败: {e}")

def test_duplicate_request():
    """
    测试同一手机号重复请求（应该覆盖旧验证码）
    """
    url = "http://localhost:8002/api/generate_authcode"
    phone = "13900139000"
    
    print("\n测试同一手机号重复请求...")
    
    test_data = {"phoneNumber": phone}
    
    # 第一次请求
    print(f"\n第一次请求手机号: {phone}")
    try:
        response1 = requests.post(url, json=test_data)
        print(f"响应状态码: {response1.status_code}")
        print(f"响应内容: {json.dumps(response1.json(), ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"❌ 第一次请求失败: {e}")
    
    # 第二次请求（应该覆盖第一次的验证码）
    print(f"\n第二次请求同一手机号: {phone}")
    try:
        response2 = requests.post(url, json=test_data)
        print(f"响应状态码: {response2.status_code}")
        print(f"响应内容: {json.dumps(response2.json(), ensure_ascii=False, indent=2)}")
        
        if response2.status_code == 200:
            print("✅ 重复请求处理成功！旧验证码应该已被删除")
        else:
            print("❌ 重复请求处理失败！")
    except Exception as e:
        print(f"❌ 第二次请求失败: {e}")

if __name__ == "__main__":
    print("开始测试验证码API...")
    print("请确保服务器正在运行: python start_server.py")
    print("=" * 50)
    
    test_generate_authcode()
    test_duplicate_request()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n注意：验证码已在数据库中生成，请直接查看数据库获取验证码。")