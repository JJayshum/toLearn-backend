#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
腾讯云TTS服务测试脚本
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目路径到Python路径
sys.path.append('/home/devbox/project')

# 加载.env文件
env_path = '/home/devbox/project/.env'
load_dotenv(env_path)
print(f"✅ 已加载环境变量文件: {env_path}")

try:
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.tts.v20190823 import tts_client, models
    print("✅ 腾讯云SDK导入成功")
except ImportError as e:
    print(f"❌ 腾讯云SDK导入失败: {e}")
    print("请确保已安装腾讯云SDK: pip install tencentcloud-sdk-python")
    sys.exit(1)

def test_tts_service():
    """
    测试腾讯云TTS服务
    """
    # 从环境变量获取配置
    secret_id = os.getenv('TENCENT_SECRET_ID', 'your_tencent_secret_id_here')
    secret_key = os.getenv('TENCENT_SECRET_KEY', 'your_tencent_secret_key_here')
    region = os.getenv('TENCENT_REGION', 'ap-beijing')
    
    print(f"调试信息:")
    print(f"  原始 Secret ID: {repr(secret_id)}")
    print(f"  原始 Secret Key: {repr(secret_key)}")
    print(f"  原始 Region: {repr(region)}")
    
    print(f"配置信息:")
    print(f"  Secret ID: {secret_id[:8]}...")
    print(f"  Secret Key: {secret_key[:8]}...")
    print(f"  Region: {region}")
    
    if secret_id == 'your_tencent_secret_id_here' or secret_key == 'your_tencent_secret_key_here':
        print("⚠️  请在.env文件中配置正确的腾讯云密钥")
        return False
    
    try:
        # 实例化一个认证对象
        cred = credential.Credential(secret_id, secret_key)
        
        # 实例化一个http选项
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tts.tencentcloudapi.com"
        
        # 实例化一个client选项
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        
        # 实例化要请求产品的client对象
        client = tts_client.TtsClient(cred, region, clientProfile)
        
        # 实例化一个请求对象
        req = models.TextToVoiceRequest()
        req.Text = "你好，这是腾讯云语音合成测试。"
        req.SessionId = "test_session_001"
        req.Volume = 0
        req.Speed = 0
        req.ProjectId = 0
        req.ModelType = 1
        req.VoiceType = 101002  # 智瑶 - 温暖女声
        req.PrimaryLanguage = 1
        req.SampleRate = 16000
        req.Codec = "wav"
        req.EnableSubtitle = False
        
        print("🔄 正在测试腾讯云TTS服务...")
        
        # 返回的resp是一个TextToVoiceResponse的实例
        resp = client.TextToVoice(req)
        
        if resp.Audio:
            print("✅ 腾讯云TTS服务测试成功！")
            print(f"   音频数据长度: {len(resp.Audio)} bytes")
            return True
        else:
            print("❌ 腾讯云TTS服务返回空音频数据")
            return False
            
    except Exception as e:
        print(f"❌ 腾讯云TTS服务测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 腾讯云TTS服务测试 ===")
    success = test_tts_service()
    if success:
        print("\n🎉 所有测试通过！腾讯云TTS服务配置正确。")
    else:
        print("\n❌ 测试失败，请检查配置。")
        sys.exit(1)