import asyncio
import os
from main import MathChains, parse_formatted_response

async def test_formatter():
    # 初始化 MathChains
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return
    
    chains = MathChains(api_key)
    
    # 测试问题
    question = """证明：若函数f在开区间(a,b)上一致连续，则f在(a,b)上有界。"""
    
    # 先获取解答
    print("获取初始解答...")
    solution = await chains.solver_chain.ainvoke({"question": question})
    print("初始解答获取完成。")
    
    # 使用格式化器格式化解答
    print("\n正在格式化解答...")
    formatted = await chains.formatter_chain.ainvoke({
        "question": question,
        "solution": solution
    })
    
    # 解析并打印结果
    print("\n" + "="*80)
    print("格式化后的解答：")
    print("="*80)
    
    # 解析格式化后的响应
    parsed = parse_formatted_response(formatted)
    print(parsed['detailed_solution'])

if __name__ == "__main__":
    asyncio.run(test_formatter())
