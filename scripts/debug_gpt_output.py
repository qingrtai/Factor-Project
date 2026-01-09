# -*- coding: utf-8 -*-
# debug_gpt_output.py
"""
调试脚本：查看 GPT 实际生成的内容

运行方法：
python debug_gpt_output.py
"""

import json
import re
from common.gpt_runner import call_gpt
from common.column_desc import COLUMN_DESC

ALLOWED_FIELDS = list(COLUMN_DESC.keys())
_FIELDS_FOR_PROMPT = ", ".join(ALLOWED_FIELDS[:20])  # 只取前 20 个避免太长

# 模拟 baseline 因子
positives = [
    {
        "code": "data['factor_score'] = np.where(data['revtq']==0, 0, (data['niq'] - data['txpq']) / data['revtq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)",
        "train_score": 1.01,
        "val_score": 5.94
    },
    {
        "code": "data['factor_score'] = np.where(data['lctq']==0, 0, (data['cheq'] + data['rectq']) / data['lctq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)",
        "train_score": 0.36,
        "val_score": 5.12
    },
]

# 构建 prompt
prompt = f"""You are generating factor formulas. Follow these examples EXACTLY:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP PERFORMING FACTORS (COPY THEIR STRUCTURE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#1  Train=1.0100  Val=5.9400
data['factor_score'] = np.where(data['revtq']==0, 0, (data['niq'] - data['txpq']) / data['revtq'])
data['factor_score'] = data['factor_score'].fillna(0)

#2  Train=0.3600  Val=5.1200
data['factor_score'] = np.where(data['lctq']==0, 0, (data['cheq'] + data['rectq']) / data['lctq'])
data['factor_score'] = data['factor_score'].fillna(0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL: MANDATORY STRUCTURE (NO EXCEPTIONS)  ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVERY factor MUST follow this EXACT 2-line structure:

Line 1: data['factor_score'] = np.where(data['DENOM']==0, 0, EXPRESSION/data['DENOM'])
Line 2: data['factor_score'] = data['factor_score'].fillna(0)

Where:
- DENOM = denominator field (revtq, saleq, atq, etc.)
- EXPRESSION = numerator (can be: single field, sum, difference, etc.)

❌ ABSOLUTELY FORBIDDEN (WILL BE REJECTED):

1. ONE-LINE formats like:
   data['factor_score'] = (data['niq']-data['txpq']) / (data['revtq'] + 1e-8)
   ❌ WRONG - missing np.where, missing second line

2. Using (denom + 1e-8) instead of np.where:
   ❌ WRONG - must use np.where

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate 3 factors using the MANDATORY 2-line np.where structure above.

Available fields: {_FIELDS_FOR_PROMPT}

Output format (JSON with \\n for line breaks):

[
  {{"code": "data['factor_score'] = np.where(data['saleq']==0, 0, (data['ibq']-data['txpq'])/data['saleq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)"}},
  {{"code": "data['factor_score'] = np.where(data['atq']==0, 0, data['revtq']/data['atq'])\\ndata['factor_score'] = data['factor_score'].fillna(0)"}}
]

Start output with '[' immediately. No explanations. Exactly 3 factors.
REMEMBER: Every factor MUST have np.where. No exceptions.""".strip()

print("="*60)
print("发送给 GPT 的 Prompt:")
print("="*60)
print(prompt[:1000] + "...")
print()

print("="*60)
print("调用 GPT (temperature=0.40)...")
print("="*60)

try:
    response = call_gpt(prompt=prompt, temperature=0.40, max_tokens=2200)
    
    print("\n" + "="*60)
    print("GPT 原始响应:")
    print("="*60)
    print(response)
    print()
    
    # 尝试解析
    print("="*60)
    print("解析结果:")
    print("="*60)
    
    try:
        data = json.loads(response.strip())
        if isinstance(data, list):
            print(f"✅ 成功解析 JSON，共 {len(data)} 个因子")
            
            for i, item in enumerate(data, 1):
                code = item.get("code", "")
                print(f"\n因子 #{i}:")
                print(f"  原始代码: {repr(code)}")
                
                # 检查特征
                has_np_where = 'np.where' in code
                has_newline = '\\n' in code
                has_fillna = 'fillna' in code
                
                print(f"  - 包含 np.where: {has_np_where}")
                print(f"  - 包含 \\n: {has_newline}")
                print(f"  - 包含 fillna: {has_fillna}")
                
                if has_newline:
                    # 替换 \n 看实际效果
                    actual_code = code.replace('\\n', '\n')
                    print(f"  实际格式:")
                    for line in actual_code.split('\n'):
                        print(f"    {line}")
    except Exception as e:
        print(f"❌ JSON 解析失败: {e}")
        print("\n尝试提取代码片段...")
        
        # 查找所有 data['factor_score'] 行
        lines = response.split('\n')
        code_lines = [l for l in lines if "data['factor_score']" in l]
        
        if code_lines:
            print(f"找到 {len(code_lines)} 行包含 data['factor_score']:")
            for i, line in enumerate(code_lines, 1):
                print(f"  {i}. {line}")
        else:
            print("未找到任何 data['factor_score'] 代码")

except Exception as e:
    print(f"\n❌ 调用 GPT 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("调试完成")
print("="*60)
