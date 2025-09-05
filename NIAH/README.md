# Needle in Haystack 测试

简化的 Needle in Haystack 测试工具，支持单needle和多needle测试。

## 使用方法

```bash
# 运行所有测试
python run_needle_tests.py --model ../models/Llama-3.1-8B-Instruct

# 只运行单needle测试
python run_needle_tests.py --test-type single

# 只运行多needle测试  
python run_needle_tests.py --test-type multi
```

## 文件说明

- `simple_needle_tester.py` - 核心测试工具
- `run_needle_tests.py` - 主测试脚本  
- `PaulGrahamEssays/` - 背景文本文件夹
- `example.py` - 原始模型调用示例
