"""
主测试脚本 - 运行单needle和多needle测试
"""
import argparse
from simple_needle_tester import needle_in_haystack_pipeline


def main():
    parser = argparse.ArgumentParser(description="运行Needle in Haystack测试")
    parser.add_argument("--model", type=str, default="../models/Llama-3.1-8B-Instruct", 
                       help="模型路径")
    parser.add_argument("--test-type", type=str, choices=["single", "multi", "both"], 
                       default="both", help="测试类型")
    parser.add_argument("--attn", type=str, default="flash_attention_2", 
                       choices=["flash_attention_2"],
                       help="注意力机制类型")
    
    args = parser.parse_args()
    
    print(f"模型: {args.model}")
    print(f"测试类型: {args.test_type}")
    print(f"注意力机制: {args.attn}")
    
    results = []
    
    if args.test_type in ["single", "both"]:
        try:
            result = needle_in_haystack_pipeline(
                args.model, "single", args.attn
            )
            results.append(result)
        except Exception as e:
            print(f"单needle测试失败: {e}")
    
    if args.test_type in ["multi", "both"]:
        try:
            result = needle_in_haystack_pipeline(
                args.model, "multi", args.attn
            )
            results.append(result)
        except Exception as e:
            print(f"多needle测试失败: {e}")
    
    if results:
        print(f"\n总结:")
        for result in results:
            print(f"{result['test_type']} - 分数: {result['score']:.2f}")
        
        if len(results) > 1:
            avg_score = sum(r['score'] for r in results) / len(results)
            print(f"平均分数: {avg_score:.2f}")


if __name__ == "__main__":
    main()
