"""简化的Needle in Haystack测试工具"""
import torch
import glob
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from moba import register_moba, MoBAConfig


class SimpleNeedleTester:
    """简化的Needle测试类"""
    
    def __init__(self, model_path, haystack_dir="PaulGrahamEssays"):
        self.model_path = model_path
        self.haystack_dir = haystack_dir
        
        # 初始化tokenizer（用于文本处理）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode_text_to_tokens(self, text):
        """将文本编码为token列表"""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode_tokens(self, tokens, max_length=None):
        """将token列表解码为文本"""
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def read_context_files(self):
        """读取haystack文件内容"""
        context = ""
        base_dir = os.path.abspath(os.path.dirname(__file__))
        
        for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
            with open(file, 'r', encoding='utf-8') as f:
                context += f.read() + "\n\n"
        return context
    
    def encode_and_trim(self, context, context_length):
        """编码文本并修剪到指定长度"""
        tokens = self.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens[:context_length])
        return context
    
    def insert_single_needle(self, context, needle, depth_percent, context_length):
        """在指定深度插入单个needle"""
        tokens_needle = self.encode_text_to_tokens(needle)
        tokens_context = self.encode_text_to_tokens(context)
        
        # 预留buffer空间
        final_context_length_buffer = 200
        context_length -= final_context_length_buffer
        
        # 如果context + needle超过长度限制，缩减context
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]
        
        if depth_percent == 100:
            # 如果深度是100%，将needle放在最后
            tokens_new_context = tokens_context + tokens_needle
        else:
            # 计算插入位置
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            
            # 寻找句号位置，确保在句子边界插入
            tokens_new_context = tokens_context[:insertion_point]
            period_tokens = self.encode_text_to_tokens('.')
            
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]
            
            # 插入needle
            tokens_new_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
        
        return self.decode_tokens(tokens_new_context)
    
    def insert_multiple_needles(self, context, needles, depth_percent, context_length):
        """在指定深度范围内插入多个needles"""
        tokens_context = self.encode_text_to_tokens(context)
        final_context_length_buffer = 200
        context_length -= final_context_length_buffer
        
        # 计算所有needles的总长度
        total_needles_length = sum(len(self.encode_text_to_tokens(needle)) for needle in needles)
        
        # 确保context长度能容纳所有needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
        
        # 计算needles之间的间隔
        depth_percent_interval = (100 - depth_percent) / len(needles)
        current_depth = depth_percent
        
        # 依次插入每个needle
        for needle in needles:
            tokens_needle = self.encode_text_to_tokens(needle)
            
            if current_depth == 100:
                tokens_context = tokens_context + tokens_needle
            else:
                insertion_point = int(len(tokens_context) * (current_depth / 100))
                tokens_new_context = tokens_context[:insertion_point]
                
                # 寻找句号位置
                period_tokens = self.encode_text_to_tokens('.')
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]
                
                # 插入needle
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
            
            current_depth += depth_percent_interval
        
        return self.decode_tokens(tokens_context)
    
    def generate_prompt(self, context, question):
        """生成用于模型的prompt"""
        prompt = f"""请仔细阅读以下文档，然后回答问题。

文档内容：
{context}

问题：{question}

请基于文档内容回答："""
        return prompt
    
    def evaluate_response(self, response, needle_or_needles):
        if not response:
            return 0.0
        
        response_lower = response.lower()
        
        if isinstance(needle_or_needles, str):
            # 单needle评估
            needle_lower = needle_or_needles.lower()
            # 提取关键词
            keywords = re.findall(r'\b\w+\b', needle_lower)
            keywords = [w for w in keywords if len(w) > 2]  # 过滤短词
            
            if not keywords:
                return 1.0 if needle_lower in response_lower else 0.0
            
            matched = sum(1 for keyword in keywords if keyword in response_lower)
            return matched / len(keywords)
        
        else:
            # 多needle评估
            scores = []
            for needle in needle_or_needles:
                score = self.evaluate_response(response, needle)
                scores.append(score)
            return sum(scores) / len(scores) if scores else 0.0


def call_model_with_moba(model_path, prompt, attn="flash_attention_2"):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    input_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_tokens], device=model.device)

    with torch.no_grad():
        tokens = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    response = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()
    
    return response.strip()


def needle_in_haystack_pipeline(model_path, test_type="single", attn="flash_attention_2"):
    """
    完整的Needle in Haystack测试pipeline
    
    Args:
        model_path: model path
        test_type: "single" 或 "multi"
        attn: attn_implementation
    """
    print(f"Begin {test_type} needle eval")
    
    tester = SimpleNeedleTester(model_path)
    context = tester.read_context_files()

    context_length = 4000
    depth_percent = 50
    
    if test_type == "single":
        needle = "The secret key is: BANANA_SPLIT_2024"
        question = "What is the secret key mentioned in the document?"
        
        context = tester.encode_and_trim(context, context_length)
        context_with_needle = tester.insert_single_needle(context, needle, depth_percent, context_length)
        
    else:
        needles = [
            "The first secret code is: ALPHA_7788",
            "The second secret code is: BETA_9900", 
            "The third secret code is: GAMMA_1122"
        ]
        question = "What are all the secret codes mentioned in the document?"
        
        context = tester.encode_and_trim(context, context_length)
        context_with_needle = tester.insert_multiple_needles(context, needles, depth_percent, context_length)
        needle = needles
    
    prompt = tester.generate_prompt(context_with_needle, question)
    print("Model Responding...")
    response = call_model_with_moba(model_path, prompt, attn)
    score = tester.evaluate_response(response, needle)
    
    print(f"响应: {response}")
    print(f"分数: {score:.2f}")
    
    return {
        'test_type': test_type,
        'needle': needle,
        'question': question,
        'response': response,
        'score': score,
        'context_length': len(tester.encode_text_to_tokens(context_with_needle))
    }
