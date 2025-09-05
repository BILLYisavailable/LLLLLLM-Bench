import os, json, re, argparse, time
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== 读取配置与模板 ======
model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_rag         = open('prompts/0shot_rag.txt',        encoding='utf-8').read()
template_no_context  = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot       = open('prompts/0shot.txt',            encoding='utf-8').read()
template_0shot_cot   = open('prompts/0shot_cot.txt',        encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt',  encoding='utf-8').read()

# ====== 小工具 ======
def middle_truncate_by_ids(tokenizer, text, max_len, disallowed_special=None):
    """把超长的prompt按 token 级别做中间截断（保留前后各一半）。"""
    input_ids = tokenizer.encode(text)
    

    if len(input_ids) > max_len:
        half = max_len // 2
        input_ids = input_ids[:half] + input_ids[-(max_len - half):]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return text

def build_prompt(item, args):
    """根据 flags 选择模板，并返回 (prompt, context_used)"""
    context = item['context']
    if args.rag > 0:
        tmpl = template_rag
        retrieved = item["retrieved_context"][:args.rag]
        retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
        context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
    elif args.no_context:
        tmpl = template_no_context
    elif args.cot:
        tmpl = template_0shot_cot
    else:
        tmpl = template_0shot

    prompt = (tmpl
              .replace('$DOC$', context.strip())
              .replace('$Q$', item['question'].strip())
              .replace('$C_A$', item['choice_A'].strip())
              .replace('$C_B$', item['choice_B'].strip())
              .replace('$C_C$', item['choice_C'].strip())
              .replace('$C_D$', item['choice_D'].strip()))
    return prompt, context

def extract_answer(response: str):
    response = response.replace('*', '')
    m = re.search(r'The correct answer is \(([A-D])\)', response)
    if m: return m.group(1)
    m = re.search(r'The correct answer is ([A-D])', response)
    if m: return m.group(1)
    return None

# ====== 主推理（transformers.generate） ======
def run_inference(items, args, fout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    assert args.model in model_map, f"model {args.model} 不在 model_map 中，请检查 config/model2path.json"
    model_path = model_map[args.model]
    max_len_prompt = maxlen_map[args.model]  # 针对 prompt 的最大长度限制

    # 加载 tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # decoder-only 常见设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype if device.type == "cuda" else None,
        attn_implementation="flash_attention_2"
    ).to(device).eval()

    # 生成参数
    gen_kwargs_base = dict(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.cot:
        gen_kwargs_cot = {**gen_kwargs_base, "max_new_tokens": 1024}
    else:
        gen_kwargs_cot = gen_kwargs_base

    batch_size = args.batch_size
    decode_generated_only = True  # 只解码新生成部分；如想包含 prompt，置 False

    pbar = tqdm(range(0, len(items), batch_size))
    for start in pbar:
        batch = items[start:start+batch_size]

        # 第一段：常规或 COT 的第一步
        prompts = []
        contexts_used = []
        for it in batch:
            prompt, ctx = build_prompt(it, args)
            # 按模型允许的最大 prompt 长度（token）做中间截断
            prompt = middle_truncate_by_ids(tokenizer, prompt, max_len_prompt,
                                            disallowed_special=())
            prompts.append(prompt)
            contexts_used.append(ctx[:1000])

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else torch.no_grad():
            out_ids = model.generate(**inputs, **(gen_kwargs_cot if args.cot else gen_kwargs_base))


        # 只取新生成部分（去掉 prompt）
        if decode_generated_only:
            gen_only = out_ids[:, inputs["input_ids"].size(1):]
            outputs = tokenizer.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            outputs = tokenizer.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # 如果是 COT，需要第二段：喂入 COT 答案模板再生成“最终选项”
        if args.cot:
            # 保存第一段的 COT 文本
            cot_texts = [o.strip() for o in outputs]

            # 组装第二段 prompts（cot_ans）
            prompts2 = []
            for it, ctx, cot_resp in zip(batch, contexts_used, cot_texts):
                p2 = (template_0shot_cot_ans
                      .replace('$DOC$', ctx.strip())
                      .replace('$Q$', it['question'].strip())
                      .replace('$C_A$', it['choice_A'].strip())
                      .replace('$C_B$', it['choice_B'].strip())
                      .replace('$C_C$', it['choice_C'].strip())
                      .replace('$C_D$', it['choice_D'].strip())
                      .replace('$COT$', cot_resp))
                p2 = middle_truncate_by_ids(tokenizer, p2, max_len_prompt, disallowed_special=())
                prompts2.append(p2)

            inputs2 = tokenizer(prompts2, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else torch.no_grad():
                out_ids2 = model.generate(**inputs2, **gen_kwargs_base)

            if decode_generated_only:
                gen_only2 = out_ids2[:, inputs2["input_ids"].size(1):]
                outputs2 = tokenizer.batch_decode(gen_only2, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                outputs2 = tokenizer.batch_decode(out_ids2, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # 写出结果
        for i, it in enumerate(batch):
            try:
                if args.cot:
                    response_cot = cot_texts[i]
                    response = outputs2[i].strip()
                else:
                    response_cot = None
                    response = outputs[i].strip()

                it_out = dict(it)  # 复制一份
                if response_cot is not None:
                    it_out['response_cot'] = response_cot
                it_out['response'] = response
                it_out['pred'] = extract_answer(response)
                it_out['judge'] = (it_out['pred'] == it_out['answer'])
                it_out['context'] = contexts_used[i]
                fout.write(json.dumps(it_out, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Write one item failed: {e}")
        fout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--cot", "-cot", default=False)
    parser.add_argument("--no_context", "-nc", default=False)
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    # 输出文件名
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{args.rag}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_topk_1024.jsonl")

    # 加载 LongBench-v2
    dataset = load_dataset('../datasets/LongBench-v2', split='train')
    data_all = [{
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "answer": item["answer"],
        "context": item["context"]
    } for item in dataset]

    # 断点续跑（跳过已有）
    has_ids = set()
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            for line in f:
                try:
                    has_ids.add(json.loads(line)["_id"])
                except:
                    pass

    data = [x for x in data_all if x["_id"] not in has_ids]
    if len(data) == 0:
        print("No new items to run. Done.")
        return

    with open(out_file, 'a', encoding='utf-8') as fout:
        run_inference(data, args, fout)

if __name__ == "__main__":
    main()

