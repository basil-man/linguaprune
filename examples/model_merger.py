import os
import shutil
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', required=True, type=str, help="Path to saved model (checkpoint dir)")
    parser.add_argument('--save_dir', required=True, type=str, help="Path to save HF-compatible model")
    args = parser.parse_args()

    # 加载权重
    model_path = os.path.join(args.local_dir, "model_world_size_1_rank_0.pt")
    state_dict = torch.load(model_path, map_location="cpu")

    # 加载 config
    hf_path = os.path.join(args.local_dir, "huggingface")
    config = AutoConfig.from_pretrained(hf_path)

    # 自动选择模型类型
    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    # 构造空模型结构
    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    # 保存 Huggingface 格式模型
    print(f"Saving model to {args.save_dir}")
    model.save_pretrained(args.save_dir, state_dict=state_dict)

    # 拷贝 tokenizer 文件
    tokenizer_files = [
        "added_tokens.json", "special_tokens_map.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt"
    ]
    for t_file in tokenizer_files:
        src_path = os.path.join(args.local_dir, t_file)
        if os.path.exists(src_path):
            shutil.copy(src=src_path, dst=args.save_dir)

    print("✅ Model successfully converted and saved to:", args.save_dir)
