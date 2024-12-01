"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse
import os 
def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu", "npu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per GPU for storing model weights. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument(
        "--gptq-ckpt",
        type=str,
        default=None,
        help="Used for GPTQ. The path to the local GPTQ checkpoint.",
    )
    parser.add_argument(
        "--gptq-wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="Used for GPTQ. #bits to use for quantization",
    )
    parser.add_argument(
        "--gptq-groupsize",
        type=int,
        default=-1,
        help="Used for GPTQ. Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--gptq-act-order",
        action="store_true",
        help="Used for GPTQ. Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--awq-ckpt",
        type=str,
        default=None,
        help="Used for AWQ. Load quantized model. The path to the local AWQ checkpoint.",
    )
    parser.add_argument(
        "--awq-wbits",
        type=int,
        default=16,
        choices=[4, 16],
        help="Used for AWQ. #bits to use for AWQ quantization",
    )
    parser.add_argument(
        "--awq-groupsize",
        type=int,
        default=-1,
        help="Used for AWQ. Groupsize to use for AWQ quantization; default uses full row.",
    )
    parser.add_argument(
        "--enable-exllama",
        action="store_true",
        help="Used for exllamabv2. Enable exllamaV2 inference framework.",
    )
    parser.add_argument(
        "--exllama-max-seq-len",
        type=int,
        default=4096,
        help="Used for exllamabv2. Max sequence length to use for exllamav2 framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--exllama-gpu-split",
        type=str,
        default=None,
        help="Used for exllamabv2. Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7",
    )
    parser.add_argument(
        "--exllama-cache-8bit",
        action="store_true",
        help="Used for exllamabv2. Use 8-bit cache to save VRAM.",
    )
    parser.add_argument(
        "--enable-xft",
        action="store_true",
        help="Used for xFasterTransformer Enable xFasterTransformer inference framework.",
    )
    parser.add_argument(
        "--xft-max-seq-len",
        type=int,
        default=4096,
        help="Used for xFasterTransformer. Max sequence length to use for xFasterTransformer framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--xft-dtype",
        type=str,
        choices=["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"],
        help="Override the default dtype. If not set, it will use bfloat16 for first token and float16 next tokens on CPU.",
        default=None,
    )
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--dataset_name",type=str,default='llcm')
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--message", type=str, default="Hello! Who are you?")
parser.add_argument("--input", type=str, default="None")
# parser.add_argument("--f", type=str)
add_model_args(parser)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from fastchat.model import load_model, get_conversation_template
if "t5" in args.model_path and args.repetition_penalty == 1.0:
    args.repetition_penalty = 1.2
# Load model
model, tokenizer = load_model(
        args.model_path,
        num_gpus=args.num_gpus,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

def get_augmented_description(description, model=model, tokenizer=tokenizer, prompt="rephrase the person's description above using similar words. Answer:"):

    # Build the prompt with a conversation template
    msg = description + prompt
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    with torch.no_grad():
        inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
        output_ids = model.generate(
            **inputs,
            do_sample=True if args.temperature > 1e-5 else False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
    return outputs

new_value = get_augmented_description(args.input)
