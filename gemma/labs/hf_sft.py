"""Minimal, readable Gemma 3 LoRA/QLoRA SFT with TRL and PEFT.

This is a GPU learning script, not a production trainer. Pin working dependency
versions and add experiment tracking before using it for a real project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--train-file", type=Path, default=HERE / "data/train.jsonl")
    parser.add_argument("--eval-file", type=Path, default=HERE / "data/eval.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hf-smoke"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
    messages = example["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"{example.get('id')}: final turn must be assistant")
    return {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
    }


def main() -> int:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "A CUDA GPU is required for this learning script. "
            "Run validate_dataset.py and evaluate_outputs.py on CPU."
        )
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")

    set_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float32
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "bf16_supported": bf16,
                "model": args.model_id,
                "qlora_4bit": args.load_in_4bit,
            },
            indent=2,
        )
    )

    data = load_dataset(
        "json",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.eval_file),
        },
    )
    train_columns = data["train"].column_names
    eval_columns = data["validation"].column_names
    train_dataset = data["train"].map(
        split_prompt_completion,
        remove_columns=train_columns,
        desc="Convert train messages to prompt/completion",
    )
    eval_dataset = data["validation"].map(
        split_prompt_completion,
        remove_columns=eval_columns,
        desc="Convert eval messages to prompt/completion",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    model_kwargs: dict[str, Any] = {
        "torch_dtype": compute_dtype,
    }
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs.update(
            quantization_config=quantization_config,
            device_map={"": torch.cuda.current_device()},
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_config = SFTConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        max_length=args.max_length,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=max(1, min(10, args.max_steps)),
        save_strategy="steps",
        save_steps=max(1, min(10, args.max_steps)),
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        bf16=bf16,
        fp16=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = {
        **result.metrics,
        **{f"final_{key}": value for key, value in eval_metrics.items()},
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "model_id": args.model_id,
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

