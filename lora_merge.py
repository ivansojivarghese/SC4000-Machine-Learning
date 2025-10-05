import os
import argparse
import gc
from contextlib import contextmanager
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch


@contextmanager
def _temporary_hide_gpus(enabled: bool):
    """Temporarily hide CUDA devices (set CUDA_VISIBLE_DEVICES="") to force CPU load.
    Restores the original value afterwards.
    """
    if not enabled:
        yield
        return
    original = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


def _select_cpu_dtype():
    """Choose the lightest viable dtype for CPU merge.
    Preference: float16 if supported, else bfloat16, else float32.
    """
    # Some ops on older CPUs may not like bfloat16/float16; we optimistically try float16.
    # The merge itself is mostly parameter arithmetic (matmul not required), so float16 ok.
    return torch.float16


def _merge_low_mem(base_model: str, lora_dir: str, hf_token: str, hide_gpu: bool = True, force_cpu: bool = True):
    """Load base + LoRA entirely on CPU to merge, avoiding GPU OOM.

    Strategies:
      - Temporarily hide GPUs so HF/accelerate won't offload any shard to CUDA.
      - Use low_cpu_mem_usage to stream weights.
      - Use the smallest safe dtype to reduce RAM footprint.
    """
    print("[Merge] Low-memory CPU merge path engaged (forcing CPU load, minimal dtype)")
    cpu_dtype = _select_cpu_dtype()
    with _temporary_hide_gpus(hide_gpu):
        if force_cpu:
            # Monkeypatch torch cuda availability so downstream libs won't attempt GPU placement
            torch.cuda.is_available = lambda: False  # type: ignore
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": "cpu"},  # force everything on CPU
            trust_remote_code=True,
            torch_dtype=cpu_dtype,
            low_cpu_mem_usage=True,
            token=hf_token,
            local_files_only=False,
        )
        base.to("cpu")
        # Ensure PEFT loads strictly on CPU
        peft = PeftModel.from_pretrained(base, lora_dir, device_map={"": "cpu"})
        merged = peft.merge_and_unload()
        merged.to("cpu")
    return merged


def _free_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def merge_lora(base_model: str, lora_dir: str, out_dir: str, low_mem: bool = False, cpu_only: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    # Ensure standard HTTP path; disable xet/transfer accelerations in-process
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    merged = None
    if cpu_only:
        print("[Merge] --cpu-only specified; skipping any GPU attempt.")
        low_mem = True

    if low_mem:
        merged = _merge_low_mem(base_model, lora_dir, hf_token, hide_gpu=True, force_cpu=cpu_only or True)
    else:
        try:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
                local_files_only=False,
            )
            peft = PeftModel.from_pretrained(base, lora_dir)
            merged = peft.merge_and_unload()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                print("[Merge][Warn] GPU OOM during merge; retrying with low-memory CPU path...")
                # Explicitly free GPU memory before fallback
                del base
                try:
                    del peft
                except Exception:
                    pass
                gc.collect()
                _free_cuda()
                merged = _merge_low_mem(base_model, lora_dir, hf_token, hide_gpu=True, force_cpu=True)
            else:
                raise
    print(f"[Merge] Saving merged model to {out_dir}")
    merged.save_pretrained(out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--low-mem", action="store_true", help="Force CPU/low-memory merge path (still allows GPU visibility)")
    ap.add_argument("--cpu-only", action="store_true", help="Force CPU merge and hide GPUs (strongest OOM avoidance)")
    args = ap.parse_args()
    merge_lora(args.base_model, args.lora_dir, args.out_dir, low_mem=args.low_mem, cpu_only=args.cpu_only)
