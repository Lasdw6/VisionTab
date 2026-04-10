import argparse
import base64
import io
import json
import time
import tempfile
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


def resolve_adapter_path(
    adapter_path: Path, extract_dir: Path | None = None
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return adapter folder path; auto-extract .zip (prefers persistent dir)."""
    if adapter_path.is_dir():
        return adapter_path, None

    if adapter_path.suffix.lower() != ".zip":
        raise ValueError(f"Adapter path must be a folder or .zip file, got: {adapter_path}")

    # Reuse an existing extracted folder if available.
    persistent_target = (extract_dir or adapter_path.parent) / adapter_path.stem
    if (persistent_target / "adapter_config.json").exists() and (
        persistent_target / "adapter_model.safetensors"
    ).exists():
        return persistent_target, None

    zip_size = adapter_path.stat().st_size
    target_parent = persistent_target.parent
    free_bytes = shutil.disk_usage(target_parent).free
    # Leave headroom because extracted content + cache usually needs > zip size.
    required = int(zip_size * 1.5)
    if free_bytes < required:
        raise RuntimeError(
            f"Not enough free disk space to extract adapter zip.\n"
            f"Zip size: {zip_size / (1024**3):.2f} GiB\n"
            f"Free at '{target_parent}': {free_bytes / (1024**3):.2f} GiB\n"
            f"Required (estimated): {required / (1024**3):.2f} GiB\n\n"
            "Fix options:\n"
            "1) Free disk space, OR\n"
            "2) Manually extract the zip and pass --adapter to that folder, OR\n"
            "3) Use --extract-dir on a drive with more space."
        )

    persistent_target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(adapter_path, "r") as zf:
        zf.extractall(persistent_target)
    return persistent_target, None


def detect_runtime() -> dict:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": None,
        "gpu_total_gb": None,
        "nvidia_smi": None,
    }

    if info["cuda_available"] and info["cuda_device_count"] > 0:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    else:
        # Probe NVIDIA driver presence even when torch is CPU-only.
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            info["nvidia_smi"] = out
        except Exception:
            pass

    return info


def build_model(base_model: str, adapter_dir: Path, load_in_4bit: bool, allow_cpu: bool):
    # Prefer adapter artifacts first (zip contains processor/tokenizer files).
    processor = None
    proc_errors = []
    for src in (adapter_dir, base_model):
        try:
            processor = AutoProcessor.from_pretrained(str(src), trust_remote_code=True)
            break
        except Exception as e:
            proc_errors.append(f"{src}: {type(e).__name__}: {e}")

    if processor is None:
        # Fallback tokenizer-only path with a clear multimodal compatibility error.
        try:
            tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        raise RuntimeError(
            "Could not load a multimodal processor. "
            "Your local transformers version likely does not support Gemma4 processor classes.\n"
            "Try upgrading:\n"
            '  pip install -U "transformers>=4.57.0"\n'
            "Processor load attempts:\n- "
            + "\n- ".join(proc_errors)
            + f"\nTokenizer is available (vocab size: {len(tok)}), but image+text inference requires AutoProcessor."
        )

    offload_dir = Path("offload_cache")
    offload_dir.mkdir(parents=True, exist_ok=True)

    def _base_kwargs():
        return {
            "device_map": "auto",
            "dtype": torch.float16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            # Reduce peak RAM/pagefile pressure while loading large checkpoints.
            "offload_state_dict": True,
            "offload_folder": str(offload_dir.resolve()),
        }

    def _is_windows_pagefile_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, OSError) and (
            "paging file is too small" in msg or "os error 1455" in msg
        )

    runtime = detect_runtime()
    print(
        f"Runtime: torch={runtime['torch_version']} | "
        f"cuda_available={runtime['cuda_available']} | "
        f"cuda_devices={runtime['cuda_device_count']}"
    )
    if runtime["gpu_name"]:
        print(f"GPU0: {runtime['gpu_name']} ({runtime['gpu_total_gb']} GiB)")
    elif runtime["nvidia_smi"]:
        print(f"nvidia-smi detected GPUs: {runtime['nvidia_smi']}")

    if not runtime["cuda_available"]:
        if not allow_cpu:
            raise RuntimeError(
                "CUDA is not available in this Python environment, but your model needs GPU/quantized loading.\n"
                "You appear to have an NVIDIA GPU, but torch is CPU-only.\n\n"
                "Recommended fix (Windows, CUDA 12.1 wheels):\n"
                '  pip uninstall -y torch torchvision torchaudio\n'
                '  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio\n\n'
                "Then rerun this script. If you still want CPU-only inference, pass --allow-cpu (very slow)."
            )
        print("WARNING: Running CPU-only inference (--allow-cpu). This will be very slow.")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": "cpu"},
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    # Strategy 1: 4-bit (fastest if it fits entirely on GPU)
    elif load_in_4bit and (runtime["gpu_total_gb"] is None or runtime["gpu_total_gb"] >= 8):
        try:
            model_kwargs = _base_kwargs()
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("Loading base model with 4-bit quantization...")
            base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        except ValueError as e:
            # Common on low-VRAM GPUs when 4-bit tries CPU/disk dispatch.
            if "Some modules are dispatched on the CPU or the disk" not in str(e):
                raise
            print("4-bit GPU load did not fit. Retrying with 8-bit + CPU offload...")
            model_kwargs = _base_kwargs()
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs["max_memory"] = {0: "5GiB", "cpu": "48GiB"}
            base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    elif load_in_4bit:
        # 6GB-class GPUs often fail pure 4-bit fitting with this model; go directly to 8-bit+offload.
        print("GPU VRAM is low; trying 8-bit GPU-only load first...")
        try:
            model_kwargs = _base_kwargs()
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = {"": 0}
            base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            print("Loaded 8-bit model on GPU.")
        except Exception as e:
            print(f"8-bit GPU-only load failed ({type(e).__name__}). Falling back to CPU offload.")
            model_kwargs = _base_kwargs()
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs["max_memory"] = {0: "5500MiB", "cpu": "48GiB"}
            try:
                base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            except Exception as e2:
                if _is_windows_pagefile_error(e2):
                    raise RuntimeError(
                        "Model load failed: Windows virtual memory (page file) is too small (OS error 1455).\n\n"
                        "Fix this in Windows:\n"
                        "1) Open System Properties -> Advanced -> Performance Settings -> Advanced -> Virtual memory.\n"
                        "2) Enable 'System managed size' OR set Custom size on your fastest drive.\n"
                        "3) Recommended minimum for this setup: 32-64 GB page file.\n"
                        "4) Reboot Windows, then start the server again.\n\n"
                        f"Temporary offload folder in use: {offload_dir.resolve()}"
                    ) from e2
                raise
    else:
        # Explicit non-4bit path
        model_kwargs = _base_kwargs()
        print("Loading base model without 4-bit quantization...")
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, processor


def build_prompt_text(processor, prompt: str) -> str:
    """Create a multimodal prompt with image token(s)."""
    prompt_text = None
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except ValueError:
            # Some processors expose the method but ship without a template.
            prompt_text = None

    if prompt_text is None:
        tok = getattr(processor, "tokenizer", None)
        image_token = "<|image|>"
        if tok is not None:
            image_token = (
                getattr(tok, "image_token", None)
                or tok.special_tokens_map.get("image_token")
                or image_token
            )
        prompt_text = f"{image_token}\n{prompt}"
    return prompt_text


def run_generation(
    model, processor, image: Image.Image, args, prompt: str, overrides: dict | None = None
) -> str:
    prompt_text = build_prompt_text(processor, prompt)

    cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_time": args.max_time,
        "max_image_size": args.max_image_size,
    }
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    max_image_size = int(cfg["max_image_size"])
    if max_image_size > 0:
        image = image.copy()
        image.thumbnail((max_image_size, max_image_size), Image.Resampling.BICUBIC)

    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    # With 8-bit + CPU offload, dispatched models often report `cpu` as primary.
    # Keep inputs on the model's reported primary device to avoid cpu/cuda mismatch.
    target_device = getattr(model, "device", torch.device("cpu"))
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "do_sample": bool(cfg["do_sample"]),
        "repetition_penalty": float(cfg["repetition_penalty"]),
        "no_repeat_ngram_size": int(cfg["no_repeat_ngram_size"]),
        "max_time": float(cfg["max_time"]),
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }
    if cfg["do_sample"]:
        gen_kwargs["temperature"] = float(cfg["temperature"])
        gen_kwargs["top_p"] = float(cfg["top_p"])

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = out[0][prompt_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=False)


def make_handler(state):
    class InferenceHandler(BaseHTTPRequestHandler):
        def _write_json(self, status_code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self._write_json(404, {"error": "not_found"})
                return
            self._write_json(
                200,
                {
                    "status": "ok",
                    "base_model": state["base_model"],
                    "adapter": state["adapter"],
                },
            )

        def do_POST(self):
            if self.path != "/generate":
                self._write_json(404, {"error": "not_found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._write_json(400, {"error": f"invalid_json: {e}"})
                return

            prompt = data.get("prompt")
            image_path = data.get("image_path")
            image_b64 = data.get("image_b64")
            if not prompt:
                self._write_json(400, {"error": "missing 'prompt'"})
                return
            if not image_path and not image_b64:
                self._write_json(400, {"error": "provide 'image_path' or 'image_b64'"})
                return

            try:
                if image_path:
                    image = Image.open(image_path).convert("RGB")
                else:
                    image_bytes = base64.b64decode(image_b64)
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                overrides = {
                    "max_new_tokens": data.get("max_new_tokens"),
                    "do_sample": data.get("do_sample"),
                    "temperature": data.get("temperature"),
                    "top_p": data.get("top_p"),
                    "repetition_penalty": data.get("repetition_penalty"),
                    "no_repeat_ngram_size": data.get("no_repeat_ngram_size"),
                    "max_time": data.get("max_time"),
                    "max_image_size": data.get("max_image_size"),
                }
                t0 = time.time()
                text = run_generation(
                    state["model"],
                    state["processor"],
                    image,
                    state["args"],
                    prompt,
                    overrides=overrides,
                )
                elapsed_ms = int((time.time() - t0) * 1000)
                self._write_json(200, {"text": text, "elapsed_ms": elapsed_ms})
            except Exception as e:
                self._write_json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            # Keep terminal noise low for inference requests.
            return

    return InferenceHandler


def run_server(model, processor, args, adapter_path: Path):
    state = {
        "model": model,
        "processor": processor,
        "args": args,
        "base_model": args.base_model,
        "adapter": str(adapter_path),
    }
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Server listening on http://{args.host}:{args.port}")
    print("POST /generate with JSON: {\"prompt\":\"...\", \"image_path\":\"...\"}")
    print("GET  /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run Gemma4 multimodal LoRA inference")
    parser.add_argument(
        "--adapter",
        type=str,
        default="gemma-4-e2b-multimodal-lora-final.zip",
        help="Path to adapter folder or adapter zip",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-4-E2B",
        help="Base HF model ID",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-time",
        type=float,
        default=90.0,
        help="Hard cap (seconds) for generation to avoid long hangs.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=448,
        help="Resize image so max(width, height) <= this value before processing.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (default: greedy decoding for stability).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (used only with --do-sample).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (used only with --do-sample).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Penalty >1.0 discourages repeated phrases.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=4,
        help="Prevent repeating n-grams of this size.",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU-only fallback if CUDA is unavailable (very slow).",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="Directory to extract adapter zip into (default: same folder as zip)",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as persistent HTTP server (loads model once).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if not args.server:
        if not args.image:
            parser.error("--image is required unless --server is used")
        if not args.prompt:
            parser.error("--prompt is required unless --server is used")

    extract_dir = Path(args.extract_dir) if args.extract_dir else None
    adapter_path, tmp_dir = resolve_adapter_path(Path(args.adapter), extract_dir=extract_dir)
    model, processor = build_model(
        args.base_model,
        adapter_path,
        load_in_4bit=not args.no_4bit,
        allow_cpu=args.allow_cpu,
    )

    if args.server:
        run_server(model, processor, args, adapter_path)
    else:
        image = Image.open(args.image).convert("RGB")
        text = run_generation(model, processor, image, args, args.prompt)
        print(text)

    if tmp_dir is not None:
        tmp_dir.cleanup()


if __name__ == "__main__":
    main()
