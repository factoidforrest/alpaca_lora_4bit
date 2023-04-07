# Alpaca Lora 4bit
Made some adjust for the code in peft and gptq for llama, and make it possible for lora finetuning with a 4 bits base model. The same adjustment can be made for 2, 3 and 8 bits.

* Install Manual by s4rduk4r: https://github.com/s4rduk4r/alpaca_lora_4bit_readme/blob/main/README.md (**NOTE:** don't use the install script, use the requirements.txt instead.)
* Also Remember to create a venv if you do not want the packages be overwritten.

# Update Logs
* Resolved numerically unstable issue
* Reconstruct fp16 matrix from 4bit data and call torch.matmul largely increased the inference speed.
* Added install script for windows and linux.
* Added Gradient Checkpointing. Now It can finetune 30b model 4bit on a single GPU with 24G VRAM with Gradient Checkpointing enabled. (finetune.py updated) (but would reduce training speed, so if having enough VRAM this option is not needed)
* Added install manual by s4rduk4r
* Added pip install support by sterlind, preparing to merge changes upstream
* Added V2 model support (with groupsize, both inference + finetune)
* Added some options on finetune: set default to use eos_token instead of padding, add resume_checkpoint to continue training
* Added offload support. load_llama_model_4bit_low_ram_and_offload_to_cpu function can be used.
* Added monkey patch for text generation webui for fixing initial eos token issue.
* Added Flash attention support. (Use --flash-attention)
* Added Triton backend to support model using groupsize and act-order. (Use --backend=triton)

# Requirements
gptq-for-llama <br>
peft<br>
The specific version is inside requirements.txt<br>

# Install
~copy files from GPTQ-for-LLaMa into GPTQ-for-LLaMa path and re-compile cuda extension~<br>
~copy files from peft/tuners/lora.py to peft path, replace it~<br>

**NOTE:** Install scripts are no longer needed! requirements.txt now pulls from forks with the necessary patches.

```
pip install -r requirements.txt
```

# Finetune
~The same finetune script from https://github.com/tloen/alpaca-lora can be used.~<br>

After installation, this script can be used:
GPTQv1:

```
python finetune.py
```
or
```
GPTQ_VERSION=1 python finetune.py
```

GPTQv2:
```
GPTQ_VERSION=2 python finetune.py
```

# Inference

After installation, this script can be used:

```
python inference.py
```

# Text Generation Webui Monkey Patch

Clone the latest version of text generation webui and copy all the files into ./text-generation-webui/
```
git clone https://github.com/oobabooga/text-generation-webui.git
```

Open server.py and insert a line at the beginning
```
import custom_monkey_patch # apply monkey patch
import gc
import io
...
```

Use the command to run

```
python server.py
```

# Flash Attention

It seems that we can apply a monkey patch for llama model. To use it, simply download the file from [MonkeyPatch](https://github.com/lm-sys/FastChat/blob/daa9c11080ceced2bd52c3e0027e4f64b1512683/fastchat/train/llama_flash_attn_monkey_patch.py). And also, flash-attention is needed, and currently do not support pytorch 2.0.
Just add --flash-attention to use it for finetuning.
