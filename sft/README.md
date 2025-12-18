Training job is run in NVIDIA A100 in Lambda w CUDA12.8 which takes around 12hrs in total (6+6 for pretraining and sft sperarately). Below is the requirements:
```
conda create -n sft python=3.8 -y

conda activate sft

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install tensorboard transformers tokenizers
```