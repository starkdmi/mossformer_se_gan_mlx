# MossFormer GAN SE 16K

Speaker enhancement model for separating speech and background sounds into separate tracks using MLX.

## Usage

### Python

```bash
cd python
pip install -r requirements.txt
python generate.py --input audio.wav --model fp32
```

## Models

| Precision | Model Size  |
| --------- | ----------- |
| [FP32](https://huggingface.co/starkdmi/MossFormer_GAN_SE_16K_MLX/resolve/main/model_fp32.safetensors)      | **12.6 MB** |
| [FP16](https://huggingface.co/starkdmi/MossFormer_GAN_SE_16K_MLX/resolve/main/model_fp16.safetensors)      | **6.37 MB** |
| [INT8](https://huggingface.co/starkdmi/MossFormer_GAN_SE_16K_MLX/resolve/main/model_int8.safetensors)      | **5.96 MB** |
| [INT4](https://huggingface.co/starkdmi/MossFormer_GAN_SE_16K_MLX/resolve/main/model_int4.safetensors)      | **4.78 MB** |

HuggingFace: [starkdmi/MossFormer_GAN_SE_16K_MLX](https://huggingface.co/starkdmi/MossFormer_GAN_SE_16K_MLX)

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
