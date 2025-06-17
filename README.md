# CompositionCap
To enhance the capabilities of vision-language models (VLMs) in Attribute-Aware Regional Captioning, Regional Dense Captioning, and Global Dense Captioning, we constructed a new human-annotated, high-quality dataset named COMPOSITIONCAP. The test set of COMPOSITIONCAP is sourced from the OpenImages dataset (image id provided in [filename2oiid.json](Data/filename2oiid.json)). We selected 1,000 images featuring diverse and complex scenes for annotation. The test set contains 7,215 masked entities, with 19,326 attribute-specific region captions. For more details, please visit our project page: https://hanghuacs.github.io/FineCaption/


## Mask Decoding
```python
def decompress_mask(comp_string, height, width):
    compressed_bytes = base64.b64decode(comp_string.encode('ascii'))
    decompressed_bytes = gzip.decompress(compressed_bytes)
    return np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape((height, width))
```

## Citation
```
@article{hua2024finecaption,
  title={FINECAPTION: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity},
  author={Hua, Hang and Liu, Qing and Zhang, Lingzhi and Shi, Jing and Kim, Soo Ye and Zhang, Zhifei and Wang, Yilin and Zhang, Jianming and Lin, Zhe and Luo, Jiebo},
  journal={CVPR}},
  year={2025}
}
```