# CompositionCap

```python
def decompress_mask(comp_string, height, width):
    compressed_bytes = base64.b64decode(comp_string.encode('ascii'))
    decompressed_bytes = gzip.decompress(compressed_bytes)
    return np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape((height, width))
```
