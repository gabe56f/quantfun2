from pathlib import Path
from safetensors.torch import load_file
from src.models.lumina.image_two.pipeline import create_2b_model, LuminaDiT

luminaf = Path(
    "../lumina-image-2.0-bf16-diffusers/transformer/diffusion_pytorch_model.safetensors"
)

transformer = create_2b_model()
state_dict = load_file(luminaf)
state_dict = LuminaDiT.transform_state_dict(state_dict)

m, u = transformer.load_state_dict(state_dict, strict=False)
for k in sorted(u):
    print(k)
print("...")
for k in sorted(m):
    print(k)
if len(m) + len(u) == 0:
    print("load success")
