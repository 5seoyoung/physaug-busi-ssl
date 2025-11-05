import torch, cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 필요한 경우 설치: pip install pytorch-grad-cam
# 모델/체크포인트 경로와 target_layer는 네트에 맞게 조정
CKPT = "outputs/runs/cv/physaug_r1.0_f0/best_cls.pt"
OUT  = "outputs/figs/cam"
IMGS = [
  "data/raw/BUSI/Dataset_BUSI_with_GT/benign/xxx.png",
  "data/raw/BUSI/Dataset_BUSI_with_GT/malignant/yyy.png",
  "data/raw/BUSI/Dataset_BUSI_with_GT/normal/zzz.png",
]
img_size = 128

os = __import__("os")
os.makedirs(OUT, exist_ok=True)

# 로더
from src.downstream.cls_train import build_model  # 네트 생성 함수(있다고 가정)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(n_classes=3, freeze_backbone=False).to(device)
state = torch.load(CKPT, map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)
model.eval()

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# target layer: mobilenetv3 마지막 conv
target_layer = [dict(model.named_modules())["backbone.features.15"]]

cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)

for p in IMGS:
    rgb = Image.open(p).convert("RGB")
    tensor = tfm(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        cls = int(torch.argmax(logits, dim=1).item())

    grayscale_cam = cam(input_tensor=tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    img = np.array(rgb.resize((img_size, img_size))) / 255.0
    vis = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    outp = Path(OUT) / (Path(p).stem + f"_cam_cls{cls}.png")
    cv2.imwrite(str(outp), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("[CAM]", outp)
