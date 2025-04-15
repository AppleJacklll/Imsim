import os
import pickle
from pathlib import Path
import pypdfium2 as pdfium
import torch
import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image, ImageDraw

DRAWINGS_PATH = os.environ.get('DRAWINGS_PATH', "drawings")
EMBEDDING_PATH = r'internal/embedding/embeddings.pkl'
MODEL = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', trust_repo="check")
ERASE_TABLE_MODEL = YOLO('models/yolov8_custom_chart.pt')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"# "xpu" if torch.xpu.is_available() else "cpu"
MODEL.to(DEVICE)

ENXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF', '.tiff', '.TIFF', '.pdf', '.PDF')


def load_image(filename: str) -> torch.Tensor:
    if not filename.lower().endswith(ENXTENSIONS):
        print("Unsupported file format.")
        return None
    
    image_path = os.path.join(DRAWINGS_PATH, filename)
    
    if filename.endswith('.pdf'):
        try:
            pages = pdfium.PdfDocument(image_path)
            if not pages or len(pages) == 0:
                print("No pages found in the provided PDF file.")
                return None
            image = pages[0].render(scale=4).to_pil().convert("RGB")
        except Exception as e:
            print("Error converting PDF to image", e)
            return None
    else:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            
    erase_result = ERASE_TABLE_MODEL.predict(source=image, show=False, save=False)[0]
            
    draw = ImageDraw.Draw(image)
    
    for box in erase_result.boxes:
        if box.conf[0] > 0.85:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

    image_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return image_transforms(image).unsqueeze(0)

def extract_features(filename: str) -> torch.Tensor:
    image_tensor = load_image(filename)
    if image_tensor is None:
        return None

    with torch.no_grad():
        tokens = MODEL.forward_features(image_tensor)["x_norm_patchtokens"]
        
    return tokens.mean(dim=1).squeeze(0).cpu().numpy()

def create_embeddings():
    Path(os.path.dirname(EMBEDDING_PATH)).mkdir(parents=True, exist_ok=True)
    with open(EMBEDDING_PATH, mode='wb') as embedding:
        pickle.dump({}, embedding)

def load_embeddings() -> dict:
    if not os.path.exists(EMBEDDING_PATH):
        create_embeddings()
    with open(EMBEDDING_PATH, mode='rb') as embedding:
        embedding_dict = pickle.load(embedding)
    return embedding_dict

def update_embeddings(candidate_embeddings: dict):
    if not os.path.exists(EMBEDDING_PATH):
        create_embeddings()
    with open(EMBEDDING_PATH, mode='wb') as embedding:
        pickle.dump(candidate_embeddings, embedding)
