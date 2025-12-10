import torch
from torchvision import transforms
from PIL import Image
from model import EmotionEfficientNetB4
import os
import torch.nn.functional as F

# ------------------------------- 
# PARAMETERS
# -------------------------------
model_path = "efficientnetb4_emotion.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7

emotion_map = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "anger"
}

# ------------------------------- 
# MODEL
# -------------------------------
model = EmotionEfficientNetB4(num_classes=num_classes).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ------------------------------- 
# TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------- 
# SINGLE IMAGE PREDICTION (top-2)
# -------------------------------
def predict_image(image_path):
    if not os.path.exists(image_path):
        print("⚠️ Image not found:", image_path)
        return None

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        top2_prob, top2_idx = torch.topk(probs, 2)
        top2_emotions = [(emotion_map[idx.item()], prob.item()) for idx, prob in zip(top2_idx[0], top2_prob[0])]

    print(f"Predicted top-2 emotions for {os.path.basename(image_path)}:")
    for em, p in top2_emotions:
        print(f"  {em}: {p*100:.2f}%")
    
    return top2_emotions

# ------------------------------- 
# BATCH PREDICTION FOR FOLDER
# -------------------------------
def predict_folder(folder_path):
    results = {}
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, img_name)
            top2 = predict_image(img_path)
            results[img_name] = top2
    return results

# ------------------------------- 
# test
# -------------------------------
test_image = "../Data/Test/disgust/d5.webp"
predict_image(test_image)

