import streamlit as st
import numpy as np
from PIL import Image
import os
import torch
from torchvision import transforms
import cv2
from classification import StrokeClassifier, StrokeTypeClassifier, predict_class
from segmentation_detection import UNet, extract_clots_from_mask

st.set_page_config(page_title="Brain CT Stroke & Clot Detection", page_icon="🧠", layout="wide")

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Stroke Classification
    model_stroke = StrokeClassifier(num_classes=2).to(device)
    if os.path.exists("stroke_classifier_weights.pth"):
        model_stroke.load_state_dict(torch.load("stroke_classifier_weights.pth", map_location=device))
    else:
        st.error("stroke_classifier_weights.pth NOT FOUND")
    model_stroke.eval()
    
    # Stroke Type
    model_type = StrokeTypeClassifier(num_classes=2).to(device)
    if os.path.exists("stroke_type_weights.pth"):
        model_type.load_state_dict(torch.load("stroke_type_weights.pth", map_location=device))
    else:
        st.error("stroke_type_weights.pth NOT FOUND")
    model_type.eval()

    # UNet
    model_unet = UNet(n_channels=3, n_classes=1).to(device)
    if os.path.exists("unet_weights.pth"):
        model_unet.load_state_dict(torch.load("unet_weights.pth", map_location=device))
    else:
        st.error("unet_weights.pth NOT FOUND")
    model_unet.eval()
    
    return model_stroke, model_type, model_unet, device

model_stroke, model_type, model_unet, device = load_models()

# -------------------- TRANSFORMS --------------------
classification_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

segmentation_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# -------------------- PREDICTION --------------------
def predict_stroke(image):
    img = classification_transforms(image).unsqueeze(0).to(device)
    return predict_class(model_stroke, img, ["Normal", "Stroke"])

def predict_stroke_type(image):
    img = classification_transforms(image).unsqueeze(0).to(device)
    return predict_class(model_type, img, ["Hemorrhagic", "Ischemic"])

def detect_clots(image):
    img = segmentation_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model_unet(img).squeeze().cpu().numpy()
    
    num_clots, _, contours, _ = extract_clots_from_mask(mask)
    
    img_array = np.array(image)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_array, (x,y), (x+w,y+h), (255,0,0), 2)
    
    return num_clots, Image.fromarray(img_array)

# -------------------- APP UI --------------------
def main():
    st.title("🧠 Brain CT Stroke Detection")

    uploaded_file = st.file_uploader("Upload CT Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

        stroke = predict_stroke(image)
        st.write("Stroke:", stroke)

        if stroke == "Stroke":
            stroke_type = predict_stroke_type(image)
            st.write("Type:", stroke_type)

        num_clots, output_img = detect_clots(image)
        st.write("Clots:", num_clots)

        st.image(output_img, caption="Detection Result", use_container_width=True)

if __name__ == "__main__":
    main()
