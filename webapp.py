import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# --------------------------
# Load model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, 1)
    model.load_state_dict(torch.load("vit_radam_best.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# --------------------------
# Image transform (same as training)
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------
# Streamlit UI
# --------------------------
st.title("🩺 Pneumonia Detection")
st.write("Upload a *Chest X-Ray image* to predict whether it is Normal or Pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output).item()

    # Threshold 0.5 → Pneumonia
    if pred > 0.5:
        st.error("⚠ Prediction: *PNEUMONIA detected*")
    else:
        st.success("✅ Prediction: *NORMAL*")
