
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np

# Load pretrained feature extractor (ResNet50 without top)
def load_feature_extractor():
    model = models.resnet50(pretrained=True)
    modules = list(model.children())[:-1]  # remove last fc
    feat_extractor = nn.Sequential(*modules)
    for param in feat_extractor.parameters():
        param.requires_grad = False
    feat_extractor.eval()
    return feat_extractor

# Load classifier
def load_classifier(path, input_dim=2048, num_classes=4):
    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, num_classes)
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    model = Classifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocessing
def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Prediction pipeline
def predict(img, feat_extractor, classifier, class_map):
    inp = preprocess_image(img)
    with torch.no_grad():
        feats = feat_extractor(inp)
        feats = feats.view(feats.size(0), -1)
        logits = classifier(feats)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = np.argmax(probs)
    return class_map[pred], probs[pred]

# Streamlit app
st.title('Steel Microstructure Classifier')

uploaded = st.file_uploader('Upload a microstructure image', type=['png','jpg','jpeg','tif'])

if uploaded is not None:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated parameter

    # Lazy load models
    if 'feat_extractor' not in st.session_state:
        st.session_state.feat_extractor = load_feature_extractor()
        st.session_state.classifier = load_classifier('classifier.pth')
        st.session_state.class_map = {0: 'Pearlite', 1: 'Spheroidite', 2: 'Carbide Network', 3: 'Widmanstätten'}

    if st.button('Classify'):
        label, confidence = predict(image, st.session_state.feat_extractor, st.session_state.classifier, st.session_state.class_map)
        st.success(f'Prediction: {label} (confidence: {confidence*100:.1f}%)')
