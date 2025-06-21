import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------
# Generator Definition
# -------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(  # <- must match training script
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, self.label_embed(labels)], dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)


# -------------------------
# Load Trained Generator
# -------------------------
@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("Project/models/mnist_gan_generator.pth", map_location='cpu'))
    model.eval()
    return model

generator = load_generator()

# -------------------------
# Streamlit App UI
# -------------------------
st.title("🧠 Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like digits using a GAN trained **from scratch**.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

if st.button("Generate Images"):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)

    with torch.no_grad():
        generated = generator(z, labels).cpu()
        generated = generated * 0.5 + 0.5  # scale [-1,1] to [0,1]

    st.subheader(f"Generated images of digit **{digit}**")
    cols = st.columns(5)
    for i in range(5):
        img = generated[i].squeeze().numpy()
        cols[i].image(img, width=100)
        cols[i].caption(f"Sample {i+1}")
