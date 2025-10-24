import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from numpy import linalg as LA
import h5py
from scipy import spatial
import os
import base64
import pickle
from annoy import AnnoyIndex
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

df = pd.read_csv("vectorData/trainValiVGG16Vectors.csv")
filenames = df["filename"].values
vectors = df.drop(columns=["filename"]).values
vector_dim = df.drop(columns=["filename"]).shape[1]
ann_index = AnnoyIndex(vector_dim, metric='euclidean')

modelPath = Path(__file__).parent.parent/"models"
pathh = Path(__file__).parent.parent/"train+vali/"

CNN = load_model(modelPath/"cnnVGG16Finetune.h5")
vgg = load_model(modelPath/"vgg16NotFinetuneModel.h5")
ann_index.load("models/annoyVGG16.ann")

import joblib
stdScaler = joblib.load('scaler/vgg16/stdScaler.pkl')
mmScaler = joblib.load('scaler/vgg16/minMaxScaler.pkl')

def extract_feature(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = vgg.predict(x, verbose=0)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    features = stdScaler.transform(features)
    features = CNN.predict(features, verbose=0)
    features = mmScaler.transform(features)
    return features.flatten()

def find_similar_images(img_file, top_k=10):
    query_vec = extract_feature(img_file)
    idxs, dists = ann_index.get_nns_by_vector(query_vec, top_k, include_distances=True)
    results = [(filenames[i], d) for i, d in zip(idxs, dists)]
    return results

st.title("VGG16 with Finetune")
st.write("")
st.write("Enter the picture to find 10 similar pictures")
uploaded_file = st.file_uploader("",type=None, label_visibility="collapsed")
if uploaded_file is not None:
    query_img = Image.open(uploaded_file)
    st.image(query_img, caption="Your picture", use_container_width=True)
    similar = find_similar_images(uploaded_file, top_k=10)
    st.subheader("Top 10 similar images")
    cols = st.columns(5)
    for i, (fname, dist) in enumerate(similar):
        img_path = os.path.join(pathh, fname)
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode("utf-8")
            with cols[i % 5]:
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <img src="data:image/jpeg;base64,{img_base64}" 
                             style="width:150px; height:100px; object-fit:cover; border-radius:10px;">
                        <p style="font-size:13px; color:gray;">Similar no. {i+1}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
# <p style="font-size:13px; color:gray;">path: {fname}</p>