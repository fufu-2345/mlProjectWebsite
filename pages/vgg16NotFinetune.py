import streamlit as st
from PIL import Image
import numpy as np
from numpy import linalg as LA
import h5py
from scipy import spatial
import os
import base64

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

class VGG16:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.model = load_model('models/vgg16NotFinetuneModel.h5')
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract_feat(self, img_path):
        img =  image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

h5f = h5py.File("./vectorData/VGG16Vector.h5", 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
model = VGG16()
base_path = "train+vali/"
    
st.title("VGG16 with not Finetune")
st.write("")
st.write("Enter the picture to find 10 similar pictures")
uploaded_file = st.file_uploader("",type=None, label_visibility="collapsed")
if uploaded_file is not None:
    queryImg = Image.open(uploaded_file)
    st.image(queryImg, caption="Your picture", use_container_width=True)
    X = model.extract_feat(uploaded_file)
    scores = []
    for i in range(feats.shape[0]):
        score = 1 - spatial.distance.cosine(X, feats[i])
        scores.append(score)
    scores = np.array(scores)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    imlist = [imgNames[index].decode('utf-8') for index in rank_ID[0:10]]
    cols = st.columns(5)
    for i, img_name in enumerate(imlist):
        img_path = os.path.join(base_path, img_name)
        with open(img_path, "rb") as f:
            data = f.read()
        img_base64 = base64.b64encode(data).decode("utf-8")
        with cols[i % 5]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <img src="data:image/jpeg;base64,{img_base64}" 
                        style="width:150px; height:100px; object-fit:cover; border-radius:10px;">
                    <p style="font-size:14px; color:gray;">Similar no. {i + 1}</p>
                </div>
                """,
                unsafe_allow_html=True
            )