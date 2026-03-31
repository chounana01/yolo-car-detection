import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

# ------------------- HEADER / TITRE -------------------
st.set_page_config(page_title="Détection de voitures", page_icon="🚗", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚗 Détection de Voitures avec YOLOv8</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- INFOS / INTRO -------------------
with st.expander("ℹ️ À propos de cette application", expanded=True):
    st.write("""
        Cette interface permet de détecter automatiquement les **voitures** dans une image en utilisant un modèle **YOLOv8 préentraîné**.
        - Téléversez une image (JPG, PNG, JPEG)
        - Obtenez une image annotée et les données de détection
        - Téléchargez le résultat 🔽
    """)

# ------------------- CHARGEMENT DU MODÈLE -------------------
model = YOLO("yolov8n.pt") # À NE PAS inclure dans l'archive .zip

# ------------------- UPLOADER D'IMAGE -------------------
uploaded_file = st.file_uploader("📁 Téléversez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Image téléversée", use_container_width=True)

    # Traitement
    image_array = np.array(image)
    results = model(image_array)

    # Vérifier s'il y a des voitures détectées
    car_class_id = None
    for cls_id, name in model.names.items():
        if name == "car":
            car_class_id = cls_id
            break
    car_boxes = [box for box in results[0].boxes.cls if int(box) == car_class_id]

    # Annoter image
    annotated = results[0].plot()
    st.image(annotated, caption="📍 Résultat : Objets détectés", use_container_width=True)

    # Télécharger le résultat
    buf = BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Télécharger l’image annotée",
        data=byte_im,
        file_name="resultat_voiture.png",
        mime="image/png"
    )

    # Avertissement si aucune voiture
    if len(car_boxes) == 0:
        st.warning("🚫 Aucune voiture détectée dans l'image.")

    # ------------------- 📊 Graphique camembert -------------------
    all_classes = [int(cls) for cls in results[0].boxes.cls.cpu().numpy()]
    class_names = [model.names[i] for i in all_classes]
    unique_classes = list(set(class_names))
    counts = [class_names.count(c) for c in unique_classes]

    if unique_classes:
        st.markdown("### 📈 Répartition des objets détectés")
        fig, ax = plt.subplots()
        ax.pie(counts, labels=unique_classes, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    # ------------------- Données brutes -------------------
    st.markdown("### 📊 Données de détection")
    st.dataframe(results[0].boxes.data.cpu().numpy())

else:
    st.info("Veuillez téléverser une image pour commencer l'analyse.")

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Projet SDD1004 • Réalisé avec ❤️ par cheikhouna kebe</p>", unsafe_allow_html=True)
