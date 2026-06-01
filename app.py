import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Deteksi Penyakit Kulit Kucing",
    page_icon="🐱",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model"

@st.cache_resource
def load_models():
    mobilenet_path = MODEL_DIR / "mobilenetv2_cat_skin_disease.h5"
    efficientnet_path = MODEL_DIR / "efficientnetb1_cat_skin_disease_final.keras"

    if not mobilenet_path.exists() or not efficientnet_path.exists():
        missing = [
            str(p) for p in (mobilenet_path, efficientnet_path) if not p.exists()
        ]
        raise FileNotFoundError(
            f"Model file(s) not found: {', '.join(missing)}. "
            f"Letakkan model di folder {MODEL_DIR}."
        )

    mobilenet = load_model(mobilenet_path)
    efficientnet = load_model(efficientnet_path)

    return mobilenet, efficientnet

mobilenet_model, efficientnet_model = load_models()

# ==========================
# CLASS NAMES
# GANTI SESUAI DATASET
# ==========================
class_names = [
    "Flea_Allergy",
    "Health",
    "Ringworm",
    "Scabies"
]

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("Pengaturan")

selected_model = st.sidebar.selectbox(
    "Pilih Model",
    [
        "MobileNetV2",
        "EfficientNetB1"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Sistem Deteksi Penyakit Kulit Kucing
    
    Model:
    - MobileNetV2
    - EfficientNetB1
    """
)

# ==========================
# TITLE
# ==========================
st.title("🐱 Deteksi Penyakit Kulit Kucing")

st.write(
    "Upload gambar kulit kucing untuk mendeteksi jenis penyakit menggunakan model CNN."
)

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Gambar yang Diunggah",
            use_container_width=True
        )

        st.success("Gambar berhasil diunggah")

    # ==========================
    # PREPROCESSING
    # ==========================
    img = image.resize((224,224))

    img_array = np.array(img)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ==========================
    # PILIH MODEL
    # ==========================
    if selected_model == "MobileNetV2":

        model = mobilenet_model

    else:

        model = efficientnet_model

    # ==========================
    # PREDIKSI
    # ==========================
    prediction = model.predict(img_array)

    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100

    disease = class_names[
        predicted_class
    ]

    # ==========================
    # HASIL
    # ==========================
    with col2:

        st.subheader("Hasil Prediksi")

        st.success(
            f"Prediksi: {disease}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.write(
            f"Model: {selected_model}"
        )

    # ==========================
    # DETAIL PROBABILITAS
    # ==========================
    st.markdown("---")

    st.subheader(
        "Probabilitas Semua Kelas"
    )

    prob = np.asarray(prediction).squeeze() * 100
    if prob.ndim != 1:
        prob = prob.flatten()

    labels = class_names
    if len(labels) != len(prob):
        st.warning(
            f"Jumlah label ({len(labels)}) tidak cocok dengan jumlah output model ({len(prob)}). "
            "Menggunakan label numerik untuk semua kelas."
        )
        labels = [
            class_names[i] if i < len(class_names) else f"Kelas {i + 1}"
            for i in range(len(prob))
        ]

    df = pd.DataFrame(
        {
            "Penyakit": labels,
            "Probabilitas (%)": prob
        }
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.bar_chart(
        df.set_index(
            "Penyakit"
        )
    )

    # ==========================
    # TOP 3 PREDIKSI
    # ==========================
    st.subheader(
        "Top 3 Prediksi"
    )

    top3_idx = np.argsort(
        prob
    )[-3:][::-1]

    for i in top3_idx:

        st.write(
            f"{labels[i]} : {prob[i]:.2f}%"
        )

    # ==========================
    # HISTORY
    # ==========================
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append(
        {
            "Model": selected_model,
            "Prediksi": disease,
            "Confidence": round(
                confidence,
                2
            )
        }
    )

    st.markdown("---")

    st.subheader(
        "Riwayat Prediksi"
    )

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )