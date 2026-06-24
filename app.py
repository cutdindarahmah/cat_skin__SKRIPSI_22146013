import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
except Exception:
    try:
        import keras
        from keras.models import load_model
        from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
        from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
    except Exception:
        load_model = None
        mobilenet_preprocess_input = None
        efficientnet_preprocess_input = None

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
if not MODEL_DIR.exists():
    MODEL_DIR = SCRIPT_DIR / "main" / "model"

def load_models():
    if load_model is None:
        return None, None

    mobilenet_path = MODEL_DIR / "mobilenetv2_cat_skin_disease.h5"
    efficientnet_path = MODEL_DIR / "efficientnetb1_cat_skin_disease_final.keras"

    if not mobilenet_path.exists() or not efficientnet_path.exists():
        return None, None

    try:
        mobilenet = load_model(mobilenet_path)
        efficientnet = load_model(efficientnet_path)
        return mobilenet, efficientnet
    except Exception:
        return None, None

mobilenet_model, efficientnet_model = load_models()

if mobilenet_model is None or efficientnet_model is None:
    st.warning("Model tidak dapat dimuat di environment ini. Aplikasi tetap berjalan, tetapi prediksi tidak tersedia sampai dependency TensorFlow compatible terpasang.")

# ==========================
# CLASS NAMES
# SESUAIKAN URUTAN INI DENGAN URUTAN OUTPUT MODEL SAAT TRAINING
# ==========================
DEFAULT_CLASS_NAMES = [
    "Scabies",
    "Flea_Allergy",
    "Health",
    "Ringworm"
]

st.sidebar.markdown("---")
class_names_input = st.sidebar.text_input(
    "Urutan label kelas model",
    ",".join(DEFAULT_CLASS_NAMES),
    help="Pisahkan dengan koma sesuai urutan output model. Contoh: Scabies,Flea_Allergy,Health,Ringworm"
)

class_names = [
    name.strip() for name in class_names_input.split(",") if name.strip()
]

if len(class_names) != 4:
    st.sidebar.warning(
        "Harap isi 4 label yang dipisahkan koma agar sesuai dengan output model."
    )
    class_names = DEFAULT_CLASS_NAMES

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
    # PILIH MODEL
    # ==========================
    if selected_model == "MobileNetV2":
        model = mobilenet_model
        preprocess_func = mobilenet_preprocess_input
    else:
        model = efficientnet_model
        preprocess_func = efficientnet_preprocess_input

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    target_size = 224
    if len(input_shape) >= 3:
        for dim in input_shape[1:3]:
            if isinstance(dim, int) and dim > 0:
                target_size = dim
                break

    # ==========================
    # PREPROCESSING
    # ==========================
    img = image.resize((target_size, target_size))

    img_array = np.array(img, dtype="float32")
    img_array = preprocess_func(img_array)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

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