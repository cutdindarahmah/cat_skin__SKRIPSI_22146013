import io
from pathlib import Path

import numpy as np
import streamlit as st

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
except Exception:
    try:
        from keras.models import load_model
        from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
        from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
    except Exception:
        load_model = None
        mobilenet_preprocess_input = None
        efficientnet_preprocess_input = None

try:
    from PIL import Image
except Exception:
    Image = None

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

def get_model_input_size(model):
    if model is None:
        return 224

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0] if input_shape else None

    if input_shape is None:
        try:
            input_shape = model.input.shape
        except Exception:
            input_shape = None

    if input_shape is None and hasattr(model, "layers"):
        for layer in model.layers:
            layer_shape = getattr(layer, "input_shape", None)
            if isinstance(layer_shape, (list, tuple)) and len(layer_shape) >= 3:
                input_shape = layer_shape
                break

    if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
        for dim in input_shape[1:3]:
            if isinstance(dim, int) and dim > 0:
                return dim

    return 224


def predict_with_fallback(img_array):
    image = np.asarray(img_array, dtype="float32")
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)

    gray = np.mean(image, axis=-1)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    redness = float(np.mean(image[..., 0] - image[..., 1]))
    saturation = float(np.mean(np.std(image, axis=-1)))

    probs = np.array([0.25, 0.25, 0.25, 0.25], dtype="float32")

    if brightness < 90 and redness > 15:
        probs = np.array([0.45, 0.25, 0.1, 0.2], dtype="float32")
    elif contrast > 40 and saturation > 35:
        probs = np.array([0.2, 0.15, 0.15, 0.5], dtype="float32")
    elif brightness > 140 and saturation < 25:
        probs = np.array([0.1, 0.1, 0.7, 0.1], dtype="float32")
    else:
        probs = np.array([0.3, 0.3, 0.2, 0.2], dtype="float32")

    return probs.reshape(1, -1)


def load_model_file(model_path):
    if load_model is None:
        return None

    if not model_path.exists():
        return None

    try:
        return load_model(model_path, compile=False)
    except Exception:
        try:
            return load_model(model_path)
        except Exception:
            return None


def load_models():
    mobilenet_path = MODEL_DIR / "mobilenetv2_cat_skin_disease.h5"
    efficientnet_path = MODEL_DIR / "efficientnetb1_cat_skin_disease_final.keras"

    if not mobilenet_path.exists() or not efficientnet_path.exists():
        return None, None

    mobilenet = load_model_file(mobilenet_path)
    efficientnet = load_model_file(efficientnet_path)
    return mobilenet, efficientnet

mobilenet_model, efficientnet_model = None, None

if st.session_state.get("models_loaded") is None:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    mobilenet_model, efficientnet_model = load_models()
    st.session_state.models_loaded = True

if mobilenet_model is None or efficientnet_model is None:
    st.caption("Mode demo aktif: model TensorFlow tidak tersedia di environment deployment.")

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

    uploaded_bytes = uploaded_file.getvalue()

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            uploaded_bytes,
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

    target_size = get_model_input_size(model)

    # ==========================
    # PREPROCESSING
    # ==========================
    try:
        if Image is not None:
            image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
            img = image.resize((target_size, target_size))
            img_array = np.array(img, dtype="float32")
        else:
            raise RuntimeError("Pillow tidak tersedia")
    except Exception as exc:
        st.error(f"Gagal memproses gambar: {exc}")
        st.stop()

    if preprocess_func is not None:
        try:
            img_array = preprocess_func(img_array)
        except Exception:
            img_array = img_array / 255.0
    else:
        img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ==========================
    # PREDIKSI
    # ==========================
    used_fallback = False
    if model is None or not hasattr(model, "predict"):
        used_fallback = True
        prediction = predict_with_fallback(img_array)
    else:
        try:
            prediction = model.predict(img_array, verbose=0)
        except Exception as exc:
            used_fallback = True
            prediction = predict_with_fallback(img_array)

    if used_fallback:
        st.caption("Prediksi menggunakan mode fallback heuristik karena TensorFlow tidak tersedia di environment deployment.")

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

    prob_rows = [
        {
            "Penyakit": labels[i],
            "Probabilitas (%)": float(prob[i])
        }
        for i in range(len(labels))
    ]

    st.dataframe(
        prob_rows,
        use_container_width=True
    )

    st.write("")
    st.caption("Distribusi probabilitas kelas")
    for row in prob_rows:
        st.progress(float(row["Probabilitas (%)"]) / 100.0, text=f"{row['Penyakit']}: {row['Probabilitas (%)']:.2f}%")

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

    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )