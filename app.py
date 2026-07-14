import io
from pathlib import Path

import numpy as np
import streamlit as st

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
except Exception as exc:
    try:
        import keras as tf
        from keras.models import load_model
        from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
        from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
    except Exception as exc2:
        tf = None
        load_model = None
        mobilenet_preprocess_input = None
        efficientnet_preprocess_input = None
        import sys
        print("TensorFlow import failed", exc, exc2, file=sys.stderr)

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


def resolve_model_dir():
    roots = []
    for base in [SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd()]:
        if base is None:
            continue
        roots.append(base)
        try:
            roots.append(base.parent)
        except Exception:
            pass

    seen = set()
    for base in roots:
        for candidate in [base / "model", base / "main" / "model", base / "app" / "model"]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if not candidate.exists():
                continue

            mobilenet_path = candidate / "mobilenetv2_cat_skin_disease.h5"
            efficientnet_path = candidate / "efficientnetb1_cat_skin_disease_final.keras"
            if mobilenet_path.exists() and efficientnet_path.exists():
                return candidate

    for base in roots:
        try:
            for model_file in base.rglob("mobilenetv2_cat_skin_disease.h5"):
                if not model_file.is_file():
                    continue
                model_dir = model_file.parent
                if (model_dir / "efficientnetb1_cat_skin_disease_final.keras").exists():
                    return model_dir
        except Exception:
            continue

    return SCRIPT_DIR / "model"


MODEL_DIR = resolve_model_dir()

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
    blueness = float(np.mean(image[..., 2] - image[..., 0]))
    darkness = float(np.mean(gray < 120))

    probs = np.array([0.30, 0.25, 0.25, 0.20], dtype="float32")

    if brightness < 90 and redness > 15:
        probs = np.array([0.55, 0.20, 0.10, 0.15], dtype="float32")
    elif contrast > 40 and saturation > 35:
        probs = np.array([0.20, 0.15, 0.15, 0.50], dtype="float32")
    elif brightness > 140 and saturation < 25:
        probs = np.array([0.10, 0.15, 0.65, 0.10], dtype="float32")
    elif darkness > 0.45 and blueness > 8:
        probs = np.array([0.35, 0.40, 0.10, 0.15], dtype="float32")
    elif redness < -5 and saturation > 25:
        probs = np.array([0.25, 0.20, 0.20, 0.35], dtype="float32")
    else:
        probs = np.array([0.30, 0.28, 0.22, 0.20], dtype="float32")

    return probs.reshape(1, -1)


def load_model_file(model_path):
    if load_model is None:
        return None, "TensorFlow/Keras tidak tersedia."

    if not model_path.exists():
        return None, f"File model tidak ditemukan: {model_path}"

    try:
        return load_model(model_path, compile=False), None
    except Exception as exc:
        try:
            return load_model(model_path), None
        except Exception as exc2:
            return None, f"{type(exc2).__name__}: {exc2}"


def load_models():
    mobilenet_path = MODEL_DIR / "mobilenetv2_cat_skin_disease.h5"
    efficientnet_path = MODEL_DIR / "efficientnetb1_cat_skin_disease_final.keras"

    if not mobilenet_path.exists() or not efficientnet_path.exists():
        return None, None, "Satu atau lebih file model tidak ditemukan."

    mobilenet, mobilenet_error = load_model_file(mobilenet_path)
    efficientnet, efficientnet_error = load_model_file(efficientnet_path)

    errors = []
    if mobilenet_error:
        errors.append(f"MobileNetV2: {mobilenet_error}")
    if efficientnet_error:
        errors.append(f"EfficientNetB1: {efficientnet_error}")

    return mobilenet, efficientnet, "; ".join(errors) if errors else None

mobilenet_model, efficientnet_model = None, None
model_load_message = ""

mobilenet_model, efficientnet_model, model_load_message = load_models()

if mobilenet_model is None or efficientnet_model is None:
    st.warning(
        "TensorFlow model belum bisa dimuat di environment ini. Aplikasi akan memakai mode fallback heuristik untuk prediksi."
    )
    if model_load_message:
        st.caption(model_load_message)

# ==========================
# CLASS NAMES
# SESUAIKAN URUTAN INI DENGAN URUTAN OUTPUT MODEL SAAT TRAINING
# ==========================
DEFAULT_CLASS_NAMES = [
    "Flea_Allergy",
    "Healthy",
    "Ringworm",
    "Scabies"
]

st.sidebar.markdown("---")

if "class_names" not in st.session_state:
    st.session_state.class_names = DEFAULT_CLASS_NAMES.copy()

class_names = st.session_state.class_names.copy()
if len(class_names) != 4:
    class_names = DEFAULT_CLASS_NAMES
    st.session_state.class_names = class_names

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

    if model is None:
        st.warning(
            f"Model {selected_model} tidak tersedia, sehingga prediksi dilakukan dengan mode fallback heuristik."
        )

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

    img_array = img_array.astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

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
        st.caption("Prediksi menggunakan mode fallback heuristik karena model tidak bisa diproses.")

    prediction = np.asarray(prediction).reshape(-1)
    probs = np.clip(prediction, 1e-8, 1.0)
    probs = probs / probs.sum()

    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)
    disease = class_names[predicted_class]
    confidence_text = f"{confidence:.2f}%"

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
            confidence_text
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

    prob = np.asarray(probs).squeeze() * 100
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