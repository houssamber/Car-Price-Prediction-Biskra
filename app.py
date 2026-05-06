import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="نظام أسعار السيارات - بسكرة", page_icon="🚗")

logo_path = r'C:\Users\houssam\Desktop\Projet\rapport\logo_biskra.png'
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.warning("شعار الجامعة غير موجود.")

st.sidebar.title("تفاصيل المشروع الجامعي")
st.sidebar.info("""
**الموضوع:** تقدير أسعار السيارات - بسكرة
**الجامعة:** جامعة محمد خيضر - بسكرة

**إعداد الطلبة:**
* حسام الدين بربيش
* محمد يوسف
* مأمن محمد عبد الغني

**تحت إشراف:**
* أ. ساولي عبد الحق
""")

# ── Price Scaler ──────────────────────────────────────────────────
SCALER_FILE = 'price_scaler.json'
if os.path.exists(SCALER_FILE):
    with open(SCALER_FILE, 'r') as f:
        _s = json.load(f)
    PRICE_MEAN = _s['mean']
    PRICE_STD  = _s['std']
else:
    PRICE_MEAN = 197.93
    PRICE_STD  = 123.82
    st.sidebar.warning("price_scaler.json غير موجود")

# ── Patch slider ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("اعدادات التحليل")
patch_size = st.sidebar.select_slider("حجم الـ Patch", options=[16, 32, 64, 128], value=32)
n_patches  = (224 // patch_size) ** 2
st.sidebar.caption(f"النموذج سيحلل {n_patches} قطعة لكل صورة")

# ── OCR: auto plate reading ───────────────────────────────────────
@st.cache_resource
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(['en', 'ar'], gpu=False, verbose=False)
    except ImportError:
        return None

def extract_plate(image_pil: Image.Image, reader) -> str:
    if reader is None:
        return ""
    try:
        w, h = image_pil.size
        # نقص الثلث السفلي من الصورة حيث توجد اللوحة
        crop = image_pil.crop((0, int(h * 0.6), w, h))
        results = reader.readtext(np.array(crop), detail=0, paragraph=False)
        combined = re.sub(r'[^A-Z0-9\-]', '', ' '.join(results).upper())
        match = re.search(r'\d{3,}\w*', combined)
        return match.group(0)[:12] if match else combined[:12]
    except Exception:
        return ""

# ── Patch prediction ──────────────────────────────────────────────
_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_price(model, car_pil: Image.Image, ps: int) -> float:
    arr = np.array(car_pil.resize((224, 224)))
    h, w = arr.shape[:2]
    patches = [arr[r:r+ps, c:c+ps]
               for r in range(0, h-ps+1, ps)
               for c in range(0, w-ps+1, ps)]
    if not patches:
        patches = [arr]
    tensors = []
    for p in patches:
        try:
            tensors.append(_tf(Image.fromarray(p)))
        except Exception:
            continue
    if not tensors:
        tensors = [_tf(car_pil)]
    with torch.no_grad():
        raw = model(torch.stack(tensors)).mean().item()
    return max((raw * PRICE_STD) + PRICE_MEAN, 30.0)

# ── Predictions log ───────────────────────────────────────────────
PREDICTIONS_LOG = 'predictions_log.csv'

def read_log() -> pd.DataFrame:
    if not os.path.exists(PREDICTIONS_LOG):
        return pd.DataFrame(columns=['Date', 'Image_Name', 'Registration', 'Predicted_Price'])
    try:
        # on_bad_lines='skip' يحل مشكلة "Expected 3 fields saw 5"
        return pd.read_csv(PREDICTIONS_LOG, encoding='utf-8-sig', on_bad_lines='skip')
    except Exception:
        return pd.DataFrame(columns=['Date', 'Image_Name', 'Registration', 'Predicted_Price'])

def save_prediction(img_name: str, reg: str, price: float):
    row = pd.DataFrame([{
        'Date'           : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Image_Name'     : img_name,
        'Registration'   : reg.upper() if reg else 'UNKNOWN',
        'Predicted_Price': round(price, 2),
    }])
    if not os.path.isfile(PREDICTIONS_LOG):
        row.to_csv(PREDICTIONS_LOG, index=False, encoding='utf-8-sig')
    else:
        row.to_csv(PREDICTIONS_LOG, mode='a', header=False, index=False, encoding='utf-8-sig')

def lookup_previous(reg: str):
    if not reg or reg.upper() == 'UNKNOWN':
        return None
    df = read_log()
    if 'Registration' not in df.columns or 'Predicted_Price' not in df.columns:
        return None
    match = df[df['Registration'].astype(str).str.upper() == reg.upper()]
    return float(match.iloc[-1]['Predicted_Price']) if not match.empty else None

# ── Load models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    pm   = models.resnet18()
    pm.fc = nn.Linear(pm.fc.in_features, 1)
    if os.path.exists('car_price_model.pth'):
        pm.load_state_dict(torch.load('car_price_model.pth', map_location='cpu'))
        pm.eval()
    else:
        st.error("car_price_model.pth غير موجود")
    return yolo, pm

yolo_model, price_model = load_models()
ocr_reader              = load_ocr()

# ── Main UI ───────────────────────────────────────────────────────
st.title("تقدير أسعار السيارات - مشروع بسكرة")

uploaded_file = st.file_uploader("اختر صورة السيارة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)

    # قراءة اللوحة تلقائياً
    with st.spinner("جاري قراءة رقم اللوحة تلقائياً..."):
        auto_plate = extract_plate(image, ocr_reader)

    registration = st.text_input(
        "رقم تسجيل السيارة (لوحة الأرقام)",
        value=auto_plate,
        help="تم اكتشافه تلقائياً من الصورة - يمكنك تعديله إذا كان خاطئاً"
    )

    if auto_plate:
        st.caption(f"تم اكتشاف اللوحة تلقائياً: **{auto_plate}**")
    else:
        st.caption("لم يتم اكتشاف اللوحة - يمكنك إدخالها يدوياً")

    if st.button('احسب السعر'):
        with st.spinner('جاري التحليل...'):
            reg = registration.strip()

            # تحقق من السجل: نفس اللوحة = نفس السعر
            prev = lookup_previous(reg)
            if prev is not None:
                st.info(f"هذه السيارة (لوحة: {reg}) سبق تقييمها - نعيد نفس السعر")
                st.metric("السعر التقديري (مليون سنتيم)", f"{prev:,.2f}")
                st.caption(f"= {prev * 10000:,.0f} دج تقريباً")
            else:
                results   = yolo_model(image)
                cars      = results.pandas().xyxy[0]
                car_boxes = cars[cars['name'] == 'car']

                if not car_boxes.empty:
                    car_boxes = car_boxes.copy()
                    car_boxes['area'] = ((car_boxes['xmax'] - car_boxes['xmin']) *
                                        (car_boxes['ymax'] - car_boxes['ymin']))
                    lg = car_boxes.loc[car_boxes['area'].idxmax()]
                    car_crop = image.crop((int(lg['xmin']), int(lg['ymin']),
                                          int(lg['xmax']), int(lg['ymax'])))

                    price = predict_price(price_model, car_crop, patch_size)

                    st.success(f"تم اكتشاف {len(car_boxes)} سيارة - تحليل {n_patches} قطعة")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("السعر التقديري (مليون سنتيم)", f"{price:,.2f}")
                        st.caption(f"= {price * 10000:,.0f} دج تقريباً")
                    with col2:
                        st.image(car_crop, caption=f"الجزء المحلل ({n_patches} patches)",
                                 use_container_width=True)

                    save_prediction(uploaded_file.name, reg, price)
                    st.sidebar.success("تم حفظ النتيجة")
                else:
                    st.error("لم يتم اكتشاف سيارة واضحة في الصورة.")

# سجل التوقعات
if st.sidebar.checkbox("اظهار سجل التوقعات"):
    df_log = read_log()
    if df_log.empty:
        st.sidebar.info("السجل فارغ.")
    else:
        st.sidebar.dataframe(df_log.tail(10))
