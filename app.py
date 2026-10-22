from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Қызанақтың өсуін болжау", page_icon="🍅", layout="centered")
st.title("🍅 Қызанақтың өсуін болжау")
st.write("Жылыжай жағдайлары бойынша өсімдіктің 30 күндегі биіктігін болжау")
st.info(
    "Оқу демонстрациясы. Модель 300 жасанды жазбамен оқытылған. "
    "Нәтиже нақты өсімдікке кепілдік бермейді."
)

OPT_TEMP = 24
OPT_MOISTURE = 60


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "plant_model.joblib")


bundle = load_model()
model = bundle["model"]
features = bundle["features"]

with st.form("plant_form"):
    light = st.slider("Жарық (сағат/тәулік)", 4.0, 16.0, 10.0, step=0.5)
    temp = st.slider("Температура (°C)", 15.0, 35.0, 24.0, step=0.5)
    moisture = st.slider("Топырақ ылғалдылығы (%)", 20, 80, 60)
    fert = st.slider("Тыңайтқыш (мл/апта)", 0, 40, 15)
    clicked = st.form_submit_button("Болжам жасау")

if clicked:
    temp_dev = abs(temp - OPT_TEMP)
    moist_dev = abs(moisture - OPT_MOISTURE)

    row = pd.DataFrame([{
        "light_hours": light,
        "temp_deviation": temp_dev,
        "moisture_deviation": moist_dev,
        "fertilizer_ml": fert,
    }])
    prediction = float(model.predict(row[features])[0])

    st.metric("Болжамды биіктік (30 күннен кейін)", f"{prediction:.1f} см")
    a, b = st.columns(2)
    a.metric("Температураның ауытқуы", f"{temp_dev:.1f} °C")
    b.metric("Ылғалдың ауытқуы", f"{moist_dev:.0f} %")

    st.subheader("Кеңес")
    tips = []
    if temp_dev > 4:
        tips.append(f"Температура оңтайлыдан ({OPT_TEMP} °C) алшақ. Жылыжайды {OPT_TEMP} °C шамасына жақындатыңыз.")
    if moist_dev > 15:
        tips.append(f"Топырақ ылғалы оңтайлыдан ({OPT_MOISTURE} %) алшақ. Суаруды реттеңіз.")
    if light < 8:
        tips.append("Жарық аз. Қосымша жарықтандыруды қарастырыңыз.")
    if fert > 30:
        tips.append("Тыңайтқыш көп. Артық мөлшер өсуді баяулатуы мүмкін.")
    if tips:
        for t in tips:
            st.warning(t)
    else:
        st.success("Жағдайлар оңтайлыға жақын.")

    st.caption("Кеңестер қарапайым ережелерге негізделген, нақты агрономиялық нұсқау емес.")

st.divider()
st.subheader("Модель туралы")
c1, c2 = st.columns(2)
c1.metric("Модель MAE", f"{bundle['mae']:.2f} см")
c2.metric("Базалық MAE", f"{bundle['baseline_mae']:.2f} см")
st.write("Белгілердің маңыздылығы:")
st.bar_chart(pd.Series(bundle["importance"]))
