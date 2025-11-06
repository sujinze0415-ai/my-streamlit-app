import streamlit as st
st.write("THIS IS DANTE'S STREAMLIT APP")
st.title("HAJIMI Arising")

st.balloons()
st.progress(50)

st.header("This is my first step")
st.image("https://i.17173cdn.com/2fhnvk/YWxqaGBf/cms3/adxJDQbsyhluvuF.png!a-3-480x.png")
st.latex("NANBEILVDOU")
st.video("https://www.youtube.com/watch?v=UJAtXLfR0n8"）
st.latex("HAJIMI WAKE UP")
st.video("https://www.youtube.com/watch?v=IEWEiUEPmiM"）
st.latex("TIAO LOU JI")
st.video("https://www.youtube.com/watch?v=dztust9BNqE"）

st.checkbox("IF YOU LIKE HAJIMI")
st.button("SAY HI!")
st.selectbox("RATE HAJIMI", options=["1", "5", "10"])
st.multiselect("WHICH HAJIMI SOUND YOU LIKE THE MOST", options=["A", "B", "C"])
st.slider("This is a slider", min_value=0, max_value=100, value=50)
st.select_slider("This is a select slider", options=["A", "B", "C"])


st.markdown("MARK DOWN IS HERE")
