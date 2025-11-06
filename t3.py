import streamlit as st
st.write("THIS IS DANTE'S STREAMLIT APP")
st.title("HAJIMI Arising")

st.balloons()
st.progress(50)

st.header("This is my first step")
st.image("https://i.17173cdn.com/2fhnvk/YWxqaGBf/cms3/adxJDQbsyhluvuF.png!a-3-480x.png")
st.latex("真不错")
st.video("https://www.youtube.com/watch?v=UJAtXLfR0n8")
st.video(VIDEO_URL, autoplay=True, muted=True, loop=True) 

st.checkbox("This is a checkbox")
st.button("This is a button")
st.selectbox("This is a selectbox", options=["A", "B", "C"])
st.multiselect("This is a multiselect", options=["A", "B", "C"])
st.slider("This is a slider", min_value=0, max_value=100, value=50)
st.select_slider("This is a select slider", options=["A", "B", "C"])


st.markdown("MARK DOWN IS HERE")
