import streamlit as st
from PIL import Image

def about_page():
    st.title("Learn About Norbel Team")


    st.write("Berthama e ekipit tone perbehet nga individe ambicioze te fokusuar ne zhvillimin dhe integrimin e Inteligjences Artificiale (AI).  Me anen e algoritmeve te avancuara te Machine Learming (ML) ne kemi si qellim t'i japim zgjidhje problemeve te ndryshme"
             " ne fushat e : ")
    st.markdown("- Biznesit: Retail Companies, Banking Networks, Stock Prediction etc.")
    st.markdown("- Civil Engineering: Urban Traffic")
    st.markdown("- Imazherise: MRI scans, Stomatology etc.")

    st.subheader("Team Members")

    ##For members
    st.header("Enid Vyshka: Senior Data Science Enigneer (Project Lead)")
    st.info('Data Analyst|Machine Learning|Python|Data Visualization')
    # lead=Image.open('Team photos/photo_2023-03-16_18-13-22.jpg')
    # size=(400,400)
    # lead_image=lead.resize(size)
    # st.image(lead_image)

    st.subheader("Connect with Enid:")

    # Button to send an email
    if st.button("Contact Me via Email"):
        st.markdown('<a href="mailto:enid.vyshka@gmail.com">Send Email</a>', unsafe_allow_html=True)

    # Button to visit LinkedIn profile
    if st.button("Visit My LinkedIn Profile"):
        st.markdown('<a href="https://www.linkedin.com/in/enid-vyshka-36b448104/">LinkedIn</a>', unsafe_allow_html=True)
