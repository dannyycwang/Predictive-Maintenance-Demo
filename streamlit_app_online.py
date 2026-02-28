import streamlit as st
import streamlit_app as base_app

# Mark online/cloud mode so the app uses remote API settings by default.
st.session_state["_online_mode"] = True

if __name__ == "__main__":
    base_app.main()
