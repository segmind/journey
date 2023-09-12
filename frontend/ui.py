import os

import streamlit as st

st.set_page_config(layout="wide")

def main():
    from frontend.ui_helpers import (instantiate_modules, set_active_tabs,
                                     toggle_tabs_form, show_active_tabs,
                                     get_active_tabs)
    from backend.singleton_params import instantiate_singleton_params

    instantiate_singleton_params()

    if "modules" not in st.session_state:
        instantiate_modules()

    set_active_tabs()

    tabs = get_active_tabs()

    with tabs[len(tabs) - 1]:

        toggle_tabs_form()

    show_active_tabs(tabs)

if __name__ == "__main__":
    main()