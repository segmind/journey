import streamlit as st
import importlib.util
import os
import json

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def save_tabs_to_json():
    with open('config/tabs.json', 'w') as f:
        json.dump({key: value["active"] for key, value in st.session_state.modules.items()}, f)
def load_tabs_from_json():
    try:
        with open('config/tabs.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def instantiate_modules():
    module_files = [f for f in os.listdir('frontend/modules') if f.endswith('.py') and f != '__init__.py']
    st.session_state.modules = {}
    tab_names = []
    if "modules" not in st.session_state:
        st.session_state.modules = {}
    for file in module_files:
        module_name = file.replace('.py', '')
        module = import_module_from_path(module_name, os.path.join('frontend/modules', file))

        st.session_state.modules[module_name] = {"name": module.plugin_info.get("name", "default_module_name"),
                                                 "module": module,
                                                 "active": True,
                                                 "prio": module.plugin_info.get("prio", 0)
                                                 }
        tab_names.append(module.plugin_info["name"])
    st.session_state.modules["toggle_tab"] = {"name": "Toggle Tabs",
                                              "module": None,
                                              "active": True,
                                              "prio": 99,
                                              }
    sorted_modules = dict(sorted(st.session_state.modules.items(), key=lambda item: item[1]['prio']))
    st.session_state.modules = sorted_modules
    if "Toggle Tabs" not in tab_names:
        tab_names.append("Toggle Tabs")
    st.session_state.tab_names = tab_names


def toggle_tabs_form():
    with st.form("Toggle Tabs"):
        # toggles = {}
        for key, value in st.session_state.modules.items():
            if value["name"] != "Toggle Tabs":
                value["active"] = st.toggle(f'Enable {value["name"]} tab', value=value["active"])

        if st.form_submit_button("Set Active Tabs"):
            save_tabs_to_json()
            st.experimental_rerun()
def set_active_tabs():
    if "active_tabs" not in st.session_state:
        st.session_state.active_tabs = st.session_state.modules

    active_tabs_from_json = load_tabs_from_json()
    for module_name, is_active in active_tabs_from_json.items():
        if module_name in st.session_state.modules:
            st.session_state.modules[module_name]["active"] = is_active

def get_active_tabs():
    return st.tabs([value["name"] for key, value in st.session_state.modules.items() if value["active"]])
def show_active_tabs(tabs):
    active_modules = {}
    x = 0
    for key, value in st.session_state.modules.items():
        if value["active"]:
            if value["name"] != "Toggle Tabs":
                active_modules[key] = value
                with tabs[x]:
                    value["module"].plugin_tab()
                    if hasattr(value["module"], "display_sidebar"):
                        value["module"].display_sidebar()
                x += 1
