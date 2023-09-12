import os
import json
import streamlit as st
import importlib.util
from typing import Any, Dict, List, Optional, Union

def import_module_from_path(module_name: str, file_path: str) -> Any:
    """
    Import a Python module from a given file path.

    Args:
        module_name: The desired name for the module.
        file_path: The path to the .py file to be imported.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def save_tabs_to_json() -> None:
    """
    Save active tab states to a JSON file.
    """
    with open('config/tabs.json', 'w') as f:
        json.dump({key: value["active"] for key, value in st.session_state.modules.items()}, f)

def load_tabs_from_json() -> Dict[str, bool]:
    """
    Load active tab states from a JSON file.

    Returns:
        A dictionary mapping module names to their active states.
    """
    try:
        with open('config/tabs.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def instantiate_modules() -> None:
    """
    Instantiate the frontend modules and organize them based on their priorities.
    """
    module_files = [f for f in os.listdir('frontend/modules') if f.endswith('.py') and f != '__init__.py']
    st.session_state.modules = {}
    tab_names = []
    for file in module_files:
        module_name = file.replace('.py', '')
        module = import_module_from_path(module_name, os.path.join('frontend/modules', file))
        if hasattr(module, "plugin_info"):
            st.session_state.modules[module_name] = {
                "name": module.plugin_info.get("name", "default_module_name"),
                "module": module,
                "active": True,
                "prio": module.plugin_info.get("prio", 0)
            }
            tab_names.append(module.plugin_info["name"])

    st.session_state.modules["toggle_tab"] = {
        "name": "Toggle Tabs",
        "module": None,
        "active": True,
        "prio": 99,
    }
    sorted_modules = dict(sorted(st.session_state.modules.items(), key=lambda item: item[1]['prio']))
    st.session_state.modules = sorted_modules
    if "Toggle Tabs" not in tab_names:
        tab_names.append("Toggle Tabs")
    st.session_state.tab_names = tab_names

def toggle_tabs_form() -> None:
    """
    Display a form to toggle the visibility of different tabs.
    """
    with st.form("Toggle Tabs"):
        for key, value in st.session_state.modules.items():
            if value["name"] != "Toggle Tabs":
                value["active"] = st.toggle(f'Enable {value["name"]} tab', value=value["active"])

        if st.form_submit_button("Set Active Tabs"):
            save_tabs_to_json()
            st.experimental_rerun()

def set_active_tabs() -> None:
    """
    Set the active tabs based on the JSON configuration.
    """
    if "active_tabs" not in st.session_state:
        st.session_state.active_tabs = st.session_state.modules

    active_tabs_from_json = load_tabs_from_json()
    for module_name, is_active in active_tabs_from_json.items():
        if module_name in st.session_state.modules:
            st.session_state.modules[module_name]["active"] = is_active

def get_active_tabs() -> List[str]:
    """
    Retrieve the list of active tabs.

    Returns:
        A list containing the names of the active tabs.
    """
    return st.tabs([value["name"] for key, value in st.session_state.modules.items() if value["active"]])

def show_active_tabs(tabs: Union[List[str], List[Any]]) -> None:
    """
    Display the content of the active tabs.

    Args:
        tabs: A list of Streamlit container objects representing the tabs.
    """
    active_modules = {}
    x = 0
    for key, value in st.session_state.modules.items():
        if value["active"]:
            if value["name"] != "Toggle Tabs":
                active_modules[key] = value
                with tabs[x]:
                    if hasattr(value["module"], "plugin_tab"):
                        value["module"].plugin_tab()
                    if hasattr(value["module"], "display_sidebar"):
                        value["module"].display_sidebar()
                x += 1
