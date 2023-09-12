import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from backend.sdxl_styles import style_keys

plugin_info = {"name": "Main",
              "prio":0}

defaults = {
    'method':{
        'default':'txt2img'
    },
    'args':{
        'default':{}
    },
    'use_img2img':{
        'default':False
    },
    'use_inpaint':{
        'default':False
    },
    'current_image':{
        'default':None
    }
}

def prepare_session_state():
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value['default']

def show_controls():

    col1, col2 = st.columns([5, 1])
    with col1:
        if st.session_state.method in ['txt2img', 'img2img']:
            prompt = st.text_area('Prompt')
            negative_prompt = st.text_area('Negative Prompt')
    with col2:
        style_1 = st.selectbox('Prompt Style', style_keys)

        if st.session_state.method in ['txt2img', 'img2img']:
            submit = st.form_submit_button('Process')
        #style_2 = st.selectbox('Style 2', ['None'])
    st.session_state.args.update(**locals())

    def handle_events():
        pass
        # if test_button:
        #     print(st.session_state.args)

    st.session_state.handle_events = handle_events

class SyncedSliderAndInput:
    def __init__(self, label, min_value, max_value, value, step):
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.step = step

        # Initialize the state variable if it doesn't exist
        if f'{self.label}_value' not in st.session_state:
            st.session_state[f'{self.label}_value'] = value

    def display(self):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state[f'{self.label}_value'] = st.slider(self.label, self.min_value, self.max_value, st.session_state[f'{self.label}_value'], self.step)
        with col2:
            st.session_state[f'{self.label}_value'] = st.number_input(self.label, self.min_value, self.max_value, st.session_state[f'{self.label}_value'], self.step)
        return st.session_state[f'{self.label}_value']


class SessionHandler:

    promptable = True

    def __init__(self):
        self.method = 'default'

    def preprocess(self, image=None):
        return image
    def process(self, image):
        return image
    def postprocess(self, image):
        return image

    def __call__(self, *args, **kwargs):
        pass

def show_previewer(image, stroke_width):
    fill_color = "rgba(255, 255, 255, 0.0)"
    stroke_color = "rgba(255, 255, 255, 1.0)"
    bg_color = "rgba(0, 0, 0, 1.0)"
    drawing_mode = "freedraw"

    if st.session_state.current_image is not None:

        width, height = st.session_state.current_image.size

        # Check if either width or height exceeds 1280
        max_dimension = max(width, height)
        if max_dimension > 1280:
            scaling_factor = 1280 / max_dimension
            width = int(width * scaling_factor)
            height = int(height * scaling_factor)

        st.session_state.previewer = st_canvas(
            fill_color=fill_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=image,
            update_streamlit=False,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key="canvas",
        )

def plugin_tab(*args, **kwargs):

    global main_form
    #global subtabs
    #
    prepare_session_state()

    main_form = st.form('main')
    # #
    with main_form:
        show_controls()
    st.session_state.input_image = st.file_uploader("Import Image")
    if st.session_state.input_image is not None:
        col1, col2 = st.columns([2,8])
        with col1:
            stroke_width = st.number_input("Brush Size",
                                           value=64,
                                           min_value=1,
                                           max_value=100)
            use_init = st.checkbox("Use Init Image")
            if use_init:
                init_method = st.selectbox("Select Process", ["Image to Image", "Inpaint", "Outpaint", "Upscale"])

                if init_method == "Outpaint":
                    target_width = st.number_input('Target width', min_value=64, value=768, step=8)
                    target_height = st.number_input('Target height', min_value=64, value=768, step=8)
                    scale = st.slider('Scale', min_value=0.1, max_value=2.0, step=0.01, value=0.4)
                    offset_x = st.slider('X Offset', min_value=0, max_value=1024, step=8, value=64)
                    offset_y = st.slider('Y Offset', min_value=0, max_value=1024, step=8, value=64)

        with col2:
            st.session_state.current_image = Image.open(st.session_state.input_image)
            show_previewer(st.session_state.current_image, stroke_width)
