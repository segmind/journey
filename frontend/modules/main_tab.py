import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from backend.diffusers_helpers import scheduler_type_values, aspect_ratios
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
    'use_init':{
        'default':False
    },
    'use_inpaint':{
        'default':False
    },
    'current_image':{
        'default':None
    },
    'images':{
        'default':[]
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
        scheduler = st.selectbox('Scheduler', scheduler_type_values)
        resolution = st.selectbox('Resolution', list(aspect_ratios.keys()))
        if st.session_state.method in ['txt2img', 'img2img']:
            submit = st.form_submit_button('Process')
        #style_2 = st.selectbox('Style 2', ['None'])
    st.session_state.args.update(**locals())

    def handle_events():
        pass
        # if test_button:
        #     print(st.session_state.args)

    st.session_state.handle_events = handle_events

    return ({"prompt":prompt,
            "negative_prompt":negative_prompt,
            "style":style_1,
            "scheduler":scheduler,
            "resolution":resolution}, submit)

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
        self.method = 'txt2img'
        self.preprocessors = []
        self.processors = []
        self.postprocessors = []

    def reset_processors(self):
        self.preprocessors.clear()
        self.processors.clear()
        self.postprocessors.clear()

    def preprocess(self, gen_args):
        image = None
        if len(self.preprocessors) > 0:
            for preprocessor in self.preprocessors:
                image = preprocessor(gen_args)
            return image

    def process(self, gen_args):
        image = None
        if len(self.processors) > 0:
            for processor in self.processors:
                image = processor(gen_args)
        return image

    def postprocess(self, gen_args):
        image = None
        if len(self.postprocessors) > 0:
            for postprocessor in self.postprocessors:
                image = postprocessor(gen_args)
        return image

    def __call__(self, gen_args:dict, *args, **kwargs):

        self.reset_processors()
        use_init = gen_args.get("use_init", False)
        init_image = gen_args.get("init_image", None)
        method = gen_args.get("method", "txt2img")
        method = "txt2img" if not use_init else method

        if init_image is None and method != "txt2img":
            print("Setting method to Text to Image since no Init Image was provided,\nplease load, or generate an image to use this feature")

        valid_methods = [value for _, value in methods.items()]
        if method in valid_methods:

            if method == "txt2img":
                self.processors.append(run_txt2img)
                image = self.process(gen_args)
                return image


def run_txt2img(gen_args):

    from backend.diffusers_helpers import load_pipeline, generate_st

    model_repo = gen_args.get("model_repo", None)
    load_pipeline(model_repo=model_repo)
    gen_args["show_image"] = True
    image = generate_st(gen_args)

    st.session_state.images.append(image)

    return image


def show_previewer(image, stroke_width):
    fill_color = "rgba(255, 255, 255, 0.0)"
    stroke_color = "rgba(255, 255, 255, 1.0)"
    bg_color = "rgba(0, 0, 0, 1.0)"
    drawing_mode = "freedraw"
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

methods = {"Image to Image":"img2img",
           "Inpaint":"inpaint",
           "Outpaint":"outpaint",
           "Upscale":"upscale",
           "Text2Image":"txt2img"}

method_list = ["Image to Image", "Inpaint", "Outpaint", "Upscale"]


def show_inpainter():
    col1, col2 = st.columns([2, 8], gap="small")
    with col1:
        if st.session_state.current_image is not None or st.session_state.input_image is not None:
            stroke_width = st.number_input("Brush Size",
                                           value=64,
                                           min_value=1,
                                           max_value=100)
            st.session_state.use_init = st.checkbox("Use Init Image")
            if st.session_state.use_init:
                st.session_state.init_method = methods[st.selectbox("Select Process", method_list)]

                if st.session_state.init_method == "Outpaint":
                    target_width = st.number_input('Target width', min_value=64, value=768, step=8)
                    target_height = st.number_input('Target height', min_value=64, value=768, step=8)
                    scale = st.slider('Scale', min_value=0.1, max_value=2.0, step=0.01, value=0.4)
                    offset_x = st.slider('X Offset', min_value=0, max_value=1024, step=8, value=64)
                    offset_y = st.slider('Y Offset', min_value=0, max_value=1024, step=8, value=64)
                if st.session_state.init_method == "Inpaint":
                    pass

    with col2:
        if st.session_state.input_image is not None:
            st.session_state.current_image = Image.open(st.session_state.input_image)
        if st.session_state.current_image is not None:
            show_previewer(st.session_state.current_image, stroke_width)


def show_advanced_controls(gen_args):

    col1, col2, col3 = st.columns([3,3,3])

    with col1:
        steps = st.number_input("Steps", min_value=1, max_value=1000, value=50)
        guidance_scale = st.number_input("Guidance Scale", min_value=0.0, max_value=25.0, value=6.5)
        strength = st.number_input("Strength", min_value=0.0, max_value=1.0, value=1.0)
        seed = st.number_input("Seed", min_value=-1, max_value=999999999999, value=-1)

    args = {"steps":steps,
            "guidance_scale":guidance_scale,
            "strength":strength,
            "seed":seed}

    gen_args.update(args)
    return gen_args

def image_history():
    # Determine the number of columns (up to 8)
    num_images = len(st.session_state.images)
    num_columns = min(num_images, 6)

    # If there's no start_index in the session state, initialize it to 0
    if "start_index" not in st.session_state:
        st.session_state.start_index = 0
    buttons = {}
    if num_columns > 0:
        # Create the columns
        image_columns = st.columns(num_columns)

        # Display the images in the columns
        for idx, col in enumerate(image_columns):
            if idx + st.session_state.start_index < num_images:
                col.image(st.session_state.images[
                              idx + st.session_state.start_index], width=256)  # Replace with your method of displaying images
                with col:
                    buttons[idx] = st.button("View Image", key=f"main_preview_button_{idx}")

                    if buttons[idx]:

                        st.session_state.current_image = st.session_state.images[
                                  idx + st.session_state.start_index]
                        st.experimental_rerun()

        # Step through images
        left, _, right = st.columns([1, 6, 1])
        if left.button("←", key=f"main_left_button") and st.session_state.start_index > 0:
            st.session_state.start_index -= 1  # Move the images to the left

        if right.button("→", key=f"main_right_button") and st.session_state.start_index + num_columns < num_images:
            st.session_state.start_index += 1  # Move the images to the right


def plugin_tab(*args, **kwargs):

    # Initializing the short form of Session Handler
    if "sh" not in st.session_state:
        st.session_state.sh = SessionHandler()

    prepare_session_state()
    tab1, tab2 = st.tabs(["Prompt", "Generation Parameters"])
    with tab1:
        with st.form('main'):
            gen_args, submit = show_controls()
        st.session_state.input_image = st.file_uploader("Import Image")

    with tab2:
        gen_args = show_advanced_controls(gen_args)

    if submit:
        gen_args["init_image"] = st.session_state.current_image
        gen_args["use_init"] = st.session_state.use_init
        st.session_state.current_image = st.session_state.sh(gen_args)

    show_inpainter()

    image_history()
