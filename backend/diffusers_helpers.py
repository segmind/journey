# Stable Diffusion XL / Midjourney experience
import secrets
from enum import Enum
from diffusers import (DiffusionPipeline, DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverMultistepScheduler,
                       AutoencoderTiny, DPMSolverSinglestepScheduler, StableDiffusionXLPipeline,
                       StableDiffusionXLImg2ImgPipeline, KandinskyV22CombinedPipeline, AutoPipelineForText2Image,
                       StableDiffusionXLInpaintPipeline)
from main import singleton as gs
import gc

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch

aspect_ratios = {
    "1024 x 1024 (1:1 Square)": (1024, 1024),
    "1536 x 1024 (3:2 Photo)": (1536,1024),
    "1152 x 896 (9:7)": (1152, 896),
    "896 x 1152 (7:9)": (896, 1152),
    "1216 x 832 (19:13)": (1216, 832),
    "832 x 1216 (13:19)": (832, 1216),
    "1344 x 768 (7:4 Horizontal)": (1344, 768),
    "768 x 1344 (4:7 Vertical)": (768, 1344),
    "1536 x 640 (12:5 Horizontal)": (1536, 640),
    "640 x 1536 (5:12 Vertical)": (640, 1536),
}

class SchedulerType(Enum):
    DDIM = "ddim"
    HEUN = "heun"
    DPM_DISCRETE = "dpm_discrete"
    DPM_ANCESTRAL = "dpm_ancestral"
    LMS = "lms"
    PNDM = "pndm"
    EULER = "euler"
    EULER_A = "euler_a"
    DPMPP_SDE_ANCESTRAL = "dpmpp_sde_ancestral"
    DPMPP_2M = "dpmpp_2m"

scheduler_type_values = [item.value for item in SchedulerType]


controls_config = {
    "BasePipeline":{
        "expose":True,
        "type":"selectbox",
       "params": {
                "options":["XL", "Kandinsky", "revAnimated"]}

        },
    "Prompt": {
        "obj_name":"prompt",
        "expose":True,
        "type": "text_input",
        "params": {
            "value": "cyberpunk landscape"
        }
    },
    "Classic Prompt": {
        "obj_name":"prompt_2",
        "expose": True,

        "type": "text_input",
        "params": {
            "value": ""
        }
    },
    "Negative Prompt": {
        "obj_name":"n_prompt",
        "expose": True,

        "type": "text_input",
        "params": {
            "value": "watermark, characters, distorted hands, poorly drawn, ugly"
        }
    },
    "Classic Negative Prompt": {
        "obj_name":"n_prompt_2",
        "expose": True,

        "type": "text_input",
        "params": {
            "value": ""
        }
    },
    "Aspect Ratio": {
        "obj_name": "resolution",
        "expose": True,

        "type": "selectbox",
        "params": {
            "options": list(aspect_ratios.keys())
        }
    },

    "Scheduler":{
        "type": "selectbox",
        "expose": True,

        "params": {
            "options": scheduler_type_values,
            "index":8,
        }

    },

    "Mid Point": {
        "obj_name": "mid_point",
        "expose": False,

        "type": "number_input",
        "params": {
            "min_value": 0.01,
            "max_value": 1.00,
            "step": 0.01,
            "value": 0.80
        }
    },
    "Guidance Scale": {
        "obj_name": "scale",
        "expose": False,

        "type": "number_input",
        "params": {
            "min_value": 0.01,
            "max_value": 25.00,
            "step": 0.01,
            "value": 7.5
        }
    },
    "Steps": {
        "obj_name": "steps",
        "expose": False,

        "type": "slider",
        "params": {
            "min_value": 5,
            "max_value": 125,
            "value": 50
        }
    },
    "Count": {
        "type": "slider",
        "expose": False,

        "params": {
            "min_value": 1,
            "max_value": 4,
            "value": 4
        }
    },
    "Strength": {
        "obj_name": "strength",
        "expose": False,

        "type": "number_input",
        "params": {
            "min_value": 0.01,
            "max_value": 1.00,
            "step": 0.01,
            "value": 0.3
        }
    }
}
class SchedulerType(Enum):
    DDIM = "ddim"
    HEUN = "heun"
    DPM_DISCRETE = "dpm_discrete"
    DPM_ANCESTRAL = "dpm_ancestral"
    LMS = "lms"
    PNDM = "pndm"
    EULER = "euler"
    EULER_A = "euler_a"
    DPMPP_SDE_ANCESTRAL = "dpmpp_sde_ancestral"
    DPMPP_2M = "dpmpp_2m"

scheduler_type_values = [item.value for item in SchedulerType]

def get_scheduler(pipe, scheduler: SchedulerType):
    scheduler_mapping = {
        SchedulerType.DDIM: DDIMScheduler.from_config,
        SchedulerType.HEUN: HeunDiscreteScheduler.from_config,
        SchedulerType.DPM_DISCRETE: KDPM2DiscreteScheduler.from_config,
        SchedulerType.DPM_ANCESTRAL: KDPM2AncestralDiscreteScheduler.from_config,
        SchedulerType.LMS: LMSDiscreteScheduler.from_config,
        SchedulerType.PNDM: PNDMScheduler.from_config,
        SchedulerType.EULER: EulerDiscreteScheduler.from_config,
        SchedulerType.EULER_A: EulerAncestralDiscreteScheduler.from_config,
        SchedulerType.DPMPP_SDE_ANCESTRAL: DPMSolverSinglestepScheduler.from_config,
        SchedulerType.DPMPP_2M: DPMSolverMultistepScheduler.from_config
    }

    new_scheduler = scheduler_mapping[scheduler](pipe.scheduler.config)
    pipe.scheduler = new_scheduler

    return



def do_not_watermark(image):
    return image



def check_args(args, pipe):
    gen_args = {
        "prompt": args.get("prompt", "test"),
        "num_inference_steps": args.get("num_inference_steps", 10)
    }

    if isinstance(pipe, (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)):
        gen_args["prompt_2"] = args.get('prompt_2')
        gen_args["negative_prompt_2"] = args.get('negative_prompt_2')

    if isinstance(pipe, StableDiffusionXLPipeline):
        gen_args["width"], gen_args["height"] = aspect_ratios.get(args.get('resolution', "1024 x 1024 (1:1 Square)"))
        gen_args["denoising_end"] = args.get('mid_point', 1.0)
        gen_args["num_images_per_prompt"] = int(args.get('num_images_per_prompt', 1))

    elif isinstance(pipe, StableDiffusionXLImg2ImgPipeline):
        gen_args["strength"] = args.get('strength', 0.3)
        gen_args["denoising_start"] = args.get('mid_point', 0.87)
        #gen_args["image"] = gs.data.get('latents')

    elif isinstance(pipe, KandinskyV22CombinedPipeline):
        gen_args["width"], gen_args["height"] = aspect_ratios.get(args.get('resolution', "1024 x 1024 (1:1 Square)"))
        gen_args["num_images_per_prompt"] = int(args.get('num_images_per_prompt', 1))

    scheduler = args['scheduler']
    scheduler_enum = SchedulerType(scheduler)
    get_scheduler(pipe, scheduler_enum)


    return gen_args


def generate_st(args):
    import streamlit as st
    target_device = "cuda"
    if gs.data["models"]["base"].device.type != target_device:
        gs.data["models"]["base"].to(target_device)

    gen_args = check_args(args, gs.data["models"]["base"])
    progressbar = args.get('progressbar', None)

    def callback(i, t, latents):
        if progressbar:
            normalized_i = i / args.get('num_inference_steps', 10)
            progressbar.progress(normalized_i)

    gen_args["callback"] = callback

    if args["show_image"]:
        if isinstance(gs.data["models"]["base"], StableDiffusionXLPipeline):
            result = gs.data["models"]["base"].generate(**gen_args)
            gs.data["latents"] = result[0]
            result_dict = {"result_image": result[1]}
            st.session_state.preview = result[1][0]

        else:
            result_dict = {"result_image": gs.data["models"]["base"](**gen_args).images}
            gs.data["latents"] = result_dict['result_image']
            st.session_state.preview = result_dict['result_image'][0]
    else:
        gen_args['output_type'] = 'latent'
        result = gs.data["models"]["base"](**gen_args).images
        gs.data["latents"] = result
        result_dict = {}
    if progressbar:
        progressbar.progress(1.0)

    return st.session_state.preview


def load_pipeline(model_repo=None):

    print("LOADED:", gs.base_loaded)
    if model_repo == None:
        model_repo = "stabilityai/stable-diffusion-xl-base-1.0"
    if gs.base_loaded != model_repo:
        try:
            base_pipe = StableDiffusionXLPipeline.from_pretrained(
                model_repo, torch_dtype=torch.float16, variant="fp16",
                use_safetensors=True, device_map='auto'
            )
        except:
            base_pipe = StableDiffusionXLPipeline.from_pretrained(
                model_repo, torch_dtype=torch.float16, device_map='auto'
            )

        base_pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
        lowvram = False
        if lowvram:
            base_pipe.enable_model_cpu_offload()
            base_pipe.enable_vae_slicing()
            base_pipe.enable_vae_tiling()
        else:
            base_pipe.disable_attention_slicing()

        def replace_call(pipe, new_call):
            def call_with_self(*args, **kwargs):
                return new_call(pipe, *args, **kwargs)

            return call_with_self

        base_pipe.generate = replace_call(base_pipe, new_call)
        gs.data["models"]["base"] = base_pipe
        print("XL LOADED")
        gs.base_loaded = model_repo
def load_refiner_pipeline(model_repo=None):

    print("LOADED:", gs.refiner_loaded)
    if model_repo == None:
        model_repo = "stabilityai/stable-diffusion-xl-base-1.0"
    if gs.refiner_loaded != model_repo:
        lowvram = False

        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_repo,
            # text_encoder_2=gs.data["models"]["base"].text_encoder_2,
            # vae=gs.data["models"]["base"].vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map='auto'
        )
        if lowvram:
            refiner_pipe.enable_model_cpu_offload()
            refiner_pipe.enable_vae_slicing()
            refiner_pipe.enable_vae_tiling()
        else:
            refiner_pipe.disable_attention_slicing()
        refiner_pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
        gs.data["models"]["refiner"] = refiner_pipe
        print("Refiner Loaded")
        gs.refiner_loaded = model_repo

@torch.no_grad()
def new_call(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        denoising_end (`float`, *optional*):
            When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
            completed before it is intentionally prematurely terminated. As a result, the returned sample will
            still retain a substantial amount of noise as determined by the discrete timesteps selected by the
            scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
            "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
            Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
        guidance_scale (`float`, *optional*, defaults to 5.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
            of a plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.
        original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
            `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
            explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
            `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
            `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            For most cases, `target_size` should be set to the desired height and width of the generated image. If
            not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
            section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

    Examples:

    Returns:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """
    # 0. Default height and width to unet
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = self._get_add_time_ids(
        original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    # 7.1 Apply denoising_end
    num_target_steps = num_inference_steps
    if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
        discrete_timestep_cutoff = int(
            round(
                self.scheduler.config.num_train_timesteps
                - (denoising_end * self.scheduler.config.num_train_timesteps)
            )
        )
        num_target_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        #timesteps = timesteps[:num_target_steps]

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if i == num_target_steps - 1:
                return_latents = latents.clone().detach().cpu()
                if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
                    self.upcast_vae()
                    return_latents = return_latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # make sure the VAE is in float32 mode, as it overflows in float16
    if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
        self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

    latents.to('cpu')
    del latents
    print('cuda collect')
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return (return_latents, image)



def do_inpaint(prompt, init_image, mask_image, scheduler_type, steps, guidance_scale, strength, seed, selected_repo):

    scheduler = SchedulerType(scheduler_type)

    try:
        seed = int(seed)
    except:
        seed = secrets.randbelow(999999999)
    if seed == 0:
        seed = secrets.randbelow(999999999)
    generator = torch.Generator("cuda").manual_seed(seed)
    base_pipe = gs.data["models"]["base"]
    pipe = StableDiffusionXLInpaintPipeline(text_encoder=base_pipe.text_encoder,
                                             text_encoder_2=base_pipe.text_encoder_2,
                                             tokenizer=base_pipe.tokenizer,
                                             tokenizer_2=base_pipe.tokenizer_2,
                                             unet=base_pipe.unet,
                                             vae=base_pipe.vae,
                                             scheduler=base_pipe.scheduler)
    with torch.inference_mode():
        get_scheduler(pipe, scheduler)
        image = pipe(prompt=prompt,
                     image=init_image,
                     mask_image=mask_image,
                     width=init_image.size[0],
                     height=init_image.size[1],
                     strength=strength,
                     guidance_scale=guidance_scale,
                     num_inference_steps=steps,
                     generator=generator).images[0]
    return image
