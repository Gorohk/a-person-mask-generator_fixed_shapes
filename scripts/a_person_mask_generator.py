from functools import reduce
import cv2
import gradio as gr
import os
from PIL import Image
import numpy as np
import mediapipe as mp

import modules.scripts as scripts
from modules.paths_internal import models_path
from modules.ui_components import FormRow, FormGroup

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MASK_OPTION_0_BACKGROUND = 'background'
MASK_OPTION_1_HAIR = 'hair'
MASK_OPTION_2_BODY = 'body (skin)'
MASK_OPTION_3_FACE = 'face (skin)'
MASK_OPTION_4_CLOTHES = 'clothes'

title = "A Person Mask Generator"


class Script(scripts.Script):

    def __init__(self):
        self.img2img: gr.Image = None

    def title(self):
        return title

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def get_mediapipe_image(self, image: Image) -> mp.Image:
        # Convert gr.Image to NumPy array
        numpy_image = np.asarray(image)

        image_format = mp.ImageFormat.SRGB

        # Convert BGR to RGB (if necessary)
        if numpy_image.shape[-1] == 4:
            image_format = mp.ImageFormat.SRGBA
        elif numpy_image.shape[-1] == 3:
            image_format = mp.ImageFormat.SRGB
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=image_format, data=numpy_image)

    def generate_mask(self, image: Image, mask_targets: list[str], mask_dilation: int) -> Image:
        if image is not None and len(mask_targets) > 0:
            model_folder_name = 'mediapipe'
            model_file_name = 'selfie_multiclass_256x256.tflite'
            model_folder_path = os.path.join(models_path, model_folder_name) if not models_path.endswith(model_folder_name) else models_path
            os.makedirs(model_folder_path, exist_ok=True)
        
            model_path = os.path.join(model_folder_path, model_file_name)
            model_url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
            if not os.path.exists(model_path):
                import urllib.request

                print(f"Downloading 'selfie_multiclass_256x256.tflite' model")
                urllib.request.urlretrieve(model_url, model_path)

            options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path),
                                            running_mode=VisionRunningMode.IMAGE,
                                            output_category_mask=True)
            # Create the image segmenter
            with ImageSegmenter.create_from_options(options) as segmenter:

                # Retrieve the masks for the segmented image
                media_pipe_image = self.get_mediapipe_image(image=image)
                segmented_masks = segmenter.segment(media_pipe_image)

                masks = []
                for target in mask_targets:
                    # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
                    # 0 - background
                    # 1 - hair
                    # 2 - body - skin
                    # 3 - face - skin
                    # 4 - clothes
                    # 5 - others(accessories)
                    mask_index = 0
                    if target == MASK_OPTION_1_HAIR:
                        mask_index = 1
                    elif target == MASK_OPTION_2_BODY:
                        mask_index = 2
                    elif target == MASK_OPTION_3_FACE:
                        mask_index = 3
                    elif target == MASK_OPTION_4_CLOTHES:
                        mask_index = 4
                    else:
                        continue

                    confidence_mask = segmented_masks.confidence_masks[mask_index].numpy_view()
                    if len(confidence_mask.shape) == 3 and confidence_mask.shape[2] == 1:
                        confidence_mask = confidence_mask[:, :, 0]
                    masks.append(confidence_mask)

                if not masks:
                    return None

                image_shape = media_pipe_image.numpy_view().shape
                height, width = image_shape[0], image_shape[1]

                combined_mask = np.zeros((height, width), dtype=np.float32)

                for mask in masks:
                    combined_mask = np.maximum(combined_mask, mask)

                binary_mask = (combined_mask > 0.25).astype(np.uint8) * 255

                if mask_dilation != 0:
                    kernel_size = 2 * abs(mask_dilation) + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    if mask_dilation > 0:
                        binary_mask = cv2.dilate(binary_mask, kernel)
                    else:
                        binary_mask = cv2.erode(binary_mask, kernel)

                mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
                mask_rgba[binary_mask == 255] = [255, 255, 255, 255]
                mask_rgba[binary_mask == 0] = [0, 0, 0, 255]

                mask_image = Image.fromarray(mask_rgba, mode='RGBA')
                return mask_image
        else:
            return None

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):
        if is_img2img:
            with gr.Accordion(title, open=False, elem_id="a_person_mask_generator_accordion") as accordion:
                with gr.Blocks():
                    with gr.Row():
                        with gr.Column():
                            enabled = gr.Checkbox(
                                label="Enable",
                                value=False,
                                elem_id=f"a_person_mask_generator_enable_checkbox",
                            )


                            mask_targets = gr.Dropdown(
                                label="Mask",
                                multiselect=True,
                                choices=[
                                    MASK_OPTION_0_BACKGROUND,
                                    MASK_OPTION_1_HAIR,
                                    MASK_OPTION_2_BODY,
                                    MASK_OPTION_3_FACE,
                                    MASK_OPTION_4_CLOTHES,
                                ],
                                value=[MASK_OPTION_3_FACE],
                                elem_id="a_person_mask_generator_mask_dropdown",
                                interactive=True,
                            )

                            gr.HTML("<div style='margin: 8px 0px !important;'></div>")

                            preview_button = gr.Button(value="Preview Mask *", elem_id="person_mask_generator_preview_button")
                            gr.HTML("<div style='margin: 8px 0px !important; opacity: 0.75;'>* Not available on 'Inpaint' tab.</div>")

                            inpaint_dilation = gr.Slider(label='Mask dilation, pixels', minimum=0, maximum=64, step=1, value=0,
                                                         elem_id="a_person_mask_generator_inpaint_dilation")

                            override_inpaint_enabled = gr.Checkbox(
                                label="Override mask settings",
                                value=False,
                                elem_id=f"a_person_mask_generator_enable_checkbox",
                            )

                            with FormGroup(elem_id="a_person_mask_generator_inpaint_controls", visible=False) as a_person_mask_generator_inpaint_controls:
                                with FormRow():
                                    mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="a_person_mask_generator_mask_blur")
                                with FormRow():
                                    inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index",
                                                                      elem_id="a_person_mask_generator_mask_mode")
                                with FormRow():
                                    inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original',
                                                               type="index",
                                                               elem_id="a_person_mask_generator_inpainting_fill")
                                with FormRow():
                                    inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture",
                                                                elem_id="a_person_mask_generator_inpaint_full_res")

                                with FormRow():
                                    inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32,
                                                                         elem_id="a_person_mask_generator_inpaint_full_res_padding")

                        with gr.Column():
                            preview_mask_image = gr.Image(value=None, label="Mask Preview", show_label=True, interactive=False, height=256, width=256,
                                                          elem_id="a_person_mask_generator_preview_mask_image")

                def update_preview_image(mask_targets: list[str], inpaint_dilation: float):
                    try:
                        mask_image = self.generate_mask(image=self.img2img, mask_targets=mask_targets, mask_dilation=int(inpaint_dilation))

                        return {
                            preview_mask_image: gr.update(value=mask_image)
                        }
                    except:
                        pass

                # change handlers
                mask_targets.change(
                    fn=update_preview_image,
                    inputs=[mask_targets, inpaint_dilation],
                    outputs=preview_mask_image
                )

                enabled.change(
                    fn=update_preview_image,
                    inputs=[mask_targets, inpaint_dilation],
                    outputs=preview_mask_image
                )

                preview_button.click(
                    fn=update_preview_image,
                    inputs=[mask_targets, inpaint_dilation],
                    outputs=preview_mask_image
                )

                inpaint_dilation.release(
                    fn=update_preview_image,
                    inputs=[mask_targets, inpaint_dilation],
                    outputs=preview_mask_image
                )

                def toggle_inpaint_controls(show: bool):
                    return {
                        a_person_mask_generator_inpaint_controls: gr.update(visible=show)
                    }

                override_inpaint_enabled.change(
                    fn=toggle_inpaint_controls,
                    inputs=[override_inpaint_enabled],
                    outputs=a_person_mask_generator_inpaint_controls
                )

                return [
                    enabled,
                    mask_targets,
                    override_inpaint_enabled,
                    mask_blur,
                    inpainting_mask_invert,
                    inpainting_fill,
                    inpaint_full_res,
                    inpaint_full_res_padding,
                    inpaint_dilation,
                ]
        else:
            return ()

    # From: https://github.com/LonicaMewinsky/gif2gif/blob/main/scripts/gif2gif.py
    # Grab the img2img image components for update later
    # Maybe there's a better way to do this?
    def after_component(self, component, **kwargs):
        def update_image(image: Image):
            if image is not None:
                self.img2img = image

        if component.elem_id == "img2img_image":
            self.img2img_image = component
            self.img2img_image.change(fn=update_image, inputs=[self.img2img_image])
            return self.img2img_image
        if component.elem_id == "img2img_sketch":
            self.img2img_sketch = component
            self.img2img_sketch.change(fn=update_image, inputs=[self.img2img_sketch])
            return self.img2img_sketch
        # The inpaint component is borked.  No Preview for that tab yet.
        # if component.elem_id == "img2maskimg":
        #    self.img2maskimg = component
        #    self.img2maskimg.select(fn=update_image, inputs=[self.img2maskimg])
        #    return self.img2maskimg
        if component.elem_id == "inpaint_sketch":
            self.inpaint_sketch = component
            self.inpaint_sketch.change(fn=update_image, inputs=[self.inpaint_sketch])
            return self.inpaint_sketch
        if component.elem_id == "img_inpaint_base":
            self.img_inpaint_base = component
            self.img_inpaint_base.change(fn=update_image, inputs=[self.img_inpaint_base])
            return self.img_inpaint_base

        return component

    """
    This function is called very early during processing begins for AlwaysVisible scripts.
    You can modify the processing object (p) here, inject hooks, etc.
    args contains all values returned by components from ui()
    """

    def before_process(
            self,
            p,
            enabled: bool = False,
            mask_targets: list[str] = [],
            override_inpaint_enabled: bool = False,
            mask_blur: int = 0,
            inpainting_mask_invert: int = 0,
            inpainting_fill: int = 0,
            inpaint_full_res: bool = False,
            inpaint_full_res_padding: int = 0,
            inpaint_dilation: int = 0,
    ):
        if enabled and len(p.init_images) > 0 and len(mask_targets) > 0:
            p.image_mask = self.generate_mask(image=p.init_images[-1], mask_targets=mask_targets, mask_dilation=inpaint_dilation)
            if override_inpaint_enabled:
                p.mask_blur = mask_blur
                p.inpainting_mask_invert = inpainting_mask_invert
                p.inpainting_fill = inpainting_fill
                p.inpaint_full_res = inpaint_full_res
                p.inpaint_full_res_padding = inpaint_full_res_padding
