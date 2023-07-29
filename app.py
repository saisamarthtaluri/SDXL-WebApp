import streamlit as st
from diffusers import StableDiffusionXLPipeline
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
from PIL import Image
import numpy as np

def generate_image_from_text(prompt):
    base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    base.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    n_steps = 40
    high_noise_frac = 0.8

    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    return image

def generate_output_image(input_image, prompt):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe = pipe.to("cuda")
    init_image = load_image(input_image)
    image = pipe(prompt, image=init_image).images[0]

    return image


def main():
    st.title("Stable Diffusion XL Generator")

    model_choice = st.selectbox("Choose a model:", ("Text-to-Image", "Image-to-Image"))

    if model_choice == "Text-to-Image":
        user_input = st.text_input("Enter your text prompt here:")

        if st.button("Generate Image"):

            if user_input.strip():
                generated_image = generate_image_from_text(user_input)
                st.image(generated_image, caption="Generated Image", use_column_width=True)
            else:
                st.warning("Please enter a text prompt.")

    elif model_choice == "Image-to-Image":
        input_image = st.file_uploader("Upload an input image:", type=["jpg", "jpeg", "png"])

        user_input = st.text_input("Enter your text prompt here:")

        if st.button("Generate Image"):
            if input_image is not None and user_input.strip():
                input_image = Image.open(input_image)
                output_image = generate_output_image(input_image, user_input)
                st.image([input_image, output_image], caption=["Input Image", "Generated Image"], use_column_width=True)
            else:
                st.warning("Please upload an image and enter a text prompt.")

if __name__ == "__main__":
    main()


