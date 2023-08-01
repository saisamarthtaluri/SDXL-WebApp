# Stable Diffusion XL WebApp

This repository contains an application built with Streamlit and PyTorch that leverages the Stable Diffusion XL models for Text-to-Image and Image-to-Image generation tasks. The application allows you to use two types of pre-trained models, Text-to-Image and Image-to-Image, to generate images from text prompts or transform input images based on text prompts.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

You will need to install the following Python libraries:

```bash
pip install streamlit
pip install torch
pip install pillow
pip install diffusers
pip install transformers
pip install accelerate
pip install safetensors
```

Please make sure you have a CUDA-compatible GPU setup on your machine for running this project, as the current code uses CUDA for computation.

### Project Setup

Clone the repository to your local machine:

```bash
git clone https://github.com/saisamarthtaluri/SDXL-WebApp.git
```
Navigate to the project directory:

```bash
cd SDXL-WebApp
```
Run the Streamlit app:

```bash
streamlit run app.py
```
Now open your browser and go to http://localhost:8501 to see the app running.

### Using the App

Choose a model: "Text-to-Image" or "Image-to-Image".

If you select "Text-to-Image", enter your text prompt in the provided field and click "Generate Image". The app will generate an image based on your text prompt.

If you choose "Image-to-Image", upload an image and enter your text prompt in the provided fields. Then click "Generate Image". The app will generate a modified version of the uploaded image based on your text prompt.

### Example

<p align="center">
<img width="551" alt="3" src="https://github.com/saisamarthtaluri/SDXL-WebApp/assets/95733415/01d288c2-b74a-4169-9c13-647c904c9515">
</p>
