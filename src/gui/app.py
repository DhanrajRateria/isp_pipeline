import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from isp_main.pipeline import ISPPipeline
from utils.image_io import load_image, save_image
from utils.metrics import ImageMetrics
from utils.validators import InputValidator
from isp_main.stages.base import PipelineStage

class ISPApplication:
    def __init__(self, master):
        self.master = master
        self.master.title("ISP Pipeline")
        self.master.geometry("1200x800")

        # Initialize pipeline and variables
        self.pipeline = ISPPipeline()
        self.raw_image = None  # Raw image loaded from file
        self.processed_image = None  # Image after processing

        # Create main containers
        self.create_widgets()

    def create_widgets(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.master, padding="5")
        self.control_frame.grid(row=0, column=0, sticky="nsew")

        self.image_frame = ttk.Frame(self.master, padding="5")
        self.image_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid weights
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(0, weight=1)

        # Add controls
        self.create_controls()
        self.create_image_view()

    def create_controls(self):
        # File controls
        ttk.Button(self.control_frame, text="Open Image",
                   command=self.open_image).grid(row=0, column=0, pady=5)
        ttk.Button(self.control_frame, text="Save Image",
                   command=self.save_image).grid(row=0, column=1, pady=5)

        # Pipeline stage controls
        self.stage_vars = {}
        self.stage_params = {}

        stages = [
            ("Demosaic", self.pipeline.demosaic),
            ("White Balance", self.pipeline.white_balance),
            ("Denoise", self.pipeline.denoise),
            ("Gamma Correction", self.pipeline.gamma_correction),
            ("Sharpen", self.pipeline.sharpen)
        ]

        for i, (name, func) in enumerate(stages):
            frame = ttk.LabelFrame(self.control_frame, text=name, padding="5")
            frame.grid(row=i+1, column=0, columnspan=2, pady=5, sticky="ew")

            # Enable/Disable checkbox
            var = tk.BooleanVar(value=True)
            self.stage_vars[name] = var
            ttk.Checkbutton(frame, text="Enable",
                           variable=var).grid(row=0, column=0)

            # Parameter slider
            slider = ttk.Scale(frame, from_=0, to=100, orient="horizontal")
            slider.set(50)  # Default midpoint value
            self.stage_params[name] = slider
            slider.grid(row=1, column=0, sticky="ew")

        # Process button
        ttk.Button(self.control_frame, text="Process Image",
                   command=self.process_image).grid(row=len(stages)+1, column=0, columnspan=2, pady=10)

    def create_image_view(self):
        # Create canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg='gray')
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

    def open_image(self):
        # Open file dialog to load an image
        file_path = filedialog.askopenfilename(title="Select a RAW image file",
                                               filetypes=[("RAW files", "*.raw"), ("All files", "*.*")])
        if not file_path:
            return

        # Load the raw image using RawImageReader
        reader = RawImageReader(width=1920, height=1280, bits=12)
        try:
            self.raw_image = reader.read_raw(file_path)
            self.display_image(self.raw_image)
        except Exception as e:
            print(f"Failed to load image: {e}")

    def save_image(self):
        # Save processed image to file
        if self.processed_image is None:
            print("No processed image to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            ImageWriter.save_image(self.processed_image, file_path)

    def process_image(self):
        if self.raw_image is None:
            print("No image loaded to process.")
            return

        image = self.raw_image
        for stage_name, stage_func in [
            ("Demosaic", self.pipeline.demosaic),
            ("White Balance", lambda img: self.pipeline.white_balance(img, self.stage_params["White Balance"].get())),
            ("Denoise", lambda img: self.pipeline.denoise(img, int(self.stage_params["Denoise"].get()))),
            ("Gamma Correction", lambda img: self.pipeline.gamma_correction(img, self.stage_params["Gamma Correction"].get() / 50)),
            ("Sharpen", lambda img: self.pipeline.sharpen(img, self.stage_params["Sharpen"].get() / 50))
        ]:
            if self.stage_vars[stage_name].get():  # If stage is enabled
                try:
                    image = stage_func(image)
                except Exception as e:
                    print(f"Error processing stage {stage_name}: {e}")

        self.processed_image = image
        self.display_image(self.processed_image)

    def display_image(self, image):
        # Convert image for display in tkinter
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image
        pil_image = Image.fromarray(image)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))