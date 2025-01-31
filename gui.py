# gui.py (GUI components - fixed and enhanced)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from sam_integration import load_sam_model, generate_masks
from annotation_processing import create_mask_annotation, polygons_to_bboxes
from dataset_export import export_yolo_dataset

class MainApplication(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)
        self.current_image_index = 0
        self.images = []
        self.annotations = {}
        self.temp_mask = None
        self.scale_factor = (1, 1)
        self.sam_model = None
        self.create_widgets()
        self.load_sam_model()
        
    def create_widgets(self):
        # Control panel
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.upload_btn = ttk.Button(control_frame, text="Upload Images", command=self.upload_images)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.prev_btn = ttk.Button(control_frame, text="Previous", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(control_frame, text="Next", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = ttk.Button(control_frame, text="Export Dataset", command=self.export_dataset)
        self.export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Image display
        self.canvas = tk.Canvas(self, bg='gray', width=1280, height=1280)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_image_click)
        
        # Status bar
        self.status = ttk.Label(self, text="Upload images to start annotating")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def load_sam_model(self):
        self.sam_model = load_sam_model()
        
    def upload_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if files:
            self.images = list(files)
            self.current_image_index = 0
            self.show_image()
            
    def show_image(self):
        self.clear_canvas()
        if not self.images:
            return
            
        image_path = self.images[self.current_image_index]
        pil_image = Image.open(image_path).convert('RGB')
        self.original_image = pil_image.copy()
        
        # Resize image to fit canvas while maintaining aspect ratio
        pil_image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        self.display_image = pil_image
        
        # Calculate scale factors
        self.scale_factor = (
            self.original_image.width / self.display_image.width,
            self.original_image.height / self.display_image.height
        )
        
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.config(width=self.display_image.width, height=self.display_image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Draw existing annotations
        self.draw_existing_annotations()
        self.update_status()
        
    def draw_existing_annotations(self):
        image_path = self.images[self.current_image_index]
        if image_path in self.annotations:
            for ann in self.annotations[image_path]:
                scaled_polygon = [(x/self.scale_factor[0], y/self.scale_factor[1]) for x, y in ann['polygon']]
                self.canvas.create_polygon(scaled_polygon, outline='red', fill='', width=2)

    def on_image_click(self, event):
        if not self.images:
            return
            
        # Convert click coordinates to original image scale
        original_x = event.x * self.scale_factor[0]
        original_y = event.y * self.scale_factor[1]
        
        # Show click marker
        self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, 
                              fill='blue', outline='white', tags='click_marker')
        
        # Generate mask
        self.generate_and_show_mask(original_x, original_y)

    def generate_and_show_mask(self, x, y):
        image_path = self.images[self.current_image_index]
        image_array = np.array(self.original_image)
        
        masks = generate_masks(self.sam_model, image_array, np.array([[x, y]]))
        
        # Fix: Proper check for masks
        if masks is not None and len(masks) > 0:
            mask = masks[0]
            polygon = create_mask_annotation(mask)
            
            # Create mask preview
            mask_preview = self.create_mask_preview(polygon)
            self.temp_mask = {
                'polygon': polygon,
                'preview_image': mask_preview
            }
            
            # Show confirmation dialog
            self.show_confirmation_dialog()
        else:
            messagebox.showwarning("No Mask", "No mask found for selected point")
            self.clear_temp_mask()

    def create_mask_preview(self, polygon):
        # Create transparent overlay
        overlay = Image.new('RGBA', self.display_image.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        # Scale polygon coordinates for display
        scaled_polygon = [(x/self.scale_factor[0], y/self.scale_factor[1]) for x, y in polygon]
        draw.polygon(scaled_polygon, fill=(255, 0, 0, 50), outline=(255, 0, 0, 200))
        
        return ImageTk.PhotoImage(overlay)

    def show_confirmation_dialog(self):
        popup = tk.Toplevel()
        popup.title("Confirm Mask")
        popup.geometry("300x100")
        popup.transient(self.master)
        
        label = ttk.Label(popup, text="Keep this mask?")
        label.pack(pady=10)
        
        btn_frame = ttk.Frame(popup)
        btn_frame.pack()
        
        def keep_mask():
            self.save_mask()
            popup.destroy()
            
        def discard_mask():
            self.clear_temp_mask()
            popup.destroy()
            
        ttk.Button(btn_frame, text="Keep", command=keep_mask).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Discard", command=discard_mask).pack(side=tk.RIGHT, padx=10)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.temp_mask['preview_image'], tags='mask_preview')

    def save_mask(self):
        image_path = self.images[self.current_image_index]
        if image_path not in self.annotations:
            self.annotations[image_path] = []
            
        # Convert polygon to bbox
        x_coords = [p[0] for p in self.temp_mask['polygon']]
        y_coords = [p[1] for p in self.temp_mask['polygon']]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        
        self.annotations[image_path].append({
            'polygon': self.temp_mask['polygon'],
            'bbox': bbox
        })
        
        self.temp_mask = None
        self.show_image()

    def clear_temp_mask(self):
        self.temp_mask = None
        self.canvas.delete('mask_preview')
        self.canvas.delete('click_marker')

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()
            
    def next_image(self):
        if self.current_image_index < len(self.images)-1:
            self.current_image_index += 1
            self.show_image()
            
    def update_status(self):
        total = len(self.images)
        annotated = len([img for img in self.images if img in self.annotations])
        self.status.config(text=f"Image {self.current_image_index+1}/{total} | Annotated: {annotated}/{total}")
        
    def export_dataset(self):
        if not self.annotations:
            messagebox.showerror("Error", "No annotations to export!")
            return
            
        export_dir = filedialog.askdirectory()
        if export_dir:
            try:
                export_yolo_dataset(self.annotations, export_dir)
                messagebox.showinfo("Success", f"Dataset exported to {export_dir}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def clear_canvas(self):
        self.canvas.delete("all")
        self.temp_mask = None