# gui.py (complete with dynamic classes, undo/redo, and enhanced features)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import copy
from sam_integration import load_sam_model, generate_masks
from annotation_processing import create_mask_annotation
from dataset_export import export_yolo_dataset

class ClassManager:
    def __init__(self):
        self.classes = {0: {"name": "Class 0", "color": "#FF0000"}}
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
            '#00FFFF', '#800000', '#008000', '#000080', '#808000'
        ]
        
    def add_class(self, class_id, class_name):
        if class_id in self.classes:
            raise ValueError(f"Class ID {class_id} already exists")
        if class_id >= len(self.colors):
            raise ValueError("Maximum 10 classes supported")
        self.classes[class_id] = {
            "name": class_name,
            "color": self.colors[class_id]
        }
    
    def edit_class(self, old_id, new_id, new_name):
        if old_id not in self.classes:
            raise ValueError("Class ID doesn't exist")
        if new_id != old_id and new_id in self.classes:
            raise ValueError("New class ID already exists")
            
        self.classes[new_id] = {
            "name": new_name,
            "color": self.classes[old_id]["color"]
        }
        if new_id != old_id:
            del self.classes[old_id]
    
    def get_class_info(self, class_id):
        return self.classes.get(class_id, {"name": "Unknown", "color": "#FFFFFF"})

    def get_available_classes(self):
        return sorted(self.classes.keys())

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
        self.zoom_level = 1.0
        self.pan_start = None
        self.class_manager = ClassManager()
        self.annotation_history = []
        self.history_pointer = -1
        
        self.create_widgets()
        self.setup_bindings()
        self.sam_model = load_sam_model()
        self.save_state()

    def create_widgets(self):
        # Control Panel
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Redo", command=self.redo).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="📁", command=self.upload_images).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Export", command=self.export_dataset).pack(side=tk.RIGHT, padx=2)
        
        # Class Panel
        class_frame = ttk.Frame(self.master, width=200)
        class_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(class_frame, text="Class Legend").pack(pady=5)
        self.class_canvas = tk.Canvas(class_frame, width=180, height=300)
        self.class_canvas.pack()
        ttk.Button(class_frame, text="Manage Classes", command=self.show_class_manager).pack(pady=5)
        
        # Main Canvas
        self.canvas = tk.Canvas(self.master, bg='gray', width=1280, height=1280)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        self.status = ttk.Label(self.master, text="Ready")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.update_class_display()

    def update_class_display(self):
        self.class_canvas.delete("all")
        y = 10
        for class_id in sorted(self.class_manager.classes.keys()):
            class_info = self.class_manager.get_class_info(class_id)
            self.class_canvas.create_rectangle(10, y, 30, y+20, 
                                             fill=class_info['color'], outline='black')
            self.class_canvas.create_text(40, y+10, anchor=tk.W, 
                                        text=f"{class_id}: {class_info['name']}")
            self.class_canvas.create_window(160, y+10, window=ttk.Button(
                self.class_canvas, text="✎", width=2,
                command=lambda cid=class_id: self.edit_class_dialog(cid)
            ))
            y += 30

    def show_class_manager(self):
        dialog = tk.Toplevel()
        dialog.title("Class Manager")
        
        ttk.Label(dialog, text="Class ID:").grid(row=0, column=0, padx=5, pady=5)
        id_entry = ttk.Entry(dialog)
        id_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Class Name:").grid(row=1, column=0, padx=5, pady=5)
        name_entry = ttk.Entry(dialog)
        name_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(dialog, text="Add/Update", 
                 command=lambda: self.handle_class_update(
                     id_entry.get(), name_entry.get(), dialog)
                ).grid(row=2, column=1, pady=10)

    def edit_class_dialog(self, class_id):
        dialog = tk.Toplevel()
        dialog.title("Edit Class")
        class_info = self.class_manager.get_class_info(class_id)
        
        ttk.Label(dialog, text="Class ID:").grid(row=0, column=0, padx=5, pady=5)
        id_entry = ttk.Entry(dialog)
        id_entry.insert(0, str(class_id))
        id_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Class Name:").grid(row=1, column=0, padx=5, pady=5)
        name_entry = ttk.Entry(dialog)
        name_entry.insert(0, class_info['name'])
        name_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(dialog, text="Update", 
                 command=lambda: self.handle_class_update(
                     id_entry.get(), name_entry.get(), dialog, class_id)
                ).grid(row=2, column=1, pady=10)

    def handle_class_update(self, new_id_str, new_name, dialog, old_id=None):
        try:
            new_id = int(new_id_str)
            if not new_name:
                raise ValueError("Class name cannot be empty")
                
            if old_id is None:
                self.class_manager.add_class(new_id, new_name)
            else:
                self.class_manager.edit_class(old_id, new_id, new_name)
                self.update_annotations_class(old_id, new_id)
            
            self.save_state()
            self.update_class_display()
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_annotations_class(self, old_id, new_id):
        for img_path in self.annotations:
            for ann in self.annotations[img_path]:
                if ann['class_id'] == old_id:
                    ann['class_id'] = new_id

    def save_state(self):
        state = {
            "annotations": copy.deepcopy(self.annotations),
            "classes": copy.deepcopy(self.class_manager.classes)
        }
        self.annotation_history = self.annotation_history[:self.history_pointer+1]
        self.annotation_history.append(state)
        self.history_pointer = len(self.annotation_history) - 1

    def undo(self):
        if self.history_pointer > 0:
            self.history_pointer -= 1
            self.restore_state()

    def redo(self):
        if self.history_pointer < len(self.annotation_history)-1:
            self.history_pointer += 1
            self.restore_state()

    def restore_state(self):
        state = self.annotation_history[self.history_pointer]
        self.annotations = copy.deepcopy(state["annotations"])
        self.class_manager.classes = copy.deepcopy(state["classes"])
        self.update_class_display()
        self.show_image()

    def setup_bindings(self):
        self.canvas.bind("<Button-1>", self.on_image_click)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)

    def upload_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png")])
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
        
        # Apply zoom
        w, h = pil_image.size
        new_size = (int(w * self.zoom_level), int(h * self.zoom_level))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        
        self.scale_factor = (
            self.original_image.width / pil_image.width,
            self.original_image.height / pil_image.height
        )
        
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.draw_existing_annotations()
        self.update_status()

    def draw_existing_annotations(self):
        image_path = self.images[self.current_image_index]
        if image_path in self.annotations:
            for ann in self.annotations[image_path]:
                class_info = self.class_manager.get_class_info(ann['class_id'])
                scaled_poly = [(x/self.scale_factor[0], y/self.scale_factor[1]) 
                             for x, y in ann['polygon']]
                self.canvas.create_polygon(scaled_poly, outline=class_info['color'], 
                                         fill='', width=2)

    def on_image_click(self, event):
        if not self.images:
            return
            
        original_x = event.x * self.scale_factor[0]
        original_y = event.y * self.scale_factor[1]
        
        self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3,
                              fill='blue', tags='click_marker')
        self.generate_and_show_mask(original_x, original_y)

    def generate_and_show_mask(self, x, y):
        image_path = self.images[self.current_image_index]
        image_array = np.array(self.original_image)
        
        masks = generate_masks(self.sam_model, image_array, np.array([[x, y]]))
        
        if masks is not None and len(masks) > 0:
            mask = masks[0]
            polygon = create_mask_annotation(mask)
            self.temp_mask = {
                'polygon': polygon,
                'preview_image': self.create_mask_preview(polygon)
            }
            self.show_confirmation_dialog()
        else:
            messagebox.showwarning("No Mask", "No mask found")
            self.clear_temp_mask()

    def create_mask_preview(self, polygon):
        class_info = self.class_manager.get_class_info(0)
        rgb = tuple(int(class_info['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        overlay = Image.new('RGBA', self.original_image.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        scaled_poly = [(x/self.scale_factor[0], y/self.scale_factor[1]) 
                     for x, y in polygon]
        draw.polygon(scaled_poly, fill=rgb + (50,), outline=rgb + (200,))
        return ImageTk.PhotoImage(overlay.resize(
            (int(self.original_image.width),
             int(self.original_image.height))))

    def show_confirmation_dialog(self):
        popup = tk.Toplevel(self.master)
        popup.title("Confirm Mask")
        popup.geometry("300x150")
        
        class_frame = ttk.Frame(popup)
        class_frame.pack(pady=10)
        
        ttk.Label(class_frame, text="Class:").pack(side=tk.LEFT)
        self.class_var = tk.IntVar(value=0)
        class_menu = ttk.Combobox(class_frame, textvariable=self.class_var, 
                                values=list(self.class_manager.classes.keys()))
        class_menu.pack(side=tk.LEFT, padx=5)
        
        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Keep", command=lambda: self.finalize_mask(popup)).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Discard", command=lambda: self.discard_mask(popup)).pack(side=tk.RIGHT, padx=10)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.temp_mask['preview_image'], tags='mask_preview')

    def finalize_mask(self, popup):
        try:
            class_id = self.class_var.get()
            if class_id not in self.class_manager.classes:
                raise ValueError("Invalid class selected")
            
            image_path = self.images[self.current_image_index]
            if image_path not in self.annotations:
                self.annotations[image_path] = []
                
            self.annotations[image_path].append({
                'polygon': self.temp_mask['polygon'],
                'class_id': class_id
            })
            
            self.save_state()
            popup.destroy()
            self.show_image()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def discard_mask(self, popup):
        self.clear_temp_mask()
        popup.destroy()

    def clear_temp_mask(self):
        self.temp_mask = None
        self.canvas.delete('mask_preview')
        self.canvas.delete('click_marker')

    def zoom(self, event):
        self.zoom_level *= 1.1 if event.delta > 0 else 0.9
        self.zoom_level = max(0.1, min(5.0, self.zoom_level))
        self.show_image()

    def start_pan(self, event):
        self.pan_start = (event.x, event.y)

    def pan(self, event):
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.canvas.scan_dragto(dx, dy, gain=1)
            self.pan_start = (event.x, event.y)

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def next_image(self):
        if self.current_image_index < len(self.images)-1:
            self.current_image_index += 1
            self.show_image()

    def export_dataset(self):
        if not self.annotations:
            messagebox.showerror("Error", "No annotations to export!")
            return
            
        export_dir = filedialog.askdirectory()
        if export_dir:
            try:
                class_names = {cid: info['name'] for cid, info in self.class_manager.classes.items()}
                export_yolo_dataset(self.annotations, export_dir, class_names)
                messagebox.showinfo("Success", f"Dataset exported to {export_dir}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def update_status(self):
        total = len(self.images)
        annotated = len([img for img in self.images if img in self.annotations])
        self.status.config(text=f"Image {self.current_image_index+1}/{total} | Annotated: {annotated}/{total}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.temp_mask = None
