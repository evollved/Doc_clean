import cv2
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, messagebox, simpledialog, Toplevel, Label, Button, Canvas, PhotoImage
import imutils
import math
import tempfile

class DocumentScanner:
    def __init__(self):
        self.points = []
        self.rotation_angle = 0
        self.documents = []

    def needs_rotation(self, image):
        """Check if image needs rotation (width > height)"""
        return image.shape[1] > image.shape[0]

    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def auto_rotate_text(self, image):
        """Automatic text orientation detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        coords = np.column_stack(np.where(gray > 0))
        
        if len(coords) < 100:
            return image, 0
        
        try:
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            rotated = self.rotate_image(image, angle)
            return rotated, angle
        except:
            return image, 0

    def save_temp_image(self, image):
        """Save image to temporary file and return PhotoImage"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Save as PNG
            cv2.imwrite(tmp.name, rgb_image)
            
            # Create PhotoImage from file
            photo = PhotoImage(file=tmp.name)
            
            # Store reference to prevent garbage collection
            if not hasattr(self, '_temp_images'):
                self._temp_images = []
            self._temp_images.append(photo)
            
            return photo

    def resize_image_for_display(self, image, max_size=800):
        """Resize image for display while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        if max(h, w) <= max_size:
            return image
        
        ratio = max_size / max(h, w)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def manual_rotation_tkinter(self, image):
        """Tkinter GUI for manual rotation with automatic orientation detection"""
        auto_rotated, auto_angle = self.auto_rotate_text(image)
        
        angles = [0, 90, 180, 270]
        current_idx = 0
        
        if auto_angle != 0:
            closest_angle = min(angles, key=lambda x: abs(x - auto_angle))
            current_idx = angles.index(closest_angle)
        
        rotated_img = self.rotate_image(image, angles[current_idx])
        
        root = Toplevel()
        root.title("Manual Rotation - A/D: Rotate, Enter: Confirm, ESC: Cancel")
        
        # Resize for display
        display_img = self.resize_image_for_display(rotated_img)
        
        # Create PhotoImage
        photo = self.save_temp_image(display_img)
        
        label = Label(root, image=photo)
        label.pack()
        
        angle_label = Label(root, text=f"Current Angle: {angles[current_idx]}°")
        angle_label.pack()
        
        original_image = image.copy()
        
        def update_display():
            nonlocal rotated_img, photo, label, angle_label
            
            # Resize for display
            display_img = self.resize_image_for_display(rotated_img)
            
            # Update PhotoImage
            photo = self.save_temp_image(display_img)
            label.config(image=photo)
            angle_label.config(text=f"Current Angle: {angles[current_idx]}°")
        
        def rotate_left():
            nonlocal current_idx, rotated_img
            current_idx = (current_idx + 1) % len(angles)
            rotated_img = self.rotate_image(original_image, angles[current_idx])
            update_display()
        
        def rotate_right():
            nonlocal current_idx, rotated_img
            current_idx = (current_idx - 1) % len(angles)
            rotated_img = self.rotate_image(original_image, angles[current_idx])
            update_display()
        
        def confirm():
            root.destroy()
            self.rotation_angle = angles[current_idx]
            root.result = rotated_img
        
        def cancel():
            root.destroy()
            root.result = None
        
        # Bind keys
        root.bind('<a>', lambda e: rotate_left())
        root.bind('<d>', lambda e: rotate_right())
        root.bind('<Return>', lambda e: confirm())
        root.bind('<Escape>', lambda e: cancel())
        
        Button(root, text="Rotate Left (A)", command=rotate_left).pack(side="left", padx=10)
        Button(root, text="Rotate Right (D)", command=rotate_right).pack(side="left", padx=10)
        Button(root, text="Confirm (Enter)", command=confirm).pack(side="left", padx=10)
        Button(root, text="Cancel (ESC)", command=cancel).pack(side="left", padx=10)
        
        root.grab_set()
        root.wait_window()
        
        return getattr(root, 'result', None)

    def auto_detect_edges(self, image):
        """Automatic edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        return None

    def draw_points_on_image(self, image, points, selected_point=None, border_size=20):
        """Draw points and document outline on image"""
        img_copy = image.copy()
        h, w = image.shape[:2]
        
        # Draw document outline in green
        if len(points) == 4:
            cv2.drawContours(img_copy, [points.astype(int)], -1, (0, 255, 0), 3)
        
        # Draw points
        for i, (px, py) in enumerate(points):
            color = (0, 0, 255) if i == selected_point else (255, 0, 0)
            cv2.circle(img_copy, (int(px), int(py)), 15, color, -1)
            cv2.putText(img_copy, str(i+1), (int(px)+20, int(py)+10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        return img_copy

    def get_magnified_view(self, image, point, radius=50):
        """Get magnified view around a point"""
        h, w = image.shape[:2]
        x, y = int(point[0]), int(point[1])
        
        # Calculate crop region
        x1 = max(0, x - radius)
        x2 = min(w, x + radius)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius)
        
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # Magnify the cropped region
        mag_size = 200
        magnified = cv2.resize(crop, (mag_size, mag_size), interpolation=cv2.INTER_CUBIC)
        
        # Draw crosshair
        center = (mag_size // 2, mag_size // 2)
        cv2.line(magnified, (center[0]-10, center[1]), (center[0]+10, center[1]), (0, 255, 0), 2)
        cv2.line(magnified, (center[0], center[1]-10), (center[0], center[1]+10), (0, 255, 0), 2)
        
        # Draw circle
        cv2.circle(magnified, center, 15, (255, 0, 0), 2)
        
        return magnified

    def manual_correction_tkinter(self, image):
        """Tkinter GUI for manual correction with magnification"""
        detected_points = self.auto_detect_edges(image)
        self.points = detected_points.tolist() if detected_points is not None else []
        
        h, w = image.shape[:2]
        if not self.points:
            # Add 20px margin from edges
            self.points = [[20, 20], [w-20, 20], [w-20, h-20], [20, h-20]]
        
        self.points = np.array(self.points, dtype=np.float32)
        
        # Create copy of image for display
        display_image = image.copy()
        
        root = Toplevel()
        root.title("Manual Correction - Click and Drag Points, R: Reset, Enter: Confirm, ESC: Cancel")
        
        # Variables
        selected_point = None
        scale = 1.0
        display_h, display_w = 0, 0
        
        # Create canvas for main image
        canvas = Canvas(root, width=800, height=600)
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create frame for controls and magnification
        right_frame = Toplevel()
        right_frame.title("Controls")
        
        # Magnification view canvas
        mag_label = Label(right_frame, text="Magnified View (200%)")
        mag_label.pack()
        mag_canvas = Canvas(right_frame, width=200, height=200, bg="white")
        mag_canvas.pack()
        
        # Instructions
        instructions = Label(right_frame, text="Instructions:\n1. Click on a point to select\n2. Drag to move\n3. R to reset\n4. Enter to confirm\n5. ESC to cancel")
        instructions.pack(pady=10)
        
        # Store PhotoImage references
        self._photo_refs = []
        
        def update_display():
            nonlocal display_image, scale, display_h, display_w
            
            # Draw points on image
            annotated_image = self.draw_points_on_image(display_image, self.points, selected_point)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            h, w = rgb_image.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                display_image_resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                display_image_resized = rgb_image
                scale = 1.0
            
            display_h, display_w = display_image_resized.shape[:2]
            
            # Create PhotoImage from numpy array
            # Convert to PPM format string
            height, width = display_image_resized.shape[:2]
            ppm_header = f'P6 {width} {height} 255\n'
            ppm_data = ppm_header.encode('ascii') + display_image_resized.tobytes()
            
            photo = PhotoImage(data=ppm_data)
            self._photo_refs.append(photo)  # Keep reference
            
            # Update canvas
            canvas.delete("all")
            canvas.config(width=display_w, height=display_h)
            canvas.create_image(0, 0, anchor="nw", image=photo)
            
            # Update magnification view if point is selected
            if selected_point is not None:
                mag_view = self.get_magnified_view(display_image, self.points[selected_point])
                if mag_view is not None:
                    mag_view_rgb = cv2.cvtColor(mag_view, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PPM
                    mag_height, mag_width = mag_view_rgb.shape[:2]
                    mag_ppm_header = f'P6 {mag_width} {mag_height} 255\n'
                    mag_ppm_data = mag_ppm_header.encode('ascii') + mag_view_rgb.tobytes()
                    
                    mag_photo = PhotoImage(data=mag_ppm_data)
                    self._photo_refs.append(mag_photo)
                    
                    mag_canvas.delete("all")
                    mag_canvas.create_image(0, 0, anchor="nw", image=mag_photo)
        
        def canvas_to_image_coords(x, y):
            """Convert canvas coordinates to image coordinates"""
            return x / scale, y / scale
        
        def on_canvas_click(event):
            nonlocal selected_point
            img_x, img_y = canvas_to_image_coords(event.x, event.y)
            
            # Check if clicked near any point
            for i, p in enumerate(self.points):
                distance = math.sqrt((p[0] - img_x)**2 + (p[1] - img_y)**2)
                if distance < 30:  # Hit radius
                    selected_point = i
                    update_display()
                    break
        
        def on_canvas_drag(event):
            if selected_point is not None:
                img_x, img_y = canvas_to_image_coords(event.x, event.y)
                
                # Update point position
                self.points[selected_point] = [img_x, img_y]
                update_display()
        
        def on_canvas_release(event):
            nonlocal selected_point
            selected_point = None
        
        def reset_points():
            nonlocal display_image
            detected_points = self.auto_detect_edges(image)
            if detected_points is not None:
                self.points = detected_points.tolist()
            else:
                self.points = [[20, 20], [w-20, 20], [w-20, h-20], [20, h-20]]
            
            self.points = np.array(self.points, dtype=np.float32)
            display_image = image.copy()
            update_display()
        
        def confirm():
            root.destroy()
            right_frame.destroy()
            root.result = self.points
        
        def cancel():
            root.destroy()
            right_frame.destroy()
            root.result = None
        
        # Bind events
        canvas.bind("<Button-1>", on_canvas_click)
        canvas.bind("<B1-Motion>", on_canvas_drag)
        canvas.bind("<ButtonRelease-1>", on_canvas_release)
        
        root.bind('<r>', lambda e: reset_points())
        root.bind('<R>', lambda e: reset_points())
        root.bind('<Return>', lambda e: confirm())
        root.bind('<Escape>', lambda e: cancel())
        
        # Add buttons to control frame
        Button(right_frame, text="Reset Points (R)", command=reset_points).pack(pady=5)
        Button(right_frame, text="Confirm (Enter)", command=confirm).pack(pady=5)
        Button(right_frame, text="Cancel (ESC)", command=cancel).pack(pady=5)
        
        # Initial display
        update_display()
        
        root.grab_set()
        root.wait_window()
        
        return getattr(root, 'result', None)

    def align_document(self, image, points):
        """Align document using points"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # top-left
        rect[2] = points[np.argmax(s)]  # bottom-right
        
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # top-right
        rect[3] = points[np.argmax(diff)]  # bottom-left
        
        width = max(
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[2] - rect[3])
        )
        height = max(
            np.linalg.norm(rect[0] - rect[3]),
            np.linalg.norm(rect[1] - rect[2])
        )
        
        dst = np.array([
            [0, 0],
            [width-1, 0], 
            [width-1, height-1],
            [0, height-1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (int(width), int(height)))

    def remove_shadows(self, img):
        """Remove shadows from image"""
        rgb_planes = cv2.split(img)
        result = []
        
        for plane in rgb_planes:
            kernel_size = min(15, max(3, min(img.shape[:2])//20*2+1))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            dilated = cv2.dilate(plane, kernel)
            bg = cv2.medianBlur(dilated, kernel_size)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            result.append(norm)
        
        return cv2.merge(result)

    def gray_world_balance(self, image):
        """Gray world white balance"""
        img_float = image.astype(np.float32)
        
        avg_b = np.mean(img_float[:,:,0])
        avg_g = np.mean(img_float[:,:,1])
        avg_r = np.mean(img_float[:,:,2])
        
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        img_float[:,:,0] = np.clip(img_float[:,:,0] * (avg_gray / avg_b), 0, 255)
        img_float[:,:,1] = np.clip(img_float[:,:,1] * (avg_gray / avg_g), 0, 255)
        img_float[:,:,2] = np.clip(img_float[:,:,2] * (avg_gray / avg_r), 0, 255)
        
        return img_float.astype(np.uint8)

    def process_single(self, image_path):
        """Process single document with shadows removal and white balance"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        # 1. Fix text orientation
        rotated = self.manual_rotation_tkinter(image)
        if rotated is None:
            return None
        
        # 2. Get alignment points
        points = self.manual_correction_tkinter(rotated)
        if points is None:
            return None
        
        # 3. Align document
        aligned = self.align_document(rotated, points)
        
        # 4. Remove shadows
        no_shadows = self.remove_shadows(aligned)
        
        # 5. White balance
        balanced = self.gray_world_balance(no_shadows)
        
        return {
            'result': cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
        }

    def save_to_pdf(self, image, output_path, dpi=300):
        """Save to PDF"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi)
            plt.imshow(image if len(image.shape)==3 else image, 
                     cmap='gray' if len(image.shape)==2 else None)
            plt.axis('off')
            plt.tight_layout(pad=0)
            pdf.savefig(fig, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()

    def process_multiple(self, file_paths):
        """Process multiple files"""
        self.documents = []
        
        for path in file_paths:
            result = self.process_single(path)
            if result is not None:
                self.documents.append(result)
        
        if not self.documents:
            return False
        
        # Single file - save result
        if len(self.documents) == 1:
            output_path = os.path.splitext(file_paths[0])[0] + "_processed.pdf"
            self.save_to_pdf(self.documents[0]['result'], output_path)
            return True
        
        # Multiple files - save to single PDF
        output_path = os.path.join(os.path.dirname(file_paths[0]), "processed_documents.pdf")
        with PdfPages(output_path) as pdf:
            for doc in self.documents:
                fig = plt.figure(figsize=(8.27, 11.69))
                plt.imshow(doc['result'])
                plt.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        return True

def main():
    scanner = DocumentScanner()
    root = Tk()
    root.withdraw()
    
    file_paths = filedialog.askopenfilenames(
        title="Select Documents",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if file_paths:
        if scanner.process_multiple(file_paths):
            messagebox.showinfo("Done", "Processing completed successfully!")
        else:
            messagebox.showwarning("Cancelled", "Processing was cancelled")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()