import cv2
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, messagebox, simpledialog
import imutils

class DocumentScanner:
    def __init__(self):
        self.points = []
        self.rotation_angle = 0
        self.documents = []

    def needs_rotation(self, image):
        """Проверка, нужно ли поворачивать изображение (ширина > высоты)"""
        return image.shape[1] > image.shape[0]

    def rotate_image(self, image, angle):
        """Поворот изображения на заданный угол"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def auto_rotate_text(self, image):
        """Автоматическое определение ориентации текста"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        # Определяем координаты ненулевых пикселей
        coords = np.column_stack(np.where(gray > 0))
        
        # Если не найдено достаточно текста, возвращаем исходное изображение
        if len(coords) < 100:
            return image, 0
        
        try:
            # Вычисляем угол ориентации
            angle = cv2.minAreaRect(coords)[-1]
            
            # Корректируем угол
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Поворачиваем изображение
            rotated = self.rotate_image(image, angle)
            return rotated, angle
        except:
            return image, 0

    def manual_rotation_gui(self, image):
        """Графический интерфейс для ручного поворота с автоматическим определением ориентации"""
        # Сначала пробуем автоматически определить ориентацию
        auto_rotated, auto_angle = self.auto_rotate_text(image)
        
        angles = [0, 90, 180, 270]
        current_idx = 0
        
        # Пытаемся найти ближайший угол из списка к автоматически определенному
        if auto_angle != 0:
            closest_angle = min(angles, key=lambda x: abs(x - auto_angle))
            current_idx = angles.index(closest_angle)
            
        rotated = self.rotate_image(image, angles[current_idx])
        
        cv2.namedWindow("Manual Rotation - A/D: Rotate, Enter: Confirm", cv2.WINDOW_NORMAL)
        
        while True:
            cv2.imshow("Manual Rotation - A/D: Rotate, Enter: Confirm", rotated)
            key = cv2.waitKey(0)
            
            if key == ord('a'):  # Поворот против часовой
                current_idx = (current_idx + 1) % len(angles)
            elif key == ord('d'):  # Поворот по часовой
                current_idx = (current_idx - 1) % len(angles)
            elif key == 13:  # Enter - подтверждение
                cv2.destroyAllWindows()
                self.rotation_angle = angles[current_idx]
                return rotated
            elif key == 27:  # ESC - отмена
                cv2.destroyAllWindows()
                return None
            
            rotated = self.rotate_image(image, angles[current_idx])

    def auto_detect_edges(self, image):
        """Автоматическое определение границ"""
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

    def manual_correction_gui(self, image):
        """Графический интерфейс для ручной коррекции с подписанными точками"""
        detected_points = self.auto_detect_edges(image)
        self.points = detected_points.tolist() if detected_points is not None else []
        
        if not self.points:
            h, w = image.shape[:2]
            self.points = [[0, 0], [w, 0], [w, h], [0, h]]
        
        self.points = np.array(self.points, dtype=np.float32)
        img_copy = image.copy()
        selected_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_point, img_copy
            
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, p in enumerate(self.points):
                    if np.linalg.norm(p - (x, y)) < 20:
                        selected_point = i
                        break
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if selected_point is not None and flags == cv2.EVENT_FLAG_LBUTTON:
                    self.points[selected_point] = [x, y]
                    img_copy = image.copy()
                    if len(self.points) == 4:
                        cv2.drawContours(img_copy, [self.points.astype(int)], -1, (0, 255, 0), 2)
                    for i, (px, py) in enumerate(self.points):
                        color = (0, 0, 255) if i == selected_point else (255, 0, 0)
                        cv2.circle(img_copy, (int(px), int(py)), 10, color, -1)
                        cv2.putText(img_copy, str(i+1), (int(px)+15, int(py)+5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            elif event == cv2.EVENT_LBUTTONUP:
                selected_point = None
        
        cv2.namedWindow("Manual Correction - Drag Points", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Manual Correction - Drag Points", mouse_callback)
        
        while True:
            display = img_copy.copy()
            if len(self.points) == 4:
                cv2.drawContours(display, [self.points.astype(int)], -1, (0, 255, 0), 2)
            for i, (x, y) in enumerate(self.points):
                color = (0, 0, 255) if i == selected_point else (255, 0, 0)
                cv2.circle(display, (int(x), int(y)), 10, color, -1)
                cv2.putText(display, str(i+1), (int(x)+15, int(y)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Manual Correction - Drag Points", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                detected_points = self.auto_detect_edges(image)
                self.points = detected_points.tolist() if detected_points is not None else []
                if not self.points:
                    h, w = image.shape[:2]
                    self.points = [[0, 0], [w, 0], [w, h], [0, h]]
                self.points = np.array(self.points, dtype=np.float32)
                img_copy = image.copy()
            elif key == 13:  # Enter - подтвердить
                break
            elif key == 27:  # ESC - отмена
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return self.points

    def align_document(self, image, points):
        """Выравнивание документа по точкам"""
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
        """Удаление теней"""
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
        """Баланс белого по среднему серому"""
        # Конвертируем в float32 для точных вычислений
        img_float = image.astype(np.float32)
        
        # Вычисляем средние значения для каждого канала
        avg_b = np.mean(img_float[:,:,0])
        avg_g = np.mean(img_float[:,:,1])
        avg_r = np.mean(img_float[:,:,2])
        
        # Вычисляем среднее значение по всем каналам
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        # Корректируем каналы
        img_float[:,:,0] = np.clip(img_float[:,:,0] * (avg_gray / avg_b), 0, 255)
        img_float[:,:,1] = np.clip(img_float[:,:,1] * (avg_gray / avg_g), 0, 255)
        img_float[:,:,2] = np.clip(img_float[:,:,2] * (avg_gray / avg_r), 0, 255)
        
        return img_float.astype(np.uint8)

    def process_single(self, image_path):
        """Обработка документа с удалением теней и балансом белого"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # 1. Сначала исправляем ориентацию текста
        rotated = self.manual_rotation_gui(image)
        if rotated is None:
            return None
        
        # 2. Затем определяем точки для выравнивания
        points = self.manual_correction_gui(rotated)
        if points is None:
            return None
        
        # 3. Выравнивание документа
        aligned = self.align_document(rotated, points)
        
        # 4. Улучшение (удаление теней)
        no_shadows = self.remove_shadows(aligned)
        
        # 5. Баланс белого по среднему серому
        balanced = self.gray_world_balance(no_shadows)
        
        return {
            'result': cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
        }

    def save_to_pdf(self, image, output_path, dpi=300):
        """Сохранение в PDF"""
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
        """Обработка нескольких файлов"""
        self.documents = []
        
        for path in file_paths:
            result = self.process_single(path)
            if result is not None:
                self.documents.append(result)
        
        if not self.documents:
            return False
        
        # Для одного файла - сохраняем результат
        if len(self.documents) == 1:
            output_path = os.path.splitext(file_paths[0])[0] + "_processed.pdf"
            self.save_to_pdf(self.documents[0]['result'], output_path)
            return True
        
        # Для нескольких файлов - сохраняем все в один PDF
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
        title="Выберите документы",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if file_paths:
        if scanner.process_multiple(file_paths):
            messagebox.showinfo("Готово", "Обработка завершена успешно!")
        else:
            messagebox.showwarning("Отмена", "Обработка отменена")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()