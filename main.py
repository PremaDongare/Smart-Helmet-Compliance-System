import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import os
from datetime import datetime
import xlwings as xw
import logging
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
    OCR_ENGINE = "paddle"
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
except ImportError:
    try:
        import easyocr
        OCR_ENGINE = "easyocr"
        ocr = easyocr.Reader(['en'])
    except ImportError:
        OCR_ENGINE = None
        logger.warning("No OCR engine installed. Please install paddleocr or easyocr")

class HelmetDetectionSystem:
    def __init__(self):
        # Initialize video capture
        self.cap = self.initialize_video_capture('final.mp4')
        
        # Load YOLOv8 model
        self.model = YOLO("best.pt")
        self.names = self.model.names
        
        # Define detection area polygon
        self.area = np.array([[1, 173], [62, 468], [608, 431], [364, 155]], np.int32)
        
        # Initialize Excel workbook
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.excel_file_path = os.path.join(self.current_date, f"{self.current_date}.xlsx")
        self.wb, self.ws = self.initialize_excel()
        
        # Tracking variables
        self.processed_track_ids = set()
        
        # Create output directory
        os.makedirs(self.current_date, exist_ok=True)
        
        # Initialize mouse callback
        cv2.namedWindow('RGB')
        cv2.setMouseCallback('RGB', self.mouse_callback)

    @staticmethod
    def initialize_video_capture(video_path: str) -> cv2.VideoCapture:
        """Initialize video capture source"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {video_path}")
            raise IOError("Video source not available")
        return cap

    def initialize_excel(self) -> Tuple[xw.Book, xw.Sheet]:
        """Initialize Excel workbook and worksheet"""
        try:
            if os.path.exists(self.excel_file_path):
                wb = xw.Book(self.excel_file_path)
            else:
                wb = xw.Book()
            ws = wb.sheets[0]
            if ws.range("A1").value is None:
                ws.range("A1").value = ["Number Plate", "Date", "Time"]
            return wb, ws
        except Exception as e:
            logger.error(f"Excel initialization failed: {e}")
            raise

    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        """Mouse callback function for debugging"""
        if event == cv2.EVENT_MOUSEMOVE:
            logger.debug(f"Mouse position: ({x}, {y})")

    def perform_ocr(self, image_array: np.ndarray) -> str:
        """Perform OCR on cropped number plate image"""
        if image_array is None:
            raise ValueError("Image is None")
        
        if OCR_ENGINE == "paddle":
            results = ocr.ocr(image_array, rec=True)
            detected_text = [result[1][0] for result in results[0] if results[0] is not None]
        elif OCR_ENGINE == "easyocr":
            results = ocr.readtext(image_array)
            detected_text = [result[1] for result in results]
        else:
            logger.warning("No OCR engine available")
            return "OCR_NOT_AVAILABLE"
        
        return ''.join(detected_text) if detected_text else "NO_PLATE_DETECTED"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame"""
        frame = cv2.resize(frame, (1020, 500))
        results = self.model.track(frame, persist=True)
        
        no_helmet_detected = False
        numberplate_box = None
        numberplate_track_id = None
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = self.names[class_id]
                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                
                if cv2.pointPolygonTest(self.area, (cx, cy), False) >= 0:
                    if class_name == 'no-helmet':
                        no_helmet_detected = True
                    elif class_name == 'numberplate':
                        numberplate_box = box
                        numberplate_track_id = track_id
            
            if all([no_helmet_detected, numberplate_box, numberplate_track_id not in self.processed_track_ids]):
                self.process_numberplate(frame, numberplate_box, numberplate_track_id)
        
        # Draw detection area
        cv2.polylines(frame, [self.area], True, (255, 0, 255), 2)
        return frame

    def process_numberplate(self, frame: np.ndarray, box: List[int], track_id: int):
        """Process detected number plate"""
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (120, 85))
        
        # Perform OCR
        plate_text = self.perform_ocr(crop)
        logger.info(f"Detected Number Plate: {plate_text}")
        
        # Save cropped image
        current_time = datetime.now().strftime('%H-%M-%S-%f')[:12]
        crop_path = os.path.join(self.current_date, f"{plate_text}_{current_time}.jpg")
        cv2.imwrite(crop_path, crop)
        
        # Save to Excel
        last_row = self.ws.range("A" + str(self.ws.cells.last_cell.row)).end('up').row
        self.ws.range(f"A{last_row+1}").value = [plate_text, self.current_date, current_time]
        
        # Mark as processed
        self.processed_track_ids.add(track_id)
        
        # Visual feedback
        cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)

    def run(self):
        """Main processing loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Video processing completed")
                    break
                
                processed_frame = self.process_frame(frame)
                cv2.imshow("RGB", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Processing stopped by user")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Processing interrupted")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.wb.save(self.excel_file_path)
        self.wb.close()
        logger.info("Resources released and data saved")

if __name__ == "__main__":
    try:
        logger.info("Starting Helmet Detection System")
        system = HelmetDetectionSystem()
        system.run()
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)