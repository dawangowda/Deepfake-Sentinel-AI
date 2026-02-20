import os
import cv2
from mtcnn.mtcnn import MTCNN

print("Libraries loaded. Starting preprocessing...")

# --- Configuration ---
SOURCE_DIR = 'dataset' # The folder with train/validation videos
OUTPUT_DIR = 'processed_dataset' # The folder where face images will be saved
FRAME_SAMPLE_RATE = 15 # Process every 15th frame
TARGET_SIZE = (224, 224) # The size our model will expect

# --- Initialize Face Detector ---
detector = MTCNN()

# --- Helper function to process a directory of videos ---
def process_video_directory(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return
        
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi'))]
    print(f"Found {len(video_files)} videos in {input_dir}.")
    
    for i, video_filename in enumerate(video_files):
        video_path = os.path.join(input_dir, video_filename)
        video_capture = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break # End of video
            
            # Only process every Nth frame
            if frame_count % FRAME_SAMPLE_RATE == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(frame_rgb)
                
                if results:
                    x1, y1, width, height = results[0]['box']
                    x2, y2 = x1 + width, y1 + height
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        resized_face = cv2.resize(face_roi, TARGET_SIZE)
                        output_filename = f"{os.path.splitext(video_filename)[0]}_frame{frame_count}.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        cv2.imwrite(output_path, resized_face)
                        
            frame_count += 1
        
        video_capture.release()
        print(f"Finished processing video {i+1}/{len(video_files)}: {video_filename}")

# --- Main Execution ---
for split in ['train', 'validation']:
    for category in ['real', 'fake']:
        source_path = os.path.join(SOURCE_DIR, split, category)
        output_path = os.path.join(OUTPUT_DIR, split, category)
        
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\n--- Processing {split}/{category} ---")
        process_video_directory(source_path, output_path)
        
print("\nPreprocessing complete! Your face images are in the 'processed_dataset' folder.")