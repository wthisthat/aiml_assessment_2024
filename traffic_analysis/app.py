import streamlit as st
import cv2
import tempfile
import torch
from ultralytics import YOLO

# Load YOLOv8 model (you can choose any pre-trained model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('traffic_analysis/yolov8s_custom.pt')  # using the smallest YOLOv8 model
model.to(device)
max_track_id = 0

def process_frame(frame):
    """ Accept image frame input and inference Yolov8 object detection model with tracking function, annotate the bounding boxes and vehicle count"""
    global max_track_id
    # Run YOLOv8 inference on the frame
    results = model.track(frame, conf=0.5, persist=True, stream=True, tracker="bytetrack.yaml")
    
    # Get the dimensions of the frame
    height, width, _ = frame.shape
    # Define scaling factors based on the frame dimensions
    scale_factor = min(width, height) / 640  # 640 is a reference resolution, adjust as needed
    box_thickness = int(3 * scale_factor)
    label_font_scale = 1 * scale_factor
    label_text_thickness = int(2 * scale_factor)
    count_text_thickness = int(5 * scale_factor)
    count_font_scale = 2 * scale_factor

    # Append the bounding box and track ID for annotation
    bbox_list = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for dt in boxes.data:
            x1, y1 = int(dt[0]), int(dt[1])
            x2, y2 = int(dt[2]), int(dt[3])
            track_id = int(dt[4])
            max_track_id = max(track_id, max_track_id) # keep track total tracking count
            # score = round(float(dt[5]), 2)
            bbox_list.append((x1, y1, x2, y2, track_id))
    
    # Draw bounding boxes and track IDs
    for (x1, y1, x2, y2, track_id) in bbox_list:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        cv2.putText(frame, f"{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 0, 0), label_text_thickness)

    # vehicle count and annotation
    cv2.putText(frame, f"Current count: {len(result)}", (int(10*scale_factor), int(60*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, count_font_scale, (0,255,0), count_text_thickness)
    cv2.putText(frame, f"Total count: {max_track_id}", (int(10*scale_factor), int(120*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, count_font_scale, (0,255,0), count_text_thickness)
    return frame

def process_video(video_path):
    """ Accept uploaded video input and invoke process_frame function to perform vehicle detection and annotation """
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame and draw bounding boxes
        frame = process_frame(frame)
        # Display the processed frame
        stframe.image(frame, channels='BGR')
    
    cap.release()

def process_webcam():
    """ Accept real-time webcam input and invoke process_frame function to perform vehicle detection and annotation """
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
       st.error("Failed to retrieve video from webcam")

    # set webcam input w*h = 1280*720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to retrieve video from webcam")
            break

        # Process the frame and draw bounding boxes
        frame = process_frame(frame)
        # Display the processed frame
        stframe.image(frame, channels='BGR')
    
    cap.release()

def main():
    st.title("Vehicle Tracking/Counting App")

    # Selection for video or webcam
    option = st.selectbox("Select Input Source", ("Upload Video", "Use Webcam"))

    if option == "Upload Video":
        video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if video_file is not None:
            # Save the uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video_path = tfile.name
            process_video(video_path)

    elif option == "Use Webcam":
        if st.button("Start Webcam"):
            process_webcam()

if __name__ == '__main__':
    main()
