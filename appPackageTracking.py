from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from flask_cors import CORS
from pathlib import Path
import tempfile
import shutil
import json
import traceback
import uuid
import zipfile
from io import BytesIO
import os
import sys
import torch
from PIL import Image
import imageio.v3 as iio
import numpy as np
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
from deep_sort_realtime.deepsort_tracker import DeepSort




app = Flask(__name__)
CORS(app)


session_outputs = {}


def process_videos(video_folder_path):  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #could add sampling codition for cpu to speed processing
    person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
    image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge", device_map=device)
    
    
    output_results = {}  # Dictionary to store all video results
    
    for mp4_file in video_folder_path.glob("*.mp4"):
        video_name = mp4_file.stem
        video_pose_data = []
        frame_count = 0
        tracker = DeepSort(
            max_age=30,           # Keep tracks for 30 frames after disappearance
            n_init=3,             # Confirm track after 3 detections
            nms_max_overlap=0.3,  # Non-max suppression threshold
            max_cosine_distance=0.4,  # Feature matching threshold
            nn_budget=None,       # No limit on feature vectors stored
            override_track_class=None,
            embedder="mobilenet", # Use MobileNet for person re-identification
            half=True,           # Use half precision if available
            bgr=False,           # Input is RGB, not BGR
            embedder_gpu=True,   # Use GPU for embedder if available
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,       # Bounding boxes, not polygons
            today=None
        )
        for frame in iio.imiter(mp4_file, plugin="pyav"):
            #print('Did a frame')
            image=Image.fromarray(frame)
            inputs = person_image_processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = person_model(**inputs)

            results = person_image_processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
            )
            result = results[0]
            person_boxes = result["boxes"][result["labels"] == 0]
            person_boxes = person_boxes.cpu().numpy()
            
            if len(person_boxes) == 0:
                frame_data = {"frame": frame_count, "poses": []}
                video_pose_data.append(frame_data)
                frame_count += 1
                continue
            deepsort_detections = []
            confidences = result["scores"][result["labels"] == 0].cpu().numpy()
            
            for i, box in enumerate(person_boxes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                conf = confidences[i]
                
                # DeepSORT expects: ([x, y, w, h], confidence, detection_class)
                deepsort_detections.append(([x1, y1, w, h], conf, 'person'))
            tracks = tracker.update_tracks(deepsort_detections, frame=np.array(frame))
            tracked_boxes = []
            track_ids = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()  # Get [left, top, right, bottom]
                
                # Convert back to [x, y, w, h] for pose estimation
                x, y, w, h = ltrb[0], ltrb[1], ltrb[2] - ltrb[0], ltrb[3] - ltrb[1]
                tracked_boxes.append([x, y, w, h])
                track_ids.append(track_id)
            
            # Handle case where tracking lost all people
            if len(tracked_boxes) == 0:
                frame_data = {"frame": frame_count, "poses": []}
                video_pose_data.append(frame_data)
                frame_count += 1
                continue
            
            # Pose estimation on tracked people
            tracked_boxes = np.array(tracked_boxes)
            person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
            person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

            inputs = image_processor(image, boxes=[tracked_boxes], return_tensors="pt").to(device)
            dataset_index = torch.tensor([0], device=device)
            
            with torch.no_grad():
                outputs = model(**inputs, dataset_index=dataset_index)

            pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[tracked_boxes], threshold=0.3)
            image_pose_result = pose_results[0]
            
            frame_data = {"frame": frame_count, "poses": []}
            
            for i, person_pose in enumerate(image_pose_result):
                keypoints_array = person_pose["keypoints"].cpu().numpy().tolist()
                scores_array = person_pose["scores"].cpu().numpy().tolist()
                
                frame_data["poses"].append({
                    "person_id": track_ids[i],
                    "keypoints": keypoints_array,
                    "scores": scores_array
                })
            
            video_pose_data.append(frame_data)
            frame_count += 1
        
        
        output_results[f"{video_name}_poses.json"] = video_pose_data
    
    return output_results



def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

@app.route('/')
def serve_html():
    return send_file(resource_path('index.html'))


@app.route('/process_videos', methods=['POST'])
def handle_process_videos():
    temp_dir = None
    session_id = None
    
    try:
        print("Received video processing request...")
        
        
        session_id = str(uuid.uuid4())
                
        # Create temp directory for input videos
        temp_dir = Path(tempfile.mkdtemp())
        print(f"Created temp directory: {temp_dir}")
        
        # Get uploaded files
        uploaded_files = request.files.getlist('videos')
        print(f"Received {len(uploaded_files)} files")
        
        if not uploaded_files:
            return jsonify({"error": "No files uploaded"}), 400
        
        # Save videos to temp directory
        saved_files = []
        for video in uploaded_files:
            if video.filename and video.filename.endswith('.mp4'):
                filename = Path(video.filename).name
                save_path = temp_dir / filename
                
                print(f"Saving {video.filename} as {filename}")
                video.save(str(save_path))
                saved_files.append(filename)
        
        if not saved_files:
            return jsonify({"error": "No MP4 files found"}), 400
        
        print(f"Starting pose detection on {len(saved_files)} videos...")
        
        pose_results = process_videos(temp_dir)
        
        print("Pose detection completed successfully!")
        
        session_outputs[session_id] = {
            "results": pose_results,  
            "filenames": list(pose_results.keys())  
        }
        
        # Not sure if session clean up still needed
        if len(session_outputs) > 10:
            oldest_sessions = sorted(session_outputs.keys())[:len(session_outputs)-10]
            for old_session in oldest_sessions:
                del session_outputs[old_session]
                print(f"Cleaned up old session from memory: {old_session}")
        
        # Clean up temp input directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        response = make_response(jsonify({
            "processed_count": len(saved_files),
            "session_id": session_id,
            "files_processed": session_outputs[session_id]["filenames"]
        }))
        response.set_cookie('session_id', session_id, max_age=3600)  # 1 hour expiry
        
        return response
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        
        # Clean up on error
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        if session_id and session_id in session_outputs:
            del session_outputs[session_id]
        
        return jsonify({
            "error": f"Processing failed: {str(e)}"
        }), 500

#Download creates ZIP from memory, no disk access
@app.route('/download_results', methods=['GET'])
def download_results():
    try:
        # Get session from cookie
        session_id = request.cookies.get('session_id')
        
        if not session_id or session_id not in session_outputs:
            return jsonify({"error": "No results found"}), 404
        
        session_data = session_outputs[session_id]
        results = session_data["results"]
        
        
        memory_file = BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename, data in results.items():
                json_str = json.dumps(data, indent=2)
                zf.writestr(filename, json_str)
        
        memory_file.seek(0)
        
        # 🔴 OPTIONAL: Clean up session after download
        # del session_outputs[session_id]
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'pose_results_{session_id[:8]}.zip'
        )
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({"error": "Download failed"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Server will be available at: http://127.0.0.1:5001")
    app.run(debug=True, host='127.0.0.1', port=5001)