# tools/mediapipe_extract/mediapipe_process.py
"""
Usage:
  python mediapipe_process.py --input /path/to/video.mp4 --output /path/to/out.json

Output format:
{
  "frames": [
    {"t":0.0, "landmarks": {"nose":[x,y,z], "left_shoulder":[x,y,z], ...}},
    ...
  ],
  "meta": {...}
}
"""
import argparse, json, cv2, os, math
try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Install mediapipe: pip install mediapipe opencv-python")

mp_pose = mp.solutions.pose

def landmark_dict(lm, img_w, img_h):
    return [lm.x * img_w, lm.y * img_h, lm.z * max(img_w,img_h)]

def process_video(input_path, output_path, downscale=1, fps_sample=30):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = {"frames": [], "meta": {"source": os.path.basename(input_path), "fps": video_fps, "w":width, "h":height}}
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # sample rate control (only process up to fps_sample per sec)
        t_sec = frame_idx / video_fps
        # optional downscale
        if downscale != 1:
            frame = cv2.resize(frame, (int(width/downscale), int(height/downscale)))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # pick common names
            names = ["nose","left_eye","right_eye","left_ear","right_ear",
                     "left_shoulder","right_shoulder","left_elbow","right_elbow",
                     "left_wrist","right_wrist","left_hip","right_hip",
                     "left_knee","right_knee","left_ankle","right_ankle"]
            d = {}
            for i,name in enumerate(names):
                if i < len(lm):
                    d[name] = landmark_dict(lm[i], width, height)
            out["frames"].append({"t": round(t_sec,4), "frame": frame_idx, "landmarks": d})
        frame_idx += 1

    cap.release()
    with open(output_path, "w") as f:
        json.dump(out, f)
    print("Saved:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--downscale", type=int, default=1)
    args = parser.parse_args()
    process_video(args.input, args.output, downscale=args.downscale)
