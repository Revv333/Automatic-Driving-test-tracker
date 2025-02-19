from flask import Flask, request, jsonify, session
from flask_pymongo import PyMongo
from gridfs import GridFS
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS   
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


# from ultralytics import YOLO
# from supervision.video.dataclasses import VideoInfo
# from supervision.video.source import get_video_frames_generator
# from supervision.notebook.utils import show_frame_in_notebook
# from supervision.tools.detections import Detections, BoxAnnotator
# from supervision.draw.color import ColorPalette
# from supervision.video.sink import VideoSink
# import math
# import supervision
# from tqdm.notebook import tqdm
# from scipy.spatial import distance as dist
# import numpy as np
# import sys
# import cv2
# import os



app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/project"
app.secret_key = 'DATABASEKEY_SECRET'
mongo = PyMongo(app)
fs = GridFS(mongo.db)

login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)



@app.route('/submit-form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        try:
            email = request.form['Email']
            password = request.form['Password']
            date_of_birth = request.form['DateOfBirth']
            aadhar_number = request.form['AadharNumber']
            
            photo = request.files['Photo']
            photo_id = fs.put(photo, filename=photo.filename)

            if not email or not password:
                return jsonify({'error': 'Please provide both email and password'}), 400

            existing_user = mongo.db.collection.find_one({'Email': email})
            if existing_user:
                return jsonify({'error': 'User already exists'}), 409  # Using HTTP status code 409 for conflict
                
            hashed_password = generate_password_hash(password)

            # Save other form data to MongoDB
            mongo.db.collection.insert_one({
                'Email': email,
                'Password': hashed_password,
                'DateOfBirth': date_of_birth,
                'AadharNumber': aadhar_number,
                'PhotoId': photo_id,  # Storing GridFS file ID
            })

            return jsonify({'message': 'User registered successfully'}), 201  # Using HTTP status code 201 for created

        except KeyError as e:
            return jsonify({'error': f'Missing key in request data: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Internal Server Error for other exceptions




@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.json.get('username')
        password = request.json.get('password')

        user = mongo.db.collection.find_one({'Email': username})

        if user and check_password_hash(user['Password'], password):
            user_obj = User(user['Email'])
            login_user(user_obj)
            session['logged_in'] = True  # Set session variable for logged-in status
            return jsonify({'message': 'Login successful'}), 200
        else:
            return jsonify({'message': 'Invalid username or password'}), 401



@app.route('/logout')
def logout():
    logout_user()
    session.pop('logged_in', None)  # Remove logged_in from the session
    return jsonify({'message': 'Logged out successfully'}), 200



@app.route('/get-photo/<photo_id>', methods=['GET'])
# @login_required
def get_photo(photo_id):
    try:
        photo = fs.get(ObjectId(photo_id))
        return photo.read(), 200, {'Content-Type': 'image/jpeg'}
    except Exception as e:
        return jsonify({'error': str(e)}), 404



@app.route('/upload-video', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400

    user_id = current_user.id  # Get the ID of the logged-in user

    user = mongo.db.collection.find_one({'Email': user_id})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    video = request.files['video']
    video_id = fs.put(video, filename=video.filename)

    return jsonify({'message': 'Video uploaded successfully', 'video_id': str(video_id)}), 200
    

@app.route('/upload-video', methods=['POST'])
# @login_required
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400
    
    import ultralytics
    import supervision
    from ultralytics import YOLO
    import pickle
    with open('D:\Projects\project-3-1\backend\your_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    from supervision.video.dataclasses import VideoInfo
    from supervision.video.source import get_video_frames_generator
    from supervision.notebook.utils import show_frame_in_notebook
    from supervision.tools.detections import Detections, BoxAnnotator
    from supervision.draw.color import ColorPalette
    from supervision.video.sink import VideoSink
    import math
    from tqdm import tqdm
    from scipy.spatial import distance as dist
    import numpy as np
    import sys
    import cv2



    video = request.files['video']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model1="yolov8x.pt"
    model=YOLO(model1)
    model.fuse()
    TARGET_VIDEO_PATH='./lane_frame_detect.mp4'
    l1=[]
    generator=get_video_frames_generator(video)
    video_info=VideoInfo.from_video_path(video)
    max1=sys.maxsize
    max2=sys.maxsize
    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame in tqdm(generator, total=video_info.total_frames):
            results=loaded_model(frame)[0]
            detections=Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype(int),
            )
            new_detections=[]
            for _,confidence,class_id,tracker_id in detections:
            # print(_,confidence,class_id,tracker_id)
                if(class_id==2 or class_id ==5 or class_id==7):
                    l1=[]
                    l1.append(_)
                    new_detections.append(l1)
                    break
            for i in new_detections:
                for j in i:
                    # print(j)
                    x1=int(j[0])
                    y1=int(j[1])
                    x3=int(j[2])
                    y3=int(j[3])
                    # print(x1,y1,x3,y3)
                    roi_vertices = [
                        ((int((x1+x3)/2)-500),y3+150),  # Bottom-left
                        ((int((x1+x3)/2)-500),y1),  # Top-left
                        (int((x1+x3)/2),y1),  # Top-right
                        (int((x1+x3)/2),(y3+150)),  # Bottom-right
                    ]
                    roi_vertices1 = [
                        ((int((x1+x3)/2)+500),y3+150),  # Bottom-right
                        ((int((x1+x3)/2)+500),y1),  # Top-right
                        (int((x1+x3)/2),y1),  # Top-left
                        (int((x1+x3)/2),(y3+150)),  # Bottom-left
                    ]
                    cv2.rectangle(frame,(x1,y1),(x3,y3), (0, 255, 0), 4)
                    mask = np.zeros_like(frame)

                    # Fill the ROI polygon with white color (255, 255, 255)
                    cv2.fillPoly(mask, [np.array(roi_vertices)], (255, 255, 255))
                    cv2.fillPoly(mask, [np.array(roi_vertices1)], (255, 255, 255))
                    # Bitwise AND the original image and the mask to isolate the ROI
                    roi_image = cv2.bitwise_and(frame, mask)
                    roi_image1 = cv2.bitwise_and(frame, mask)
                    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                    gray1 = cv2.cvtColor(roi_image1, cv2.COLOR_BGR2GRAY)
                    # Apply Gaussian blur to reduce noise and enhance edges
                    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                    blurred1 = cv2.GaussianBlur(gray1, (7, 7), 0)
                    # Perform edge detection using Canny
                    edges = cv2.Canny(blurred, 100, 150)
                    edges1 = cv2.Canny(blurred1, 100, 150)
                    # Find lines in the image using Hough Line Transform
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
                    lines1 = cv2.HoughLinesP(edges1, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
                    for line in lines:
                        x, y, x2, y2 = line[0]
                        if((x>(((x1+x3)/2)-500) and x<x1 and y<(y3+150) and y>y1) and (x2>(((x1+x3)/2)-500) and x2<x1 and y2<(y3+150) and y2>y1)):
                            a1,b1=(x+x2)/2,(y+y2)/2
                            a2,b2=(x1+x3)/2,(y1+y3)/2
                            p1=(a1,b1)
                            p2=(a2,b2)
                            if(math.sqrt((a2-a1)**2+(b2-b1)**2)<325):
                                print(math.sqrt((a2-a1)**2+(b2-b1)**2),"left")
                                cv2.line(frame, (x, y), (x2, y2), (255, 0, 0), 5)
                                sink.write_frame(frame)
                    for line in lines1:
                        x, y, x2, y2 = line[0]
                        if((x<(((x1+x3)/2)+500) and x>x3 and y<(y3+150) and y>y3) and (x2<(((x1+x3)/2)+500) and x2>x3 and y2<(y3+150) and y2>y1)):
                            a1,b1=(x+x2)/2,(y+y2)/2
                            a2,b2=(x1+x3)/2,(y1+y3)/2
                            p1=(a1,b1)
                            p2=(a2,b2)
                            if(math.sqrt((a2-a1)**2+(b2-b1)**2)<325):
                                cv2.line(frame, (x, y), (x2, y2), (255, 0, 0), 5)
                                sink.write_frame(frame)


if __name__ == '__main__':
    app.run(debug=True)