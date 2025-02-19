from flask import Flask, request, jsonify, session,send_from_directory,send_file
from flask_pymongo import PyMongo
from gridfs import GridFS
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
# import os
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

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


uname=""
isLoggedin=False
# login_manager = LoginManager()
# login_manager.init_app(app)
#
#
# class User(UserMixin):
#   def __init__(self, user_id):
#     self.id = user_id


# @login_manager.user_loader
# def load_user(user_id):
#   return User(user_id)


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
  global uname
  global isLoggedin
  if request.method == 'POST':
    username = request.json.get('username')
    password = request.json.get('password')

    user = mongo.db.collection.find_one({'Email': username})

    if user and check_password_hash(user['Password'], password):
      # user_obj = User(user['Email'])
      # login_user(user_obj)
      # session['logged_in'] = True  # Set session variable for logged-in status
      uname= username
      print(uname)
      isLoggedin=True
      return jsonify({'message': 'Login successful','usen':uname,'logstatus':isLoggedin}), 200
    else:
      return jsonify({'message': 'Invalid username or password'}), 401


# @app.route('/logout')
# def logout():
  # logout_user()
#   session.pop('logged_in', None)  # Remove logged_in from the session
#   return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/logout',methods=['POST'] )
def logout():
  global uname
  global isLoggedin
  isLoggedin=False
  uname=""
  return jsonify({'message':'LoggedOut'}),200

@app.route('/get-photo/<photo_id>', methods=['GET'])
# @login_required
def get_photo(photo_id):
  try:
    photo = fs.get(ObjectId(photo_id))
    return photo.read(), 200, {'Content-Type': 'image/jpeg'}
  except Exception as e:
    return jsonify({'error': str(e)}), 404


# @app.route('/upload-video', methods=['POST'])
# # @login_required
# def upload_video():
#   if 'video' not in request.files:
#     return jsonify({'error': 'No video part in the request'}), 400
#
#   user_id = current_user.id  # Get the ID of the logged-in user
#
#   user = mongo.db.collection.find_one({'Email': user_id})
#   if not user:
#     return jsonify({'error': 'User not found'}), 404
#
#   video = request.files['video']
#   video_id = fs.put(video, filename=video.filename)
#
#   return jsonify({'message': 'Video uploaded successfully', 'video_id': str(video_id)}), 200


@app.route('/upload-data', methods=['POST'])
# @login_required
def upload_data():
  import os
  # if 'video' not in request.files:
  #   return jsonify({'error': 'No video part in the request'}), 400
  #
  # uploaded_file = request.files['video']
  # a=uploaded_file.filename
  # upload_folder = 'D:\Projects\Backend'  # Replace this with your desired folder path
  # uploaded_file.save(os.path.join(upload_folder, uploaded_file.filename))
  if 'video' not in request.files or 'coordinate' not in request.files and not isLoggedin:
    return jsonify({'error': 'No video or coordinate part in the request'}), 400

  video_file = request.files['video']
  a=video_file.filename
  coordinate_file = request.files['coordinate']

  if video_file.filename == '' or coordinate_file.filename == '':
    return jsonify({'error': 'No selected file'}), 400

  # Replace these paths with your desired folder paths
  upload_video_folder = 'D:/Projects/Backend'
  upload_coordinate_folder = 'D:/Projects/Backend'

  video_file.save(os.path.join(upload_video_folder, video_file.filename))
  coordinate_file.save(os.path.join(upload_coordinate_folder, coordinate_file.filename))

  import matplotlib.pyplot as plt
  import csv
  from matplotlib.animation import FuncAnimation

  file_name = coordinate_file.filename
  l1 = []
  with open(file_name, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
      l2 = []
      for i in row:
        l2.append(float(i))
      l1.append(l2)
  shape_coordinates_js = [
    (17.397211, 78.490262),
    (17.397217, 78.490281),
    (17.397221, 78.490298),
    (17.397220, 78.490320),
    (17.397219, 78.490342),
    (17.397211, 78.490362),
    (17.397202, 78.490381),
    (17.397188, 78.490393),
    (17.397175, 78.490405),
    (17.397137, 78.490410),
    (17.397121, 78.490403),
    (17.397107, 78.490396),
    (17.397092, 78.490381),
    (17.397081, 78.490367),
    (17.397075, 78.490327),
    (17.397070, 78.490295),
    (17.397067, 78.490270),
    (17.397065, 78.490251),
    (17.397061, 78.490232),
    (17.397052, 78.490221),
    (17.397041, 78.490210),
    (17.397023, 78.490202),
    (17.397010, 78.490199),
    (17.396990, 78.490199),
    (17.396976, 78.490198),
    (17.396961, 78.490210),
    (17.396947, 78.490219),
    (17.396936, 78.490231),
    (17.396930, 78.490245),
    (17.396923, 78.490262),
    (17.396917, 78.490280),
    (17.396916, 78.490302),
    (17.396917, 78.490321),
    (17.396922, 78.490338),
    (17.396944, 78.490357),
    (17.396955, 78.490369),
    (17.396967, 78.490372),
    (17.396979, 78.490376),
    (17.396995, 78.490373),
    (17.397009, 78.490372),
    (17.397023, 78.490364),
    (17.397034, 78.490357),
    (17.397044, 78.490346),
    (17.397051, 78.490339),
    (17.397063, 78.490320),
    (17.397074, 78.490301),
    (17.397082, 78.490286),
    (17.397089, 78.490274),
    (17.397097, 78.490260),
    # (17.397165, 78.490245),
    (17.397120, 78.490238),
    (17.397135, 78.490228),
    (17.397149, 78.490229),
    (17.397163, 78.490228),
    (17.397177, 78.490235),
    (17.397191, 78.490239),
    (17.397200, 78.490254),
    (17.397209, 78.490261),
    (17.397215, 78.490277)
  ]

  fig, ax = plt.subplots()
  latitudes, longitudes = zip(*l1)
  latitudes1, longitudes1 = zip(*shape_coordinates_js)
  ax.plot(longitudes1, latitudes1, marker='o', linestyle='-', linewidth=25, color='blue')

  # Plot the initial state of the points
  points, = ax.plot([], [], marker='o', linestyle='-', color='r')

  # Set the axis limits
  ax.set_xlim(min(longitudes1), max(longitudes1))
  ax.set_ylim(min(latitudes1), max(latitudes1))

  # Animation function
  def update(frame):
    points.set_data(longitudes[:frame], latitudes[:frame])
    return points,

  # Create the animation
  animation = FuncAnimation(fig, update, frames=len(latitudes), interval=100, blit=True)
  animation.save('coordinates_animation2.gif')

  # Show the animation
  plt.title('Coordinates Animation')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')

  # latitudes1, longitudes1 = zip(*shape_coordinates_js)
  # latitudes, longitudes = zip(*l1)
  # plt.plot(longitudes1, latitudes1, marker='o', linestyle='-', linewidth=25, color='blue')
  # plt.plot(longitudes, latitudes, marker='o', linestyle='-', color='red')
  # plt.title('Coordinates Plot')
  # plt.xlabel('Longitude')
  # plt.ylabel('Latitude')
  # plt.grid(True)
  # plt.savefig('Result_plot')



  from ultralytics import YOLO
  from supervision.video.dataclasses import VideoInfo
  from supervision.video.source import get_video_frames_generator
  from supervision.notebook.utils import show_frame_in_notebook
  from supervision.tools.detections import Detections, BoxAnnotator
  from supervision.draw.color import ColorPalette
  from supervision.video.sink import VideoSink
  import math
  import supervision
  from tqdm import tqdm
  from scipy.spatial import distance as dist
  import numpy as np
  import sys
  import cv2
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  SOURCE_VIDEO_PATH = a
  print("supervision._version:", supervision.__version__)
  model1 = "yolov8x.pt"
  model = YOLO(model1)
  model.fuse()
  VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
  CLASS_NAMES_DICT = model.model.names
  print(model.model.names)
  TARGET_VIDEO_PATH = "./lane_frame_detect.mp4"
  l1 = []
  generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
  video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
  max1 = sys.maxsize
  max2 = sys.maxsize
  with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(generator, total=video_info.total_frames):
      results = model(frame)[0]
      detections = Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int),
      )
      new_detections = []
      for _, confidence, class_id, tracker_id in detections:
        # print(_,confidence,class_id,tracker_id)
        if (class_id == 2 or class_id == 5 or class_id == 7):
          l1 = []
          l1.append(_)
          new_detections.append(l1)
          break
      for i in new_detections:
        for j in i:
          # print(j)
          x1 = int(j[0])
          y1 = int(j[1])
          x3 = int(j[2])
          y3 = int(j[3])
          # print(x1,y1,x3,y3)
          roi_vertices = [
            ((int((x1 + x3) / 2) - 500), y3 + 150),  # Bottom-left
            ((int((x1 + x3) / 2) - 500), y1),  # Top-left
            (int((x1 + x3) / 2), y1),  # Top-right
            (int((x1 + x3) / 2), (y3 + 150)),  # Bottom-right
          ]
          roi_vertices1 = [
            ((int((x1 + x3) / 2) + 500), y3 + 150),  # Bottom-right
            ((int((x1 + x3) / 2) + 500), y1),  # Top-right
            (int((x1 + x3) / 2), y1),  # Top-left
            (int((x1 + x3) / 2), (y3 + 150)),  # Bottom-left
          ]
          # cv.circle(frame, (a, y3-70), 10, (255, 0, 255), -1)  # -1 means fill the circle
          # cv.rectangle(frame,((int((x1+x3)/2)-500),y1),(int((x1+x3)/2),(y3+150)), (255, 255, 0), 5)
          cv2.rectangle(frame, (x1, y1), (x3, y3), (0, 255, 0), 4)
          # Create an empty mask with the same dimensions as the image
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
            if ((x > (((x1 + x3) / 2) - 500) and x < x1 and y < (y3 + 150) and y > y1) and (
                    x2 > (((x1 + x3) / 2) - 500) and x2 < x1 and y2 < (y3 + 150) and y2 > y1)):
              a1, b1 = (x + x2) / 2, (y + y2) / 2
              a2, b2 = (x1 + x3) / 2, (y1 + y3) / 2
              p1 = (a1, b1)
              p2 = (a2, b2)
              if (math.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2) < 325):
                print(math.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2), "left")
                cv2.line(frame, (x, y), (x2, y2), (255, 0, 0), 5)
                sink.write_frame(frame)
          for line in lines1:
            x, y, x2, y2 = line[0]
            if ((x < (((x1 + x3) / 2) + 500) and x > x3 and y < (y3 + 150) and y > y3) and (
                    x2 < (((x1 + x3) / 2) + 500) and x2 > x3 and y2 < (y3 + 150) and y2 > y1)):
              a1, b1 = (x + x2) / 2, (y + y2) / 2
              a2, b2 = (x1 + x3) / 2, (y1 + y3) / 2
              p1 = (a1, b1)
              p2 = (a2, b2)
              if (math.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2) < 325):
                cv2.line(frame, (x, y), (x2, y2), (255, 0, 0), 5)
                sink.write_frame(frame)

  from moviepy.editor import VideoFileClip

  def convert_to_h264(input_file, output_file):
    video_clip = VideoFileClip(input_file)
    video_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

  # Replace 'input_file.mp4' and 'output_file.mp4' with your file names
  input_file_path = 'lane_frame_detect.mp4'
  output_file_path = 'output_file.mp4'

  convert_to_h264(input_file_path, output_file_path)

  return "completed"
# print("completed")

# @app.route('/getresultvideo/<path:filename>', methods=['GET'])
# # def get_video(filename):
#   # directory_path = r'D:\Projects\Backend'
#   # # return send_from_directory(directory_path, filename)
#   # # return send_from_directory(directory_path, filename, mimetype='video/mp4'0)
#   # return
# def get_video(filename):
#   video_path = r'D:\Projects\Backend\lane_frame_detect.mp4'  # Replace with your video file path
#   # return send_file(video_path, as_attachment=False)
#   return send_file(video_path, as_attachment=True, mimetype='video/mp4')


@app.route('/getresultvideo/<string:file_type>', methods=['GET'])
def get_result(file_type):
    video_path = r'D:\Projects\Backend\output_file.mp4'
    image_path = r'D:\Projects\Backend\coordinates_animation2.gif'

    if file_type == 'video':
        return send_file(video_path, as_attachment=True, mimetype='video/mp4')
    elif file_type == 'image':
        return send_file(image_path, as_attachment=True, mimetype='image/gif')
    else:
        return "File not found", 404

@app.route('/fv',methods=['POST'])
def facialValidation():
  import numpy as np
  import face_recognition, os, cv2
  from tqdm import tqdm

  images = []
  classNames = []
  path = 'faces'

  # Function for Find the encoded data of the input image
  # Reading the training images and classes and storing into the corresponding lists
  for img in os.listdir(path):
    image = cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])

  print(classNames)

  def encodeImages(images):
    encodeList = []
    for i in tqdm(range(len(images)), desc="Encoding ", ascii=False, ncols=50, colour='green', unit=' Files'):
      img = images[i]
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      encode = face_recognition.face_encodings(img)[0]
      # print(encode)
      encodeList.append(encode)
    faceData = open('faces.dat', 'wb')
    np.save(faceData, encodeList)
    faceData = np.load('faces.dat')
    # print(faceData)
    return encodeList

  encodeImages(images)
  print("Encoding Completed")

  # Define the path for training images for OpenCV face recognition Project

  scale = 0.25
  box_multiplier = 1 / scale

  # Define a videocapture object
  cap = cv2.VideoCapture(0)

  # Images and names
  classNames = []
  path = 'faces'

  # Function for Find the encoded data of the input image
  # Reading the training images and classes and storing into the corresponding lists
  for img in os.listdir(path):
    classNames.append(os.path.splitext(img)[0])

  # Find encodings of training images

  encodes = open('faces.dat', 'rb')
  knownEncodes = np.load(encodes)
  print('Encodings Loaded Successfully')
  consecutive_count = 0
  max_consecutive_count = 5  # Adjust this value as needed
  previous_name = None

  while True:
    success, img = cap.read()  # Reading Each frame

    # Resize the frame
    Current_image = cv2.resize(img, (0, 0), None, scale, scale)
    Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    # Find the face location and encodings for the current frame

    face_locations = face_recognition.face_locations(Current_image, model='cnn')
    face_encodes = face_recognition.face_encodings(Current_image, face_locations)
    for encodeFace, faceLocation in zip(face_encodes, face_locations):
      matches = face_recognition.compare_faces(knownEncodes, encodeFace, tolerance=0.5)
      # matches = face_recognition.compare_faces(knownEncodes,encodeFace)
      faceDis = face_recognition.face_distance(knownEncodes, encodeFace)
      matchIndex = np.argmin(faceDis)

      # If match found then get the class name for the corresponding match

      if matches[matchIndex]:
        name = classNames[matchIndex].upper()
      else:
        name = 'Unknown'
      print(name)
      if name == previous_name:
        consecutive_count += 1
        if consecutive_count >= max_consecutive_count:
          print(f"Detected {max_consecutive_count} consecutive frames with the same face. Stopping.")
          break
      else:
        consecutive_count = 0

      previous_name = name

      y1, x2, y2, x1 = faceLocation
      y1, x2, y2, x1 = int(y1 * box_multiplier), int(x2 * box_multiplier), int(y2 * box_multiplier), int(
        x1 * box_multiplier)

      # Draw rectangle around detected face

      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
      cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("window_name", img)
    # sleep(5000)
    if cv2.waitKey(1) & 0xFF == ord('q') or consecutive_count >= max_consecutive_count:
      break
  # closing all open windows

  cap.release()
  cv2.destroyAllWindows()
  nm=name.lower()
  print(nm)
  print(uname)
  if(uname==nm):
    return jsonify({'Verification':'True'});
  return jsonify({'Verification':'False'});

if __name__ == '__main__':
  app.run(debug=True)