from flask import Flask, render_template, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp

app = Flask(__name__)
CORS(app)
camera = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
emotion_classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)

def generate_frames():
    a = 0
    b = 0
    c = time.time()
    face_3d = np.array([
        [0.0, 0.0, 0.0],         
        [-30.0, -30.0, -30.0],   
        [30.0, -30.0, -30.0],    
        [-30.0, 30.0, -30.0],    
        [30.0, 30.0, -30.0],     
        [0.0, -50.0, 0.0],       
    ], dtype=np.float64)

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray_resized]) != 0:
                roi = roi_gray_resized.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = emotion_classifier.predict(roi)[0]
                emotion_label = emotion_labels[prediction.argmax()]
                label_position = (x, y - 10)
                cv2.putText(frame, f"{emotion_label}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            eyes = eyeCascade.detectMultiScale(roi_gray)
            if (time.time() - c) >= 15:
                if a / (a + b) >= 0.2:
                    os.system('start buzz.mp3')
                    print("****ALERT*****", a, b, a / (a + b))
                else:
                    print("Safe", a / (a + b))
                c = time.time()
                a = 0
                b = 0

            if len(eyes) == 0:
                a += 1
                print('No eyes detected!')
            else:
                b += 1
                print('Eyes detected!')

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks_2d = np.array([
                        [face_landmarks.landmark[1].x * frame.shape[1], face_landmarks.landmark[1].y * frame.shape[0]],
                        [face_landmarks.landmark[33].x * frame.shape[1], face_landmarks.landmark[33].y * frame.shape[0]],
                        [face_landmarks.landmark[263].x * frame.shape[1], face_landmarks.landmark[263].y * frame.shape[0]],
                        [face_landmarks.landmark[133].x * frame.shape[1], face_landmarks.landmark[133].y * frame.shape[0]],
                        [face_landmarks.landmark[362].x * frame.shape[1], face_landmarks.landmark[362].y * frame.shape[0]],
                        [face_landmarks.landmark[152].x * frame.shape[1], face_landmarks.landmark[152].y * frame.shape[0]],
                    ], dtype=np.float64)

                    focal_length = frame.shape[1]
                    cam_matrix = np.array([[focal_length, 0, frame.shape[1] / 2],
                                           [0, focal_length, frame.shape[0] / 2],
                                           [0, 0, 1]])
                    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
                    success, rvec, tvec = cv2.solvePnP(face_3d, landmarks_2d, cam_matrix, dist_coeffs)

                    if success:
                        rmat, _ = cv2.Rodrigues(rvec)
                        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
                        singular = sy < 1e-6
                        if not singular:
                            x = np.arctan2(rmat[2, 1], rmat[2, 2])
                            y = np.arctan2(-rmat[2, 0], sy)
                            z = np.arctan2(rmat[1, 0], rmat[0, 0])
                        else:
                            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                            y = np.arctan2(-rmat[2, 0], sy)
                            z = 0
                        pitch = np.degrees(x)
                        yaw = np.degrees(y)
                        roll = np.degrees(z)
                        cv2.putText(frame, f"Yaw: {yaw:.2f} degrees", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if yaw < -10:
                            gaze_direction = "left"
                        elif yaw > 10:
                            gaze_direction = "right"
                        elif pitch > 10:
                            gaze_direction = "Up"
                        elif pitch < -10:
                            gaze_direction = "center"
                        else:
                            gaze_direction = "Unknown"
                        cv2.putText(frame, f"Gaze: {gaze_direction}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Could not determine gaze direction", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')

















# from flask import Flask, render_template, Response
# from flask_cors import CORS
# import cv2
# import numpy as np
# import os
# import time
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import mediapipe as mp

# app = Flask(__name__)
# CORS(app)
# camera = cv2.VideoCapture(0)

# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# eyeCascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
# emotion_classifier = load_model('model.h5')
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)

# def generate_frames():
#     a = 0
#     b = 0
#     c = time.time()
#     face_3d = np.array([
#         [0.0, 0.0, 0.0],         
#         [-30.0, -30.0, -30.0],   
#         [30.0, -30.0, -30.0],    
#         [-30.0, 30.0, -30.0],    
#         [30.0, 30.0, -30.0],     
#         [0.0, -50.0, 0.0],       
#     ], dtype=np.float64)

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frame[y:y+h, x:x+w]

#             roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#             if np.sum([roi_gray_resized]) != 0:
#                 roi = roi_gray_resized.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)
#                 prediction = emotion_classifier.predict(roi)[0]
#                 emotion_label = emotion_labels[prediction.argmax()]
#                 label_position = (x, y - 10)
#                 cv2.putText(frame, f"{emotion_label}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             else:
#                 cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#             eyes = eyeCascade.detectMultiScale(roi_gray)
#             if (time.time() - c) >= 15:
#                 if a / (a + b) >= 0.2:
#                     os.system('start buzz.mp3')
#                     print("****ALERT*****", a, b, a / (a + b))
#                 else:
#                     print("Safe", a / (a + b))
#                 c = time.time()
#                 a = 0
#                 b = 0

#             if len(eyes) == 0:
#                 a += 1
#                 print('No eyes detected!')
#             else:
#                 b += 1
#                 print('Eyes detected!')

#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

#             results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     landmarks_2d = np.array([
#                         [face_landmarks.landmark[1].x * frame.shape[1], face_landmarks.landmark[1].y * frame.shape[0]],
#                         [face_landmarks.landmark[33].x * frame.shape[1], face_landmarks.landmark[33].y * frame.shape[0]],
#                         [face_landmarks.landmark[263].x * frame.shape[1], face_landmarks.landmark[263].y * frame.shape[0]],
#                         [face_landmarks.landmark[133].x * frame.shape[1], face_landmarks.landmark[133].y * frame.shape[0]],
#                         [face_landmarks.landmark[362].x * frame.shape[1], face_landmarks.landmark[362].y * frame.shape[0]],
#                         [face_landmarks.landmark[152].x * frame.shape[1], face_landmarks.landmark[152].y * frame.shape[0]],
#                     ], dtype=np.float64)

#                     focal_length = frame.shape[1]
#                     cam_matrix = np.array([[focal_length, 0, frame.shape[1] / 2],
#                                            [0, focal_length, frame.shape[0] / 2],
#                                            [0, 0, 1]])
#                     dist_coeffs = np.zeros((4, 1), dtype=np.float64)
#                     success, rvec, tvec = cv2.solvePnP(face_3d, landmarks_2d, cam_matrix, dist_coeffs)

#                     if success:
#                         rmat, _ = cv2.Rodrigues(rvec)
#                         sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
#                         singular = sy < 1e-6
#                         if not singular:
#                             x = np.arctan2(rmat[2, 1], rmat[2, 2])
#                             y = np.arctan2(-rmat[2, 0], sy)
#                             z = np.arctan2(rmat[1, 0], rmat[0, 0])
#                         else:
#                             x = np.arctan2(-rmat[1, 2], rmat[1, 1])
#                             y = np.arctan2(-rmat[2, 0], sy)
#                             z = 0
#                         pitch = np.degrees(x)
#                         yaw = np.degrees(y)
#                         roll = np.degrees(z)
#                         cv2.putText(frame, f"Yaw: {yaw:.2f} degrees", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                         if yaw < -10:
#                             gaze_direction = "left"
#                         elif yaw > 10:
#                             gaze_direction = "right"
#                         elif pitch > 10:
#                             gaze_direction = "Up"
#                         elif pitch < -10:
#                             gaze_direction = "center"
#                         else:
#                             gaze_direction = "Unknown"
#                         cv2.putText(frame, f"Gaze: {gaze_direction}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                     else:
#                         cv2.putText(frame, "Could not determine gaze direction", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)