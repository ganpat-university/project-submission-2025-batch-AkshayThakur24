from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO
import MySQLdb
from flask_cors import CORS
import os
import cv2
import numpy as np
import tensorflow as tf
import base64
from flask import Flask
from flask_mail import Mail, Message

mail = Mail()  # Don’t pass app yet

app = Flask(__name__)  # Now define app

# Configuration
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='manishchoksi77@gmail.com',
    MAIL_PASSWORD=''
)

mail.init_app(app) 

#app = Flask(__name__)
app.secret_key = "your_secret_key_here"
CORS(app)
socketio = SocketIO(app)

# Upload folder
# app.config['UPLOAD_FOLDER'] = '/static_for_image/uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ensure Flask knows where “static” lives (usually automatic)
app.static_folder = os.path.join(app.root_path, 'static')

# save uploads into static/uploads
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


model = tf.keras.models.load_model('./lifeguard_model.keras')



# --------- DATABASE CONNECTION ----------
def get_db_connection():
    return MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        passwd="",
        db="d"
    )

# --------- AUTH ROUTES ----------
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/signup', methods=['POST'])
# def signup():
#     data = request.json
#     name = data.get("name")
#     email = data.get("email")
#     password = data.get("password")

#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
#     if cursor.fetchone():
#         cursor.close()
#         conn.close()
#         return jsonify({"status": "fail", "message": "Email already registered"}), 409

#     password_hash = generate_password_hash(password)
#     cursor.execute("INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)", (name, email, password_hash))
#     conn.commit()
#     cursor.close()
#     conn.close()

#     return jsonify({"status": "success", "message": "Signup successful"})

import random
import string



# Function to generate a random OTP
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

# Update the signup route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    sex = data.get("sex")
    country = data.get("country")
    age = data.get("age")

    # Generate OTP
    otp = generate_otp()

    # Send OTP to email
    msg = Message('Your OTP for Signup', sender='akshay.bharat.thakur7@gmail.com', recipients=[email])
    msg.body = f'Your OTP is {otp}'
    mail.send(msg)

    # Save OTP in session to verify later
    session['otp'] = otp
    session['signup_data'] = {'name': name, 'email': email, 'password': password, 'sex': sex, 'country': country, 'age': age}

    return jsonify({"status": "success", "message": "OTP sent to your email. Please verify."})

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.json
    otp_received = data.get("otp")

    # Verify OTP
    if otp_received == session.get('otp'):
        # OTP is valid, create user
        signup_data = session.get('signup_data')
        name = signup_data['name']
        email = signup_data['email']
        password = signup_data['password']
        sex = signup_data['sex']
        country = signup_data['country']
        age = signup_data['age']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({"status": "fail", "message": "Email already registered"}), 409

        password_hash = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, sex, country, age)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (name, email, password_hash, sex, country, age))

        conn.commit()
        cursor.close()
        conn.close()

        # Clear OTP and signup data
        session.pop('otp', None)
        session.pop('signup_data', None)

        return jsonify({"status": "success", "message": "Signup successful"})
    else:
        return jsonify({"status": "fail", "message": "Invalid OTP"}), 400


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and check_password_hash(user[6], password):
        session['user_id'] = user[0]
        session['email'] = user[5]
        session['user_type'] = user[9]
        print("hiiiiiiiiii",session['user_type'])
        return jsonify({"status": "success", "message": "Login successful"})
    return jsonify({"status": "fail", "message": "Invalid credentials"}), 401

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# --------- DASHBOARD ACCESS ----------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html')

# --------- DROWNING DETECTION ----------
def predict_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224)) / 255.0
    prediction = model.predict(np.expand_dims(resized_frame, axis=0), verbose=0)
    return "Drowning" if np.argmax(prediction) == 0 else "Swimming"

import base64



def generate_frames(video_path, user_id):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # prediction
        label = predict_frame(frame)
        update_user_stats(user_id, label)

        # annotate & emit
        color = (0,0,255) if label=="Drowning" else (0,255,0)
        cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        _, buf = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        socketio.emit('frame_data', {'image': b64, 'label': label})

    cap.release()

    print(f"Processed photo for user_id: {user_id}, label: {label}")

def update_user_stats(user_id, label):
    if not user_id:
        return

    conn = get_db_connection()
    cur = conn.cursor()
    if label == "Drowning":
        cur.execute(
          "UPDATE users SET drowning_counter = drowning_counter + 1 WHERE id = %s",
          (user_id,)
        )
    else:
        cur.execute(
          "UPDATE users SET swimming_counter = swimming_counter + 1 WHERE id = %s",
          (user_id,)
        )
    conn.commit()
    cur.close()
    conn.close()

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    
    # Log the file details for debugging
    print(f"Filename: {file.filename}")
    print(f"MIME Type: {file.mimetype}")
     
    print(f"File Size: {len(file.read())} bytes")
    
    file.seek(0)  # Reset file pointer after reading size

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Check if the file is a valid video format
        if file.mimetype.startswith('video/') or file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            print("File is a valid video.")
            user_id = session.get('user_id')
            socketio.start_background_task(generate_frames, file_path, user_id)
            return jsonify({"message": "Video processing started", "file_type": "video"})
        
        # Check if the file is an image (assuming the accepted formats)
        elif file.mimetype.startswith('image/') or file.filename.lower().endswith(('.jpg', '.jpeg',)):
            print("File is a valid image.")
            
            # Perform image processing here if needed (e.g., predict "Drowning" or "Swimming")
            user_id = session.get('user_id')
            socketio.start_background_task(generate_frames, file_path, user_id)
            return jsonify({"message": "Image processing started", "file_type": "image"})

            # Here you can add the model prediction for image if needed
            # For example, if you have a model for image-based drowning detection:
            # image = cv2.imread(file_path)
            # label = predict_image(image)  # Your image prediction function

            return jsonify({"message": "Image uploaded successfully", "file_type": "image", "label": label, "image_url": file_path})

        else:
            print("Invalid file format.")
            return jsonify({"error": "Invalid file format"}), 400

    return jsonify({"error": "No file uploaded"}), 400



def webcam_feed():
    cap = cv2.VideoCapture(0)  # Capture from webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = predict_frame(frame)
        color = (0, 0, 255) if label == "Drowning" else (0, 255, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        socketio.emit('frame_data', {'image': frame_bytes.hex(), 'label': label})

@app.route('/start_webcam', methods=['GET'])
def start_webcam():
    socketio.start_background_task(webcam_feed)
    return jsonify({"message": "Webcam streaming started"})


@app.route('/admin-dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')


@app.route('/admin-stats')
def admin_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Gender counts
    cursor.execute("SELECT sex, COUNT(*) FROM users GROUP BY sex")
    gender_data = dict(cursor.fetchall())

    # Drowning & Swimming counters
    cursor.execute("SELECT SUM(drowning_counter), SUM(swimming_counter) FROM users")
    drown_count, swim_count = cursor.fetchone()

    # Country-wise user counts
    cursor.execute("SELECT country, COUNT(*) FROM users GROUP BY country")
    country_data = dict(cursor.fetchall())

    # Country-wise gender ratio
    cursor.execute("""
        SELECT country, sex, COUNT(*) 
        FROM users 
        WHERE sex IS NOT NULL AND country IS NOT NULL 
        GROUP BY country, sex
    """)
    rows = cursor.fetchall()

    # Transform to a nested dict: {country: {'Male': count, 'Female': count, ...}}
    country_gender_data = {}
    for country, sex, count in rows:
        if country not in country_gender_data:
            country_gender_data[country] = {}
        country_gender_data[country][sex] = count

    cursor.close()
    conn.close()

    return jsonify({
        'gender': gender_data,
        'drowning': drown_count or 0,
        'swimming': swim_count or 0,
        'country': country_data,
        'country_gender': country_gender_data
    })



# --------- MAIN ---------
if __name__ == '__main__':
    socketio.run(app, debug=True, port=5052)

