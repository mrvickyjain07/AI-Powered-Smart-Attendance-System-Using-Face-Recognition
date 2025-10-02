from flask import Flask,render_template,request, Response
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
import sqlite3
import json
import pandas as pd
import requests
from flask import session
import threading
import time

name="Vicky"
app = Flask(__name__)
app.secret_key = '4x7KiqWXK9$mYpV2'  

def init_db():
    conn = sqlite3.connect('information.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS Attendance
                    (NAME TEXT NOT NULL,
                     Time TEXT NOT NULL,
                     Date TEXT NOT NULL)''')
    
    try:
        with open('attendance.csv', 'r') as f:
            attendance_data = f.readlines()
            today = date.today()
            
            for line in attendance_data:
                if line.strip():  # Skip empty lines
                    name, time = line.strip().split(',')
                    conn.execute("INSERT OR IGNORE INTO Attendance (NAME, Time, Date) VALUES (?, ?, ?)",
                               (name.strip(), time.strip(), today))
        
        conn.commit()
        print("Existing attendance data imported successfully")
    except Exception as e:
        print(f"Error importing existing attendance: {str(e)}")
    finally:
        conn.close()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_IMAGES_PATH = os.path.join(BASE_DIR, 'Training images')

import os
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

# Cache for face encodings to avoid reprocessing
ENCODINGS_CACHE = {
    'last_updated': None,
    'encodings': [],
    'names': []
}

@app.route('/new', methods=['GET', 'POST'])
def new():
    if request.method=="POST":
        return render_template('index.html')
    else:
        return "Everything is okay!"

@app.route('/name', methods=['GET', 'POST'])
def name():
    if request.method == "POST":
        try:
            name1 = request.form['name1']
            name2 = request.form['name2']

            # Initialize camera with better resolution
            cam = cv2.VideoCapture(0)  # Use laptop camera
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if not cam.isOpened():
                return "Error: Could not open camera. Please make sure your webcam is connected."

            # Display camera window for user to see themselves
            print("Camera window opened. Press SPACE to capture or ESC to cancel.")
            cv2.namedWindow("Register Student")
            
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                    
                # Display instructions on the frame
                cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow("Register Student", frame)
                
                # Check for key presses
                k = cv2.waitKey(1)
                if k == 27:  # ESC key - cancel
                    cv2.destroyAllWindows()
                    cam.release()
                    return "Registration cancelled by user."
                elif k == 32:  # SPACE key - capture
                    break
            
            # At this point, frame contains the last captured image
            if 'frame' not in locals() or frame is None:
                cv2.destroyAllWindows()
                cam.release()
                return "Error: Failed to capture image."
                
            # Check for face in the captured image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(frame_rgb, model="hog", number_of_times_to_upsample=1)
            
            # Draw rectangle around the face for feedback
            for (top, right, bottom, left) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Show the captured frame with face rectangle
            cv2.imshow("Captured Image", frame)
            cv2.waitKey(1500)  # Show the captured image for 1.5 seconds
            
            # Clean up camera
            cv2.destroyAllWindows()
            cam.release()

            if len(faces) == 0:
                return "Error: No face detected. Please try again with proper lighting and positioning."

            if len(faces) > 1:
                return "Error: Multiple faces detected. Please ensure only one person is in the frame."

            # Save the captured image
            img_name = name1 + ".png"
            save_path = os.path.join(TRAINING_IMAGES_PATH, img_name)
            
            if cv2.imwrite(save_path, frame):
                return render_template('image.html', message=f"Registration successful! Image captured for {name1}")
            else:
                return "Error: Could not save the image. Please try again."

        except Exception as e:
            if 'cam' in locals():
                cam.release()
            cv2.destroyAllWindows()
            return f"Error: {str(e)}"
    else:
        return render_template('form.html')

@app.route("/",methods=["GET","POST"])
def recognize():
    if request.method=="POST":
        # Redirect to the camera stream link
        return render_template('stream.html')
    else:
        return render_template('main.html')

def findEncodings(images, names):
    """Generate face encodings for the given images and associate with names"""
    encodeList = []
    valid_names = []
    
    for img, name in zip(images, names):
        # Reduce image size for faster processing
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            valid_names.append(name)
        except IndexError:
            print(f"No face found in image for {name}")
            continue
        except Exception as e:
            print(f"Error processing image for {name}: {str(e)}")
            continue
    return encodeList, valid_names

def get_cached_encodings():
    """Get cached face encodings or refresh if needed"""
    global ENCODINGS_CACHE
    
    current_time = datetime.now()
    cache_age = None
    
    if ENCODINGS_CACHE['last_updated']:
        cache_age = (current_time - ENCODINGS_CACHE['last_updated']).total_seconds()
    
    # Refresh cache if it's older than 5 minutes or doesn't exist
    if cache_age is None or cache_age > 300 or not ENCODINGS_CACHE['encodings']:
        print("Refreshing face encodings cache...")
        path = 'Training images'
        images = []
        classNames = []
        
        if os.path.exists(path):
            myList = os.listdir(path)
            
            for cl in myList:
                img_path = os.path.join(path, cl)
                try:
                    curImg = cv2.imread(img_path)
                    if curImg is not None:
                        images.append(curImg)
                        classNames.append(os.path.splitext(cl)[0])
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
        
        if images:
            encodeListKnown, valid_names = findEncodings(images, classNames)
            
            ENCODINGS_CACHE['encodings'] = encodeListKnown
            ENCODINGS_CACHE['names'] = valid_names
            ENCODINGS_CACHE['last_updated'] = current_time
            
            print(f"Cache refreshed with {len(valid_names)} faces")
        else:
            print("No training images found or could be loaded")
    
    return ENCODINGS_CACHE['encodings'], ENCODINGS_CACHE['names']

@app.route('/login',methods = ['POST'])
def login():
    #print( request.headers )
    json_data = json.loads(request.data.decode())
    username = json_data['username']
    password = json_data['password']
    #print(username,password)
    df= pd.read_csv('cred.csv')
    if len(df.loc[df['username'] == username]['password'].values) > 0:
        if df.loc[df['username'] == username]['password'].values[0] == password:
            session['username'] = username
            return 'success'
        else:
            return 'failed'
    else:
        return 'failed'
        


@app.route('/checklogin')
def checklogin():
    #print('here')
    if 'username' in session:
        return session['username']
    return 'False'


@app.route('/how',methods=["GET","POST"])
def how():
    return render_template('form.html')
@app.route('/data',methods=["GET","POST"])
def data():
    '''user=request.form['username']
    pass1=request.form['pass']
    if user=="tech" and pass1=="tech@321" :
    '''
    if request.method=="POST":
        today=date.today()
        print(today)
        conn = sqlite3.connect('information.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        print ("Opened database successfully");
        cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance where Date=?",(today,))
        rows=cur.fetchall()
        print(rows)
        for line in cursor:

            data1=list(line)
        print ("Operation done successfully");
        conn.close()

        return render_template('form2.html',rows=rows)
    else:
        return render_template('form1.html')


            
@app.route('/whole',methods=["GET","POST"])
def whole():
    today=date.today()
    print(today)
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row 
    cur = conn.cursor() 
    print ("Opened database successfully");
    cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance")
    rows=cur.fetchall()    
    return render_template('form3.html',rows=rows)

@app.route('/dashboard',methods=["GET","POST"])
def dashboard():
    # Connect to database
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    today = date.today()
    
    # Get recent records
    cur.execute("SELECT NAME, Time, Date FROM Attendance ORDER BY Date DESC, Time DESC LIMIT 10")
    recent_records = cur.fetchall()
    
    # Get total number of students (distinct names in the database)
    cur.execute("SELECT COUNT(DISTINCT NAME) FROM Attendance")
    total_students = cur.fetchone()[0]
    
    # Get students present today
    cur.execute("SELECT COUNT(DISTINCT NAME) FROM Attendance WHERE Date=?", (today,))
    present_today = cur.fetchone()[0]
    
    # Get total records
    cur.execute("SELECT COUNT(*) FROM Attendance")
    total_records = cur.fetchone()[0]
    
    # Calculate attendance rate
    attendance_rate = 0
    if total_students > 0:
        attendance_rate = round((present_today / total_students) * 100)
    
    # Get attendance trend for last 7 days
    trend_dates = []
    trend_counts = []
    
    for i in range(6, -1, -1):
        # Calculate date for each of the last 7 days
        day = today - pd.Timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        trend_dates.append(day.strftime("%m-%d"))
        
        # Get count of students for that day
        cur.execute("SELECT COUNT(DISTINCT NAME) FROM Attendance WHERE Date=?", (day_str,))
        count = cur.fetchone()[0]
        trend_counts.append(count)
    
    # Get top attendees
    cur.execute("""
        SELECT NAME, COUNT(*) as count 
        FROM Attendance 
        GROUP BY NAME 
        ORDER BY count DESC 
        LIMIT 5
    """)
    top_attendees_data = cur.fetchall()
    top_attendees = [{'name': row['NAME'], 'count': row['count']} for row in top_attendees_data]
    
    # Get time distribution for today
    cur.execute("""
        SELECT substr(Time, 1, 2) as hour, COUNT(*) as count 
        FROM Attendance 
        WHERE Date=? 
        GROUP BY hour 
        ORDER BY hour
    """, (today,))
    time_dist_data = cur.fetchall()
    
    time_labels = []
    time_values = []
    
    for row in time_dist_data:
        time_labels.append(f"{row['hour']}:00")
        time_values.append(row['count'])
    
    # Get student comparison data (attendance frequency for top 5 students)
    student_names = [attendee['name'] for attendee in top_attendees]
    
    comparison_datasets = []
    if student_names:
        # Add some color variety for the radar chart
        colors = [
            'rgba(70, 130, 180, 0.7)',   # SteelBlue
            'rgba(60, 179, 113, 0.7)',   # MediumSeaGreen
            'rgba(255, 165, 0, 0.7)',    # Orange
            'rgba(106, 90, 205, 0.7)',   # SlateBlue
            'rgba(178, 34, 34, 0.7)'     # FireBrick
        ]
        
        for i, name in enumerate(student_names):
            if i < len(colors):
                # Get last 7 days attendance for this student
                attendance_counts = []
                for day in trend_dates:
                    day_date = datetime.strptime(day, "%m-%d").date()
                    day_date = day_date.replace(year=today.year)
                    cur.execute("SELECT COUNT(*) FROM Attendance WHERE NAME=? AND Date=?", 
                              (name, day_date))
                    count = cur.fetchone()[0]
                    attendance_counts.append(count)
                
                # Add dataset for this student
                comparison_datasets.append({
                    'label': name,
                    'data': attendance_counts,
                    'backgroundColor': colors[i].replace('0.7', '0.2'),
                    'borderColor': colors[i],
                    'pointBackgroundColor': colors[i],
                    'pointHoverBackgroundColor': '#fff',
                    'pointHoverBorderColor': colors[i]
                })
    
    # Close database connection
    conn.close()
    
    # Prepare data for charts
    attendance_trend = {
        'labels': trend_dates,
        'values': trend_counts
    }
    
    time_distribution = {
        'labels': time_labels,
        'values': time_values
    }
    
    comparison_data = {
        'labels': trend_dates,
        'datasets': comparison_datasets
    }
    
    # Statistics for the dashboard
    stats = {
        'total_students': total_students,
        'present_today': present_today,
        'attendance_rate': attendance_rate,
        'total_records': total_records
    }
    
    return render_template('dashboard.html', 
                           recent_records=recent_records,
                           top_attendees=top_attendees,
                           stats=stats,
                           attendance_trend=attendance_trend,
                           time_distribution=time_distribution,
                           comparison_data=comparison_data)

# Sending Email about the attendance report to the faculties/ parents / etc.
# Not working currently
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText

def sendMail():
    mssg=MIMEMultipart()


    server=smtplib.SMTP("smtp.gmail.com",'587')
    server.starttls()
    print("Connected with the server")
    user=input("Enter username:")
    pwd=input("Enter password:")
    server.login(user,pwd)
    print("Login Successful!")
    send=user
    rcv=input("Enter Receiver's Email id:")
    mssg["Subject"] = "Employee Report csv"
    mssg["From"] = user
    mssg["To"] = rcv

    body='''
        <html>
        <body>
         <h1>Employee Quarterly Report</h1>
         <h2>Contains the details of all the employees</h2>
         <p>Do not share confidential information with anyone.</p>
        </body>
        </html>
         '''

    body_part=MIMEText(body,'html')
    mssg.attach(body_part)

    with open("emp.csv",'rb') as f:
        mssg.attach(MIMEApplication(f.read(),Name="emp.csv"))

    server.sendmail(mssg["From"],mssg["To"],mssg.as_string())
   # server.quit()

def markData(name):
    try:
        now = datetime.now()
        dtString = now.strftime('%H:%M')
        today = date.today()
        
        conn = sqlite3.connect('information.db')
        conn.execute('''CREATE TABLE IF NOT EXISTS Attendance
                        (NAME TEXT NOT NULL,
                         Time TEXT NOT NULL,
                         Date TEXT NOT NULL)''')
        
        # Check if attendance already marked for today
        cursor = conn.execute("SELECT COUNT(*) FROM Attendance WHERE NAME=? AND Date=?", (name, today))
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.execute("INSERT INTO Attendance (NAME,Time,Date) VALUES (?,?,?)",
                        (name, dtString, today))
            conn.commit()
            print(f"Attendance marked for {name}")
        else:
            print(f"Attendance already marked for {name} today")
            
    except Exception as e:
        print(f"Error marking attendance: {str(e)}")
    finally:
        conn.close()

def markAttendance(name):
    with open('attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M')
            f.write(f'{name},{dtString}\n')

def generate_frames():
    # Load known faces
    path = 'Training images'
    images = []
    classNames = []
    myList = os.listdir(path)
    
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
    
    # Generate encodings for known faces
    encodeListKnown = findEncodings(images, classNames)
    print('Encoding Complete')
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    recognized_names = set()  # Track recognized names in this session
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process for face recognition
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = face_recognition.face_locations(imgS, model="hog")
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            
            name = "Unknown"
            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    
                    # Add name to recognized set for later attendance marking
                    recognized_names.add(name)
            
            # Draw face rectangle and name
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Save recognized names in session for attendance tracking
        session['recognized_names'] = list(recognized_names)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/recognize_and_mark')
def recognize_and_mark():
    recognized_names = set()  # Set to store all recognized people
    error_message = None
    
    # Get cached encodings
    try:
        encodeListKnown, classNames = get_cached_encodings()
        
        if not encodeListKnown:
            return render_template('error.html', message="No registered faces found. Please register students first.")
    
        # Try to connect to the MJPG stream
        stream_url = "http://192.168.106.210:8080/?action=stream"
        
        # Maximum number of frames to process
        frames_to_process = 5  # Reduced from 10 to 5 for speed
        connection_timeout = 5  # Reduced from 10 seconds to 5
        
        # Attempt to get frames from the stream
        try:
            response = requests.get(stream_url, stream=True, timeout=connection_timeout)
            
            if response.status_code == 200:
                bytes_data = bytes()
                frames_processed = 0
                start_time = time.time()
                max_processing_time = 8  # Maximum processing time in seconds
                
                # Process multiple frames
                for chunk in response.iter_content(chunk_size=4096):  # Increased chunk size for efficiency
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        
                        # Process the frame
                        try:
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                # Resize image for faster processing
                                imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                                
                                # Use HOG model which is faster
                                facesCurFrame = face_recognition.face_locations(imgS, model="hog")
                                
                                if facesCurFrame:
                                    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
                                    
                                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)
                                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                                        
                                        if len(faceDis) > 0:
                                            matchIndex = np.argmin(faceDis)
                                            if matches[matchIndex]:
                                                name = classNames[matchIndex].upper()
                                                recognized_names.add(name)
                        except Exception as frame_error:
                            print(f"Error processing frame: {str(frame_error)}")
                            continue
                        
                        frames_processed += 1
                        
                        # Stop if we've processed enough frames or found all faces
                        if frames_processed >= frames_to_process or len(recognized_names) >= len(classNames):
                            break
                            
                        # Check if we've spent too much time processing
                        if time.time() - start_time > max_processing_time:
                            print(f"Maximum processing time reached after {frames_processed} frames")
                            break
                        
                        # Shorter delay between frames
                        time.sleep(0.1)
                
                # If we didn't process any frames, try the webcam
                if frames_processed == 0:
                    raise Exception("Could not extract any frames from stream")
                    
        except Exception as e:
            print(f"Stream error: {str(e)}")
            error_message = f"Could not connect to the camera stream. Using webcam instead."
            
            # Fall back to webcam
            try:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    return render_template('error.html', message="Could not open any camera. Please check your camera connection.")
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                for _ in range(3):  # Only try 3 frames for speed
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Process for face recognition
                    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                    
                    facesCurFrame = face_recognition.face_locations(imgS, model="hog")
                    
                    if facesCurFrame:
                        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
                        
                        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)
                            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                            
                            if len(faceDis) > 0:
                                matchIndex = np.argmin(faceDis)
                                if matches[matchIndex]:
                                    name = classNames[matchIndex].upper()
                                    recognized_names.add(name)
                
                cap.release()
            except Exception as webcam_error:
                print(f"Webcam error: {str(webcam_error)}")
                return render_template('error.html', message="Could not connect to any camera. Please check your camera connections.")
        
        # Mark attendance for recognized people
        if recognized_names:
            for name in recognized_names:
                try:
                    markAttendance(name)
                    markData(name)
                except Exception as mark_error:
                    print(f"Error marking attendance for {name}: {str(mark_error)}")
            
            # Show count of marked attendance
            return render_template('success.html', 
                names=list(recognized_names), 
                count=len(recognized_names),
                warning=error_message)
        else:
            return render_template('error.html', message="No faces recognized. Please make sure students are clearly visible in the camera frame.")
    
    except Exception as general_error:
        print(f"General error: {str(general_error)}")
        return render_template('error.html', message=f"An unexpected error occurred: {str(general_error)}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    init_db()  # Initialize database before running the app
    app.run(debug=True)



