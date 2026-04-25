from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import tensorflow as tf
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-medora-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medora.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'localhost')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 1025))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'False').lower() in ['true', '1', 't', 'y', 'yes']
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False').lower() in ['true', '1', 't', 'y', 'yes']
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME', 'noreply@medora.com')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
mail = Mail(app)

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False) # 'patient' or 'hospital'
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    zipcode = db.Column(db.String(20), nullable=False)
    working_hours_start = db.Column(db.String(10), default='09:00')
    working_hours_end = db.Column(db.String(10), default='17:00')
    is_approved = db.Column(db.Boolean, default=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    hospital_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    predicted_disease = db.Column(db.String(100), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='Pending') # Pending, Accepted, Rejected
    appointment_date = db.Column(db.Date, nullable=True)
    appointment_time = db.Column(db.Time, nullable=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('User', foreign_keys=[patient_id])
    hospital = db.relationship('User', foreign_keys=[hospital_id])

class SearchActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.Text, nullable=False)
    predicted_disease = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define custom AttentionLayer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(input_shape[-1],),
                                 initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=[2, 0]) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=[2, 0]), axis=1)
        context_vector = tf.reduce_sum(attention_weights[..., tf.newaxis] * x, axis=1)
        return context_vector

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load the ML Model objects
model = load_model("disease_prediction_model.keras", custom_objects={'AttentionLayer': AttentionLayer})
with open("preprocessing.pkl", "rb") as f:
    preprocessing = pickle.load(f)

tokenizer = preprocessing["tokenizer"]
label_encoder = preprocessing["label_encoder"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_disease(symptoms):
    symptoms_cleaned = clean_text(symptoms)
    seq = tokenizer.texts_to_sequences([symptoms_cleaned])
    
    # Robust Gibberish/Nonsense Protection
    core_symptoms = [
        'pain', 'ache', 'fever', 'cough', 'cold', 'sneeze', 'blood', 
        'swell', 'vomit', 'nausea', 'dizzy', 'fatigue', 'weak', 
        'rash', 'itch', 'breath', 'sweat', 'chill', 'diarrhea', 
        'constipat', 'vision', 'hearing', 'weight', 'appetite', 
        'muscle', 'joint', 'skin', 'bleed', 'burn', 'numb', 
        'tingl', 'swallow', 'urin', 'bowel', 'sleep', 'hair', 
        'nail', 'eye', 'ear', 'nose', 'mouth', 'throat', 'chest', 
        'stomach', 'back', 'neck', 'head', 'face', 'leg', 'arm', 
        'hand', 'foot', 'toe', 'finger', 'sick', 'ill', 'lump', 
        'bump', 'bruise', 'cramp', 'spasm', 'stiff', 'sore', 
        'tender', 'inflam', 'red', 'pale', 'yellow', 'blur', 
        'blind', 'deaf', 'ring', 'taste', 'smell', 'shiver', 
        'shake', 'tremor', 'faint', 'seizure', 'convul', 
        'paraly', 'tired', 'exhaust', 'short', 'difficul', 
        'heart', 'palpitat', 'beat', 'pulse', 'pressure', 
        'asthma', 'allergy', 'infect', 'virus', 'bacteria', 
        'fung', 'parasit', 'injur', 'wound', 'cut', 'scrape', 
        'bite', 'sting', 'poi', 'drug', 'med', 'pill', 'doctor', 
        'hosp', 'clinic', 'emerg', 'urg', 'surger', 'operat', 
        'treat', 'cur', 'heal', 'remed', 'therap', 'diagnos', 
        'test', 'exam', 'scan', 'ray', 'mri', 'ct', 'ultrasound', 
        'bloodwork', 'lab', 'result', 'report', 'prescript', 
        'pharm', 'phlegm', 'mucus', 'snot', 'booger', 'phlegmy'
    ]
    
    words = symptoms_cleaned.split()
    has_symptom = any(sym in word for word in words for sym in core_symptoms)
    
    if not has_symptom or not seq or not seq[0]:
        return "Unknown", 0.0

    pad = pad_sequences(seq, maxlen=150)
    pred = model.predict(pad, verbose=0)[0]
    predicted_index = np.argmax(pred)
    confidence = float(pred[predicted_index] * 100) # Convert to standard Python float

    if confidence < 15.0:
        return "Unknown", confidence

    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    # Log search activity
    try:
        log = SearchActivity(symptoms=symptoms_cleaned, predicted_disease=predicted_label)
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        db.session.rollback()

    return predicted_label, confidence

import urllib.parse
@app.context_processor
def utility_processor():
    def get_gcal_url(appointment):
        if not appointment.appointment_date or not appointment.appointment_time:
            return "#"
        dt_start = datetime.combine(appointment.appointment_date, appointment.appointment_time)
        from datetime import timedelta
        dt_end = dt_start + timedelta(hours=1)
        fmt_start = dt_start.strftime("%Y%m%dT%H%M%S")
        fmt_end = dt_end.strftime("%Y%m%dT%H%M%S")
        title = f"Medical Appointment: {appointment.predicted_disease}"
        details = f"Symptoms: {appointment.symptoms}"
        location = appointment.hospital.name
        url = f"https://calendar.google.com/calendar/render?action=TEMPLATE&text={urllib.parse.quote(title)}&dates={fmt_start}/{fmt_end}&details={urllib.parse.quote(details)}&location={urllib.parse.quote(location)}"
        return url
    return dict(get_gcal_url=get_gcal_url)

# Routes
@app.route('/')
@login_required
def home():
    if current_user.role == 'hospital':
        return redirect(url_for('hospital_dashboard'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '')
    prediction, confidence = predict_disease(symptoms)
    
    if prediction == "Unknown":
        flash("Unrecognized input. Please enter valid symptoms.", "danger")
        return render_template('index.html', symptoms=symptoms)
        
    return render_template('index.html', prediction=prediction, confidence=confidence, symptoms=symptoms)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        role = request.form.get('role')
        name = request.form.get('name')
        email = request.form.get('email')
        zipcode = request.form.get('zipcode')
        password = request.form.get('password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        if not email.endswith('@gmail.com'):
            flash('Only @gmail.com email addresses are allowed for now.', 'danger')
            return redirect(url_for('register'))
            
        if not (re.search(r'[A-Z]', password) and re.search(r'\d', password) and re.search(r'[^A-Za-z0-9]', password)):
            flash('Password must contain at least one uppercase letter, one number, and one special character.', 'danger')
            return redirect(url_for('register'))
            
        # Check if email exists
        if User.query.filter_by(email=email).first():
            flash('Email address is already registered.', 'danger')
            return redirect(url_for('register'))
        
        is_approved = True if role == 'patient' else False
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(role=role, name=name, email=email, password=hashed_password, zipcode=zipcode, is_approved=is_approved, age=age, gender=gender)
        
        db.session.add(user)
        db.session.commit()
        
        try:
            msg = Message("Welcome to Medora", recipients=[email])
            if role == 'hospital':
                msg.body = "Thank you for registering your hospital with Medora! Your account is currently pending approval by our admin team. You will be notified once approved."
                msg.html = render_template('emails/hospital_pending.html', name=name)
            else:
                msg.body = f"Hello {name},\n\nWelcome to Medora! Your account has been created successfully."
                msg.html = render_template('emails/welcome.html', name=name)
            mail.send(msg)
        except Exception as e:
            print(f"Failed to send registration email to {email}: {e}")

        flash('Registration successful! Please login.', 'success')
        if role == 'hospital':
            flash('Hospital accounts require admin approval before logging in.', 'warning')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email == 'admin@gmail.com' and password == 'Admin@123':
            user = User.query.filter_by(email=email).first()
            if not user:
                hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                user = User(role='admin', name='Super Admin', email=email, password=hashed_password, zipcode='00000', is_approved=True)
                db.session.add(user)
                db.session.commit()
            login_user(user)
            return redirect(url_for('admin_dashboard'))

        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            if user.role == 'hospital' and not user.is_approved:
                flash("Your account is pending admin approval.", 'warning')
                return redirect(url_for('login'))
            
            login_user(user)
            if user.role == 'hospital':
                return redirect(url_for('hospital_dashboard'))
            elif user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/hospitals')
@login_required
def hospitals_list():
    if current_user.role != 'patient':
        return redirect(url_for('home'))
        
    prediction = request.args.get('prediction', 'General')
    confidence = request.args.get('confidence', '0')
    symptoms = request.args.get('symptoms', '')
    
    hospitals = User.query.filter_by(role='hospital', zipcode=current_user.zipcode, is_approved=True).all()
    range_start = datetime.today().strftime('%Y-%m-%d')
    return render_template('hospitals.html', hospitals=hospitals, prediction=prediction, confidence=confidence, symptoms=symptoms, range_start=range_start)

@app.route('/book_appointment/<int:hospital_id>', methods=['POST'])
@login_required
def book_appointment(hospital_id):
    if current_user.role != 'patient':
        return redirect(url_for('home'))
        
    prediction = request.form.get('prediction')
    confidence = request.form.get('confidence')
    symptoms = request.form.get('symptoms')
    appt_date = request.form.get('appointment_date')
    appt_time = request.form.get('appointment_time')
    
    parsed_date = datetime.strptime(appt_date, '%Y-%m-%d').date() if appt_date else None
    parsed_time = datetime.strptime(appt_time, '%H:%M').time() if appt_time else None

    appointment = Appointment(
        patient_id=current_user.id,
        hospital_id=hospital_id,
        predicted_disease=prediction,
        confidence=float(confidence),
        symptoms=symptoms,
        appointment_date=parsed_date,
        appointment_time=parsed_time
    )
    db.session.add(appointment)
    db.session.commit()
    
    hospital = User.query.get(hospital_id)
    
    # Real Email sending
    try:
        msg = Message("New Appointment Request", recipients=[hospital.email])
        msg.body = f"You have a new appointment request from {current_user.name} for suspected {prediction} on {appt_date} at {appt_time}.\n\nSymptoms: {symptoms}"
        msg.html = render_template('emails/appointment_request.html', patient=current_user, appt_date=appt_date, appt_time=appt_time, prediction=prediction, symptoms=symptoms)
        mail.send(msg)
    except Exception as e:
        print(f"Failed to send email to {hospital.email}: {e}")
        
    flash('Appointment requested successfully!', 'success')
    return redirect(url_for('patient_dashboard'))

@app.route('/patient_dashboard')
@login_required
def patient_dashboard():
    if current_user.role != 'patient':
        return redirect(url_for('home'))
    appointments = Appointment.query.filter_by(patient_id=current_user.id).order_by(Appointment.date_created.desc()).all()
    return render_template('patient_dashboard.html', appointments=appointments)

@app.route('/hospital_dashboard')
@login_required
def hospital_dashboard():
    if current_user.role != 'hospital':
        return redirect(url_for('home'))
    appointments = Appointment.query.filter_by(hospital_id=current_user.id).order_by(Appointment.date_created.desc()).all()
    return render_template('hospital_dashboard.html', appointments=appointments)

@app.route('/cancel_appointment/<int:app_id>', methods=['POST'])
@login_required
def cancel_appointment(app_id):
    if current_user.role != 'patient':
        return redirect(url_for('home'))
        
    appointment = Appointment.query.get_or_404(app_id)
    if appointment.patient_id != current_user.id:
        return redirect(url_for('patient_dashboard'))
        
    appointment.status = 'Cancelled'
    db.session.commit()
    
    try:
        msg = Message("Appointment Cancelled by Patient", recipients=[appointment.hospital.email])
        msg.body = f"The appointment request from {current_user.name} for {appointment.predicted_disease} originally scheduled on {appointment.appointment_date} at {appointment.appointment_time} has been Cancelled by the patient."
        # Not creating a dedicated HTML template for this cancellation notice right now, falling back to plain text.
        mail.send(msg)
    except Exception as e:
        print(f"Failed to send email to {appointment.hospital.email}: {e}")
        
    flash('Appointment cancelled.', 'success')
    return redirect(url_for('patient_dashboard'))

@app.route('/reschedule_appointment/<int:app_id>', methods=['POST'])
@login_required
def reschedule_appointment(app_id):
    if current_user.role != 'patient':
        return redirect(url_for('home'))
        
    appointment = Appointment.query.get_or_404(app_id)
    if appointment.patient_id != current_user.id:
        return redirect(url_for('patient_dashboard'))
        
    appt_date = request.form.get('appointment_date')
    appt_time = request.form.get('appointment_time')
    
    if appt_date and appt_time:
        appointment.appointment_date = datetime.strptime(appt_date, '%Y-%m-%d').date()
        appointment.appointment_time = datetime.strptime(appt_time, '%H:%M').time()
        appointment.status = 'Pending' # Reset to Pending so hospital can approve
        db.session.commit()
        
        try:
            msg = Message("Appointment Rescheduled by Patient", recipients=[appointment.hospital.email])
            msg.body = f"The appointment request from {current_user.name} for {appointment.predicted_disease} has been rescheduled to {appointment.appointment_date} at {appointment.appointment_time}. Please log in to accept or reject the new time."
            mail.send(msg)
        except Exception as e:
            print(f"Failed to send email: {e}")
            
        flash('Appointment rescheduled successfully. Waiting for hospital approval.', 'success')
    return redirect(url_for('patient_dashboard'))

@app.route('/update_appointment/<int:app_id>/<status>')
@login_required
def update_appointment(app_id, status):
    if current_user.role != 'hospital':
        return redirect(url_for('home'))
        
    appointment = Appointment.query.get_or_404(app_id)
    if appointment.hospital_id != current_user.id:
        return redirect(url_for('hospital_dashboard'))
        
    if status in ['Accepted', 'Rejected']:
        appointment.status = status
        db.session.commit()
        
        # Real email
        try:
            msg = Message(f"Appointment {status}", recipients=[appointment.patient.email])
            msg.body = f"Your appointment request at {appointment.hospital.name} on {appointment.appointment_date} at {appointment.appointment_time} has been {status}."
            msg.html = render_template('emails/appointment_status.html', appointment=appointment, status=status)
            mail.send(msg)
        except Exception as e:
            print(f"Failed to send email to {appointment.patient.email}: {e}")
            
    return redirect(url_for('hospital_dashboard'))

@app.route('/update_hospital_hours', methods=['POST'])
@login_required
def update_hospital_hours():
    if current_user.role != 'hospital':
        return redirect(url_for('home'))
    
    start = request.form.get('working_hours_start')
    end = request.form.get('working_hours_end')
    
    if start and end:
        current_user.working_hours_start = start
        current_user.working_hours_end = end
        db.session.commit()
        flash("Working hours updated successfully.", "success")
        
    return redirect(url_for('profile'))

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if current_user.role == 'patient':
        age = request.form.get('age')
        gender = request.form.get('gender')
        if age:
            current_user.age = age
        if gender:
            current_user.gender = gender
    
    db.session.commit()
    flash("Profile updated successfully.", "success")
    return redirect(url_for('profile'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return redirect(url_for('home'))
        
    pending_hospitals = User.query.filter_by(role='hospital', is_approved=False).all()
    all_users = User.query.filter(User.role.in_(['patient', 'hospital'])).all()
    
    # simple analytics
    searches = SearchActivity.query.all()
    
    return render_template('admin_dashboard.html', 
                          pending_hospitals=pending_hospitals,
                          users=all_users,
                          searches=searches)

@app.route('/manage_hospital/<int:id>/<action>')
@login_required
def manage_hospital(id, action):
    if current_user.role != 'admin':
        return redirect(url_for('home'))
        
    hospital = User.query.get_or_404(id)
    if hospital.role != 'hospital':
        return redirect(url_for('admin_dashboard'))
        
    if action == 'approve':
        hospital.is_approved = True
        flash(f"Hospital {hospital.name} approved.", 'success')
        
        try:
            msg = Message("Medora Hospital Account Approved", recipients=[hospital.email])
            msg.body = "Your hospital account has been approved! You can now log in."
            msg.html = render_template('emails/hospital_approved.html', hospital=hospital)
            mail.send(msg)
        except Exception as e:
            pass
            
    elif action == 'reject':
        db.session.delete(hospital)
        flash(f"Hospital {hospital.name} rejected and removed.", 'danger')
        
    db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route("/first-aid")
def first_aid():
    return render_template("first_aid.html")

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Initialize database on run
    app.run(debug=True)
