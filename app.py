from roastbot import generate_roast 
from motivation import generate_motivation
from response import generate_response
from flask import Flask, render_template, request, redirect, url_for, flash, session ,jsonify 
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit, join_room, leave_room
import random

app = Flask(__name__)
socketio = SocketIO(app)
waiting_users = []
app.secret_key = 'your_secret_key_here'  
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


with app.app_context():
    db.create_all()

@app.route('/')
def index():
    if 'user_id' in session:  
        user_id = session['user_id']
        user = User.query.get(user_id)
        if user:
            return render_template('index.html', user=user)
    return render_template('index.html')

@socketio.on('joinRoom')
def join_room_event():
    global waiting_users
    sid = request.sid
    if waiting_users:  
        peer_sid = waiting_users.pop(0)  
        join_room(peer_sid)
        join_room(sid)
        emit('peerFound', {'isCaller': True}, room=peer_sid)
        emit('peerFound', {'isCaller': False}, room=sid)
    else:
        waiting_users.append(sid)

@socketio.on('leaveRoom')
def leave_room_event():
    sid = request.sid
    if sid in waiting_users:
        waiting_users.remove(sid)
    leave_room(sid)

@socketio.on('signal')
def signaling(data):
    emit('signal', data, broadcast=True, include_self=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id 
            flash('Login successful!', 'success')
            return redirect(url_for('index'))  
        else:
            flash('Invalid email or password.', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
        else:
            new_user = User(name=name, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None) 
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/home')
def home():
    return render_template('index.html') 


@app.route('/motivational-mode')
def motivational_mode():
    return render_template('motivational_mode.html')

@app.route('/motivate', methods=['POST'])
def motivate():
    user_message = request.json.get('message', '')
    motivation_reply = generate_motivation(user_message) 
    return jsonify({"reply": motivation_reply})


@app.route('/skills-increase-mode')
def skills_increase_mode():
    return render_template('skills_increase_mode.html')

@app.route('/meditation-mode')
def meditation_mode():
    return render_template('meditation_mode.html')

@app.route('/roast-mode')
def roast_mode():
    return render_template('roast_mode.html')

@app.route('/roast', methods=['POST'])
def roast():
    user_message = request.json.get('message', '')
    roast_reply = generate_roast(user_message) 
    return jsonify({"reply": roast_reply})

@app.route('/get-cohere-response', methods=['POST'])
def get_cohere_response():
    try:
        # Get the text from the POST request
        data = request.get_json()
        user_input = data.get('text')

        if user_input:
            # Get the response from the Cohere API
            response_text = generate_response(user_input)
            return jsonify({"response": response_text})

        else:
            return jsonify({"error": "No text provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    socketio.run(app, debug=True)
    app.run(debug=True)

