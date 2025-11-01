"""
OMR Answer Evaluation System - Main Application
Author: OMR System Team
Version: 1.0.0
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import sqlite3
from config import *

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE




os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize security modules
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# ==============================
# USER AUTHENTICATION
# ==============================

class User(UserMixin):
    def __init__(self, id, username, password, role):
        self.id = id
        self.username = username
        self.password = password
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, username, password, role FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return User(*row)
    return None

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Allow registration only if no users exist (first-time setup)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    user_count = c.fetchone()[0]
    conn.close()

    # If any user exists, block registration
    if user_count > 0:
        flash("Registration is disabled after first admin is created.", "info")
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template('register.html')

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, datetime('now'))",
                      (username, hashed_pw, 'admin'))
            conn.commit()
            flash("Admin account created successfully! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
        finally:
            conn.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, username, password, role FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if not user:
            flash(" No such user found. Please register first.", "danger")
            print("Flashed message: No such user found. Please register first")  # or similar for each case
            return redirect(url_for('login'))  #  use redirect, not render_template

        if user and bcrypt.check_password_hash(user[2], password):
            login_user(User(*user))
            flash(' Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash(' Invalid password. Please try again.', 'danger')
            print("Flashed message: Invalid password")  # or similar for each case
            return redirect(url_for('login'))  #  use redirect, not render_template

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_number TEXT NOT NULL,
        name TEXT,
        total_questions INTEGER,
        correct INTEGER,
        incorrect INTEGER,
        skipped INTEGER,
        score REAL,
        timestamp TEXT,
        answer_key TEXT
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS answer_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key_name TEXT UNIQUE NOT NULL,
        answers TEXT NOT NULL,
        total_questions INTEGER,
        created_at TEXT,
        updated_at TEXT
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT,
        created_at TEXT
    )""")
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

init_db()

# ============================================================================
# OMR PROCESSING ENGINE
# ============================================================================

class OMRProcessor:
    """Main class for processing OMR sheets."""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.gray_image = None
        self.contours = []
        self.bubbles = []
        self.answers = []
        self.metadata = {}
        
    def load_image(self):
        """Load and validate image format."""
        try:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                return False, "Unable to read image file. Check file format."
            
            height, width = self.original_image.shape[:2]
            
            if height < 100 or width < 100:
                return False, f"Image resolution too low: {width}x{height}"
            
            if height > 10000 or width > 10000:
                return False, "Image resolution too high"
            
            self.metadata['image_size'] = f"{width}x{height}"
            return True, "Image loaded successfully"
            
        except Exception as e:
            return False, f"Error loading image: {str(e)}"
    
    def preprocess_image(self):
        """Preprocess image for OMR detection."""
        try:
            import numpy as np
            import cv2

            # Convert to grayscale
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # Contrast stretch to normalize lighting
            gray = cv2.equalizeHist(gray)

            # Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Adaptive threshold (handles uneven lighting)
            thresh = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,  # note: INV -> black becomes white
                11, 2
            )

            # Small morphological opening to remove specks
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # Save for debugging
            cv2.imwrite("processed_debug.png", processed)

            self.gray_image = gray
            self.processed_image = processed
            return True, "Image preprocessed successfully"
            
        except Exception as e:
            return False, f"Preprocessing error: {str(e)}"
    
    def detect_bubbles(self):
        """Detect bubble contours in the OMR sheet."""
        cv2.imwrite("processed_debug.png", self.processed_image),
        print(f"Processed image saved. Shape: {self.processed_image.shape}")
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_OPEN, kernel)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return False, "No bubbles detected in sheet"
            
            bubble_list = []
            for contour in contours:
                print(cv2.contourArea(contour))
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by area and shape (circular/oval bubbles)
                #if BUBBLE_AREA_MIN < area < BUBBLE_AREA_MAX:
                if 200 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 1.5:
                        bubble_list.append({
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'area': area, 'center': (x + w//2, y + h//2)
                        })
           
            print(f"Total contours found: {len(contours)}")
            print(f"Valid bubbles found: {len(bubble_list)}")
            if len(bubble_list) < MIN_BUBBLES:
                return False, f"Insufficient bubbles detected: {len(bubble_list)}"
            
            # Sort bubbles by position (top to bottom, left to right)
            self.bubbles = sorted(bubble_list, key=lambda b: (b['y'], b['x']))
            return True, f"Detected {len(self.bubbles)} bubbles"
            
        except Exception as e:
            return False, f"Bubble detection error: {str(e)}"
    
    def extract_answers(self):
        """
        Extract marked answers from detected bubbles.
        Works for multi-column OMR layouts (like 4-column 60-question sheets).
        """
        import numpy as np
        import cv2
        from sklearn.cluster import KMeans
        from config import DARKNESS_THRESHOLD

        try:
            if not hasattr(self, 'bubbles') or len(self.bubbles) == 0:
                return False, "No bubbles found to extract answers from"

            # --- Step 1: Cluster bubbles into 4 columns based on X coordinate ---
            x_coords = np.array([[b['center'][0]] for b in self.bubbles])
            num_columns = 4
            kmeans = KMeans(n_clusters=num_columns, n_init=10, random_state=42).fit(x_coords)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_.flatten()

            # Assign each bubble its cluster
            for i, b in enumerate(self.bubbles):
                b['column'] = labels[i]

            # Sort columns left-to-right
            column_order = np.argsort(centers)
            columns = [[] for _ in range(num_columns)]
            for b in self.bubbles:
                col_index = int(np.where(column_order == b['column'])[0][0])
                columns[col_index].append(b)

            # --- Step 2: Sort each column top-to-bottom ---
            for col in columns:
                col.sort(key=lambda b: b['center'][1])

            # Debug output
            total_bubbles = sum(len(c) for c in columns)
            print(f"Total bubbles after ordering: {total_bubbles}")
            for i, col in enumerate(columns):
                print(f"Column {i+1}: {len(col)} bubbles")

            # --- Step 3: Extract answers column-wise ---
            answers = []
            question_index = 0

            for col_idx, col_bubbles in enumerate(columns):
                # Each question has 4 options (A–D)
                num_questions_in_col = len(col_bubbles) // 4

                for q in range(num_questions_in_col):
                    question_bubbles = col_bubbles[q * 4:(q + 1) * 4]
                    darkness_values = []

                    for bubble in question_bubbles:
                        x, y, w, h = bubble['x'], bubble['y'], bubble['w'], bubble['h']
                        pad = 2
                        x1 = max(x + pad, 0)
                        y1 = max(y + pad, 0)
                        x2 = x + w - pad
                        y2 = y + h - pad
                        roi = self.gray_image[y1:y2, x1:x2]

                        if roi.size == 0:
                            darkness = 0
                        else:
                            mean_val = np.mean(roi)
                            darkness = 255 - mean_val  # higher = darker

                        darkness_values.append(darkness)

                    max_darkness = max(darkness_values)
                    selected_option = darkness_values.index(max_darkness) if max_darkness > DARKNESS_THRESHOLD else -1
                    answers.append(selected_option)
                    question_index += 1

            self.answers = answers
            print(f"[DEBUG] Answers extracted: {len(answers)}")
            print(f"[DEBUG] First 10 detected answers: {answers[:10]}")

            return True, f"Extracted {len(answers)} answers successfully"

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return False, f"Answer extraction error: {str(e)}"

    
    def validate_sheet(self):
        """Validate OMR sheet quality and structure."""
        issues = []
        
        if self.original_image is None:
            issues.append("Image not loaded")
        
        if self.processed_image is None:
            issues.append("Image not preprocessed")
        elif self.processed_image.size < 10000:
            issues.append("Processed image too small")
        
        if not self.bubbles:
            issues.append("No bubbles detected")
        elif len(self.bubbles) < MIN_BUBBLES:
            issues.append(f"Insufficient bubbles ({len(self.bubbles)} < {MIN_BUBBLES})")
        
        # Check image clarity using Laplacian
        if self.gray_image is not None:
            laplacian = cv2.Laplacian(self.gray_image, cv2.CV_64F)
            clarity_score = laplacian.var()
            if clarity_score < CLARITY_THRESHOLD:
                issues.append(f"Image clarity poor (score: {clarity_score:.2f})")
        
        return len(issues) == 0, issues

# ============================================================================
# ANSWER KEY MANAGEMENT
# ============================================================================

class AnswerKeyManager:
    """Manage answer keys in database."""
    
    @staticmethod
    def create_key(key_name, answers):
        """Create and store answer key."""
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            timestamp = datetime.now().isoformat()
            answers_json = json.dumps(answers)
            
            c.execute("""INSERT OR REPLACE INTO answer_keys 
                        (key_name, answers, total_questions, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)""",
                     (key_name, answers_json, len(answers), timestamp, timestamp))
            
            conn.commit()
            conn.close()
            return True, f"Answer key '{key_name}' created with {len(answers)} questions"
            
        except sqlite3.IntegrityError:
            return False, "Answer key name already exists"
        except Exception as e:
            return False, f"Error creating key: {str(e)}"
    
    @staticmethod
    def get_key(key_name):
        """Retrieve answer key by name."""
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("""SELECT answers FROM answer_keys 
                        WHERE key_name = ? LIMIT 1""", (key_name,))
            result = c.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return None
            
        except Exception as e:
            print(f"Error retrieving key: {str(e)}")
            return None
    
    @staticmethod
    def list_keys():
        """List all available answer keys."""
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("""SELECT key_name, total_questions, created_at 
                        FROM answer_keys ORDER BY created_at DESC""")
            keys = []
            for row in c.fetchall():
                keys.append({
                    'name': row[0],
                    'questions': row[1],
                    'created': row[2]
                })
            conn.close()
            return keys
            
        except Exception:
            return []
    
    @staticmethod
    def delete_key(key_name):
        """Delete an answer key."""
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("DELETE FROM answer_keys WHERE key_name = ?", (key_name,))
            conn.commit()
            conn.close()
            return True, "Answer key deleted"
            
        except Exception as e:
            return False, f"Error deleting key: {str(e)}"

# ============================================================================
# EVALUATION ENGINE
# ============================================================================

class EvaluationEngine:
    """Evaluate answers against answer key."""
    
    @staticmethod
    def evaluate(extracted_answers, answer_key):
        """Compare extracted answers with answer key and calculate score."""
        try:
            if not extracted_answers:
                return None, "No answers to evaluate"
            
            if not answer_key:
                return None, "Invalid answer key"
            
            correct = 0
            incorrect = 0
            skipped = 0
            
            limit = min(len(extracted_answers), len(answer_key))
            for i in range(limit):
                extracted = extracted_answers[i]

                
                if extracted == -1:
                    skipped += 1
                elif extracted == answer_key[i]:
                    correct += 1
                else:
                    incorrect += 1
            
            total = len(answer_key)
            score = (correct / total * 100) if total > 0 else 0
            
            return {
                'total': total,
                'correct': correct,
                'incorrect': incorrect,
                'skipped': skipped,
                'score': round(score, 2),
                'accuracy': round((correct + skipped) / total * 100, 2)
            }, "Evaluation complete"
            
        except Exception as e:
            return None, f"Evaluation error: {str(e)}"
    
    @staticmethod
    def save_result(roll_number, name, evaluation, key_name):
        """Save evaluation result to database."""
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            c.execute("""INSERT INTO results
                        (roll_number, name, total_questions, correct, 
                         incorrect, skipped, score, timestamp, answer_key)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (roll_number, name, evaluation['total'], evaluation['correct'],
                      evaluation['incorrect'], evaluation['skipped'], 
                      evaluation['score'], timestamp, key_name))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving result: {str(e)}")
            return False

# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_result_chart(evaluation):
    """Generate visualizations for results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    sizes = [evaluation['correct'], evaluation['incorrect'], evaluation['skipped']]
    labels = ['Correct', 'Incorrect', 'Skipped']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Answer Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart
    ax2.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Performance Summary', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

# ============================================================================
# FILE UTILITIES
# ============================================================================

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# FLASK ROUTES - API ENDPOINTS
# ============================================================================

@app.route('/')
@login_required
def index():
    """Home page (only visible after login)."""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_omr():
    """Process uploaded OMR sheet."""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400
        
        file = request.files['file']
        answer_key_name = request.form.get('answer_key')
        roll_number = request.form.get('roll_number', 'Unknown')
        student_name = request.form.get('student_name', 'N/A')
        
        if not file or file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 
                          'message': 'Invalid format. Use JPG, PNG'}), 400
        
        if file.content_length and file.content_length > MAX_FILE_SIZE:
            return jsonify({'success': False, 'message': 'File too large'}), 400
        
        # Save file
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process OMR
        processor = OMRProcessor(filepath)
        
        success, msg = processor.load_image()
        if not success:
            os.remove(filepath)
            return jsonify({'success': False, 'message': msg}), 400
        
        success, msg = processor.preprocess_image()
        if not success:
            os.remove(filepath)
            return jsonify({'success': False, 'message': msg}), 400
        
        success, msg = processor.detect_bubbles()
        if not success:
            os.remove(filepath)
            return jsonify({'success': False, 'message': msg}), 400
        
        is_valid, issues = processor.validate_sheet()
        if not is_valid:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'message': 'Sheet validation failed',
                'issues': issues
            }), 400
        
        success, msg = processor.extract_answers()
        if not success:
            os.remove(filepath)
            return jsonify({'success': False, 'message': msg}), 400
        
        # Get answer key
        answer_key = AnswerKeyManager.get_key(answer_key_name)
        print(f"[DEBUG] Loaded answer key '{answer_key_name}' with {len(answer_key)} answers")

        if not answer_key:
            os.remove(filepath)
            return jsonify({'success': False, 'message': 'Answer key not found'}), 400
        
        # Evaluate
        evaluation, msg = EvaluationEngine.evaluate(processor.answers, answer_key)
        if not evaluation:
            os.remove(filepath)
            return jsonify({'success': False, 'message': msg}), 400
        
        # Save result
        EvaluationEngine.save_result(roll_number, student_name, evaluation, answer_key_name)
        
        # Generate chart
        chart = generate_result_chart(evaluation)
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': 'OMR processed successfully',
            'result': evaluation,
            'chart': chart,
            'roll_number': roll_number,
            'student_name': student_name
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/create-answer-key', methods=['POST'])
@login_required
def create_answer_key():
    """Create new answer key."""
    try:
        data = request.json
        key_name = data.get('key_name', '').strip()
        answers = data.get('answers')
        
        if not key_name:
            return jsonify({'success': False, 'message': 'Key name required'}), 400
        
        if not answers or not isinstance(answers, list):
            return jsonify({'success': False, 'message': 'Answers must be a list'}), 400
        
        success, msg = AnswerKeyManager.create_key(key_name, answers)
        return jsonify({'success': success, 'message': msg})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/answer-keys', methods=['GET'])
@login_required
def get_answer_keys():
    """Get all answer keys."""
    keys = AnswerKeyManager.list_keys()
    return jsonify({'keys': keys})

@app.route('/api/delete-key/<key_name>', methods=['DELETE'])
@login_required
def delete_key(key_name):
    """Delete an answer key."""
    success, msg = AnswerKeyManager.delete_key(key_name)
    return jsonify({'success': success, 'message': msg})

@app.route('/api/results', methods=['GET'])
@login_required
def get_results():
    """Get all evaluation results."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""SELECT roll_number, name, total_questions, correct, 
                           incorrect, skipped, score, timestamp FROM results 
                    ORDER BY timestamp DESC""")
        results = []
        for row in c.fetchall():
            results.append({
                'roll_number': row[0],
                'name': row[1],
                'total': row[2],
                'correct': row[3],
                'incorrect': row[4],
                'skipped': row[5],
                'score': row[6],
                'timestamp': row[7]
            })
        conn.close()
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export-results', methods=['GET'])
@login_required
def export_results():
    """Export results to CSV."""
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query('SELECT * FROM results ORDER BY timestamp DESC', conn)
        conn.close()
        
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return send_file(output, mimetype='text/csv', as_attachment=True,
                        download_name=filename)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
@login_required
def get_statistics():
    """Get overall statistics."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*), AVG(score), MAX(score), MIN(score) FROM results")
        stats = c.fetchone()
        
        return jsonify({
            'total_evaluated': stats[0] or 0,
            'avg_score': round(stats[1] or 0, 2),
            'max_score': round(stats[2] or 0, 2),
            'min_score': round(stats[3] or 0, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("OMR Answer Evaluation System")
    print("=" * 50)
    print("Starting Flask server...")
    print("Access at: http://localhost:5000")
    print("=" * 50)
    #app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=false, host='0.0.0.0', port=8080)
