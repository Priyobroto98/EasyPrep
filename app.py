from flask import Flask, request, render_template, session, redirect, url_for,jsonify,flash
from flask_mail import Mail,Message
from quiz import *
from question import *
from impTopics import *
from flask import Flask, request,render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from middleware import *
app = Flask(__name__)
Bootstrap(app)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""

        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += file.read().decode('utf-8')
        else:
            text = request.form['text']

        global_session['text'] = text
        return '', 204

    return render_template('index.html')

@app.route('/quiz', methods=['GET', 'POST'])
@auth
def quiz():
    if request.method == 'POST':
        text = ""
        # Check if files were uploaded
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    # Process PDF file
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    # Process text file
                    text += file.read().decode('utf-8')
        else:
            # Process manual input
            text = request.form['text']
        question_no = int(request.form['num_questions'])

        mcqs = generate_and_transform_mcq(text, num_questions=question_no)  
        print(mcqs)
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
        return render_template('mcqs.html', mcqs=mcqs_with_index)

    return render_template('quiz.html')

@app.route('/important_topics')
def important_topics():
    video_info = []  # Initialize video_info

    if 'text' in global_session:
        input_text = global_session['text']
        results = chain.invoke(input_text)
        results_text = results['text']
        
        video_info = vdo_id(results_text)
        
        
    return render_template('impTopics.html', video_info=video_info)

#-----------------------------------------Question Answering RAG--------------------------------------------

@app.route('/question', methods=['GET', 'POST'])
def question():
    chat_history_html = ""
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_question = request.form['user_question']
        chat_history_html = handle_userinput(user_question)

    return render_template('question.html', css=css, chat_history_html=chat_history_html)

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        flash('No file part')
        return redirect(request.url)

    uploaded_files = request.files.getlist('files')
    raw_text = get_pdf_text(uploaded_files=uploaded_files)
    text_chunks = get_text_chunks(results=raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = get_vectorstore(results=text_chunks, embeddings_model_name=embeddings_model_name,
                                  persist_directory=persist_directory, client_settings=CHROMA_SETTINGS,
                                  chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    conversation_chain = get_conversation_chain(vectorstore=vectorstore,
                                                target_source_chunks=target_source_chunks,
                                                model_type=model_type)

    conversation_id = str(uuid.uuid4())
    conversation_store[conversation_id] = conversation_chain
    session['conversation_id'] = conversation_id
    return redirect(url_for('question'))
#--------------------------------------------------------------------------------------------------------------
import pickle
import logging
from summarizer import Summarizer
logging.basicConfig(level=logging.DEBUG)

try:
    model = pickle.load(open(r'C:\Users\Nilkantha\Desktop\iTutor\model\bert_summarizer_model.pkl', 'rb'))
    logging.info('Model loaded successfully.')
except Exception as e:
    logging.error(f'Error loading model: {e}')

@app.route('/summarization')
def summarization():
    input_text = global_session.get('text', '')
    
    if not input_text:
        return render_template('summarization.html', msg="No input text provided.")
    
    logging.debug(f'Input text: {input_text}')
    summary = model(input_text, ratio=0.4)
    lines = summary.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    processed_message = '. '.join(re.sub(r'^\*+', '', line).strip() for line in lines)
    logging.debug(f'Summary: {processed_message}')

    return render_template('summarization.html', msg=processed_message)


#---------------data base work --------------------------------------------------------------------

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'app_secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    college_name = db.Column(db.String(255), nullable=False)
    college_email = db.Column(db.String(100), unique=True, nullable=False)
    personal_email = db.Column(db.String(100), unique=True, nullable=False)
    department = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.String(100), unique=True, nullable=False)
    semester = db.Column(db.String(50), nullable=False)
    year = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __init__(self, name, address, college_name, college_email, personal_email, department, roll_number, semester, year, password):
        self.name = name
        self.address = address
        self.college_name = college_name
        self.college_email = college_email
        self.personal_email = personal_email
        self.department = department
        self.roll_number = roll_number
        self.semester = semester
        self.year = year
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# Initialize the database (this should be done in your application context)
with app.app_context():
    db.create_all()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        college_name = request.form['college_name']
        college_email = request.form['college_email']
        personal_email = request.form['personal_email']
        department = request.form['department']
        roll_number = request.form['roll_number']
        password = request.form['password']
        semester = request.form['semester']
        year = request.form['year']

        new_user = User(
            name=name,
            address=address,
            college_name=college_name,
            college_email=college_email,
            personal_email=personal_email,
            department=department,
            roll_number=roll_number,
            password=password,
            semester=semester,
            year=year
        )

        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        college_email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(college_email=college_email).first()  # Updated to use college_email
        
        if user and user.check_password(password):
            session['email'] = user.college_email  # Store college_email in session
            next_url = session.pop('next', None)  # Get and remove the 'next' URL from the session
            if next_url:
                return redirect(next_url)  # Redirect to the original URL
            return render_template('login.html', msg='Successfully logged in!')
        else:
            return render_template('login.html', msg='Invalid email or password')
    return render_template('login.html')

@app.route('/dashboard')
@auth
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))

    college_email = session['email']
    user = User.query.filter_by(college_email=college_email).first()

    if user:
        return render_template('dashboard.html', user=user)
    else:
        return redirect(url_for('login'))


@app.route('/logout')

def logout():
    session.pop('email', None)
    return redirect('/login')
#--------------------------------------------feedback section--------------------------------------------

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='sonasonu9008@gmail.com',
    MAIL_PASSWORD='enzw azbn dhbr ecdj'
)

# Initialize the Mail object
mail = Mail(app)

@app.route('/mail')
def mail_page():
    return render_template('feedback.html')

@app.route('/feedback', methods=["POST"])
def feedback():
    name = request.form['name']
    email = request.form['email']
    subject = request.form['subject']
    message = request.form['message']
    
    msg = Message(subject=subject, sender=email, recipients=['sonasonu9008@gmail.com'])
    msg.body = message
    
    mail.send(msg)

    return render_template('feedback.html',msg='Thank You for Your Feedback! Please Visit our Site again.')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

    