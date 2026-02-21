from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
import json
import os
import sys
import PyPDF2
import docx
import re
from datetime import datetime
import secrets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.predict import ResumePredictor
from database import Database

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app, supports_credentials=True)
app.secret_key = secrets.token_hex(16)

def generate_smart_suggestions(resume_text, predicted_role=None):
    """Generate dynamic NLP-based suggestions from actual resume content"""
    suggestions = []
    issues = []
    text_lower = resume_text.lower()
    words = resume_text.split()

    # 1. LENGTH CHECK
    word_count = len(words)
    if word_count < 200:
        issues.append(f"Your resume is too short ({word_count} words). Aim for 400-700 words.")
    elif word_count > 900:
        issues.append(f"Your resume is too long ({word_count} words). Try to keep it under 700 words.")
    else:
        suggestions.append(f"Good resume length ({word_count} words) — within the ideal range.")

    # 2. ACTION VERBS CHECK
    action_verbs = ['developed','managed','led','created','implemented','designed',
                    'analyzed','improved','coordinated','achieved','executed',
                    'established','built','optimized','delivered','increased',
                    'reduced','launched','trained','mentored','collaborated',
                    'negotiated','presented','resolved','streamlined']
    found_verbs = [v for v in action_verbs if v in text_lower]
    missing_verbs = [v for v in action_verbs if v not in text_lower]
    if len(found_verbs) < 3:
        issues.append(f"Very few action verbs found ({len(found_verbs)}). Add more like: {', '.join(missing_verbs[:5])}.")
    elif len(found_verbs) < 6:
        suggestions.append(f"You used {len(found_verbs)} action verbs. Consider adding: {', '.join(missing_verbs[:3])}.")
    else:
        suggestions.append(f"Great use of {len(found_verbs)} action verbs including: {', '.join(found_verbs[:4])}.")

    # 3. QUANTIFICATION CHECK
    import re
    numbers = re.findall(r'\b\d+[\%\+]?\b', resume_text)
    if len(numbers) < 2:
        issues.append("No quantified achievements found. Add numbers like '30% increase', 'team of 10', '$5K budget'.")
    else:
        suggestions.append(f"Good — {len(numbers)} quantified achievements found.")

    # 4. CONTACT INFO
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text))
    has_phone = bool(re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text))
    has_linkedin = 'linkedin' in text_lower
    if not has_email:
        issues.append("No email address detected. Add your professional email.")
    if not has_phone:
        issues.append("No phone number detected. Add your contact number.")
    if not has_linkedin:
        suggestions.append("Consider adding your LinkedIn profile URL.")

    # 5. SECTION DETECTION
    sections = {'Summary': ['summary','objective','profile','about'],
                'Experience': ['experience','employment','work history'],
                'Education': ['education','qualification','degree','university'],
                'Skills': ['skills','competencies','technical skills'],
                'Projects': ['projects','portfolio','work samples'],
                'Certifications': ['certification','certificate','certified']}
    found_sections = [s for s, kws in sections.items() if any(k in text_lower for k in kws)]
    missing_sections = [s for s in sections if s not in found_sections]
    if missing_sections:
        issues.append(f"Missing sections detected: {', '.join(missing_sections)}. Add these for a complete resume.")
    else:
        suggestions.append("All key resume sections are present.")

    # 6. ROLE-SPECIFIC KEYWORD CHECK
    if predicted_role:
        role_keywords = {
            'DATA-SCIENCE': ['python','machine learning','sql','tensorflow','pandas','numpy','statistics','data analysis','sklearn','jupyter'],
            'WEB-DEVELOPER': ['html','css','javascript','react','node','api','git','responsive','frontend','backend'],
            'HR': ['recruitment','onboarding','hris','performance','talent','payroll','employee relations','training'],
            'DESIGNER': ['figma','adobe','ux','ui','wireframe','prototype','typography','color','design thinking'],
            'ENGINEERING': ['cad','autocad','solidworks','quality','tolerance','manufacturing','testing','specifications'],
            'FINANCE': ['financial analysis','excel','budgeting','forecasting','audit','tax','gaap','variance'],
            'HEALTHCARE': ['patient','clinical','emr','hipaa','diagnosis','treatment','care','medical'],
            'SALES': ['crm','revenue','target','pipeline','negotiation','lead','client','quota'],
            'INFORMATION-TECHNOLOGY': ['network','server','cloud','aws','azure','security','linux','infrastructure'],
            'CHEF': ['kitchen','menu','haccp','culinary','food safety','mise en place','recipe','cuisine'],
            'FITNESS': ['training','nutrition','hiit','exercise','program design','client','assessment','workout'],
        }
        keywords = role_keywords.get(predicted_role, [])
        found_kw = [k for k in keywords if k in text_lower]
        missing_kw = [k for k in keywords if k not in text_lower]
        if len(found_kw) < 3:
            issues.append(f"Only {len(found_kw)} keywords found for {predicted_role} role. Add: {', '.join(missing_kw[:5])}.")
        else:
            suggestions.append(f"Found {len(found_kw)} relevant keywords for {predicted_role}: {', '.join(found_kw[:4])}.")
        if missing_kw:
            suggestions.append(f"Consider adding these {predicted_role} keywords: {', '.join(missing_kw[:5])}.")

    # 7. REPETITION CHECK
    word_freq = {}
    for word in words:
        w = word.lower().strip('.,;:')
        if len(w) > 4:
            word_freq[w] = word_freq.get(w, 0) + 1
    overused = [w for w, c in word_freq.items() if c > 4 and w not in ['which','their','about','these','there','where','would','could','should']]
    if overused:
        suggestions.append(f"Overused words detected: '{', '.join(overused[:3])}'. Try varying your language.")

    return {'issues': issues, 'suggestions': suggestions}

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/mcq_test')
def mcq_test_page():
    return render_template('mcq_test.html')
app.secret_key = secrets.token_hex(16)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.pdf', '.docx']

predictor = ResumePredictor()

db = Database(
    server='localhost\\SQLEXPRESS',
    use_windows_auth=True
)

questions_path = os.path.join(os.path.dirname(__file__), 'data', 'interview_questions.json')
with open(questions_path, 'r') as f:
    interview_questions = json.load(f)


# ============================================================
# ROLE NORMALIZATION MAP
# This maps whatever your ML model outputs → exact DB role name
# ============================================================
ROLE_NORMALIZATION_MAP = {
    # Data Science variants
    'data scientist':        'DATA-SCIENCE',
    'data science':          'DATA-SCIENCE',
    'data-science':          'DATA-SCIENCE',
    'data_science':          'DATA-SCIENCE',
    'datascience':           'DATA-SCIENCE',
    'DATA SCIENTIST':        'DATA-SCIENCE',
    'DATA-SCIENTIST':        'DATA-SCIENCE',

    # Web Developer variants
    'web developer':         'WEB-DEVELOPER',
    'web development':       'WEB-DEVELOPER',
    'web-developer':         'WEB-DEVELOPER',
    'web_developer':         'WEB-DEVELOPER',
    'webdeveloper':          'WEB-DEVELOPER',
    'WEB DEVELOPER':         'WEB-DEVELOPER',

    # Python Developer variants
    'python developer':      'DATA-SCIENCE',   # maps to DATA-SCIENCE since no PYTHON role in DB
    'python':                'DATA-SCIENCE',
    'python dev':            'DATA-SCIENCE',

    # HR variants
    'hr':                    'HR',
    'human resources':       'HR',
    'human resource':        'HR',
    'hr manager':            'HR',

    # Designer variants
    'designer':              'DESIGNER',
    'ui designer':           'DESIGNER',
    'ux designer':           'DESIGNER',
    'ui/ux designer':        'DESIGNER',
    'graphic designer':      'DESIGNER',

    # Information Technology variants
    'information technology': 'INFORMATION-TECHNOLOGY',
    'information-technology': 'INFORMATION-TECHNOLOGY',
    'it':                    'INFORMATION-TECHNOLOGY',
    'it professional':       'INFORMATION-TECHNOLOGY',

    # Teacher variants
    'teacher':               'TEACHER',
    'educator':              'TEACHER',
    'professor':             'TEACHER',
    'instructor':            'TEACHER',

    # Advocate variants
    'advocate':              'ADVOCATE',
    'lawyer':                'ADVOCATE',
    'attorney':              'ADVOCATE',
    'legal':                 'ADVOCATE',

    # Business Development variants
    'business development':  'BUSINESS-DEVELOPMENT',
    'business-development':  'BUSINESS-DEVELOPMENT',
    'business developer':    'BUSINESS-DEVELOPMENT',
    'bd':                    'BUSINESS-DEVELOPMENT',

    # Healthcare variants
    'healthcare':            'HEALTHCARE',
    'health care':           'HEALTHCARE',
    'medical':               'HEALTHCARE',
    'doctor':                'HEALTHCARE',
    'nurse':                 'HEALTHCARE',

    # Fitness variants
    'fitness':               'FITNESS',
    'fitness trainer':       'FITNESS',
    'personal trainer':      'FITNESS',
    'gym trainer':           'FITNESS',

    # Agriculture variants
    'agriculture':           'AGRICULTURE',
    'agriculturist':         'AGRICULTURE',
    'farmer':                'AGRICULTURE',

    # BPO variants
    'bpo':                   'BPO',
    'call center':           'BPO',
    'customer service':      'BPO',

    # Sales variants
    'sales':                 'SALES',
    'sales executive':       'SALES',
    'sales manager':         'SALES',

    # Consultant variants
    'consultant':            'CONSULTANT',
    'consulting':            'CONSULTANT',
    'business consultant':   'CONSULTANT',

    # Digital Media variants
    'digital media':         'DIGITAL-MEDIA',
    'digital-media':         'DIGITAL-MEDIA',
    'digital marketing':     'DIGITAL-MEDIA',
    'social media':          'DIGITAL-MEDIA',

    # Automobile variants
    'automobile':            'AUTOMOBILE',
    'automotive':            'AUTOMOBILE',
    'mechanic':              'AUTOMOBILE',

    # Chef variants
    'chef':                  'CHEF',
    'cook':                  'CHEF',
    'culinary':              'CHEF',

    # Finance variants
    'finance':               'FINANCE',
    'financial analyst':     'FINANCE',
    'finance manager':       'FINANCE',

    # Apparel variants
    'apparel':               'APPAREL',
    'fashion':               'APPAREL',
    'fashion designer':      'APPAREL',

    # Engineering variants
    'engineering':           'ENGINEERING',
    'engineer':              'ENGINEERING',
    'software engineer':     'ENGINEERING',
    'civil engineer':        'ENGINEERING',
    'mechanical engineer':   'ENGINEERING',

    # Accountant variants
    'accountant':            'ACCOUNTANT',
    'accounting':            'ACCOUNTANT',
    'ca':                    'ACCOUNTANT',
    'chartered accountant':  'ACCOUNTANT',

    # Construction variants
    'construction':          'CONSTRUCTION',
    'civil':                 'CONSTRUCTION',
    'construction manager':  'CONSTRUCTION',

    # Public Relations variants
    'public relations':      'PUBLIC-RELATIONS',
    'public-relations':      'PUBLIC-RELATIONS',
    'pr':                    'PUBLIC-RELATIONS',

    # Banking variants
    'banking':               'BANKING',
    'bank':                  'BANKING',
    'banker':                'BANKING',
    'finance banking':       'BANKING',

    # Arts variants
    'arts':                  'ARTS',
    'artist':                'ARTS',
    'fine arts':             'ARTS',

    # Aviation variants
    'aviation':              'AVIATION',
    'pilot':                 'AVIATION',
    'airline':               'AVIATION',

    # Default
    'general':               'DEFAULT',
    'default':               'DEFAULT',
}

# All valid DB roles
VALID_DB_ROLES = {
    'HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
    'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO',
    'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE',
    'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS',
    'BANKING', 'ARTS', 'AVIATION', 'DATA-SCIENCE', 'WEB-DEVELOPER', 'DEFAULT'
}


def normalize_role(predicted_role: str) -> str:
    """
    Converts any ML model output into the exact role name stored in the DB.

    Steps:
    1. Check if it's already a valid DB role (exact match)
    2. Try lowercase lookup in normalization map
    3. Try partial/substring match against valid roles
    4. Fall back to DEFAULT
    """
    if not predicted_role:
        return 'DEFAULT'

    # Step 1: Direct match — already a valid DB role
    if predicted_role in VALID_DB_ROLES:
        print(f"[ROLE] Direct match: '{predicted_role}'")
        return predicted_role

    # Step 2: Lowercase lookup in normalization map
    lower = predicted_role.lower().strip()
    if lower in ROLE_NORMALIZATION_MAP:
        mapped = ROLE_NORMALIZATION_MAP[lower]
        print(f"[ROLE] Mapped: '{predicted_role}' → '{mapped}'")
        return mapped

    # Step 3: Partial match — e.g. "Data Scientist (Senior)" → DATA-SCIENCE
    for key, value in ROLE_NORMALIZATION_MAP.items():
        if key in lower or lower in key:
            print(f"[ROLE] Partial match: '{predicted_role}' → '{value}'")
            return value

    # Step 4: Check if any valid role is a substring of the predicted role
    upper = predicted_role.upper().replace(' ', '-')
    for valid_role in VALID_DB_ROLES:
        if valid_role in upper or upper in valid_role:
            print(f"[ROLE] Substring match: '{predicted_role}' → '{valid_role}'")
            return valid_role

    print(f"[ROLE] No match found for '{predicted_role}', falling back to DEFAULT")
    return 'DEFAULT'


def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        total_pages = len(pdf_reader.pages)
        max_pages = min(total_pages, 50)
        for page_num in range(max_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_num + 1}: {str(e)}")
                continue
        if not text.strip():
            raise Exception("No text could be extracted from PDF.")
        if len(text) > 50000:
            text = text[:50000]
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells if cell.text.strip()]
                if row_text:
                    tables_text.append(' '.join(row_text))
        text = '\n'.join(paragraphs + tables_text)
        if not text.strip():
            raise Exception("No text could be extracted from DOCX file.")
        if len(text) > 50000:
            text = text[:50000]
        return text
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")


def check_ats_friendliness(text):
    issues = []
    suggestions = []
    score = 100
    details = {}

    if len(text) < 300:
        issues.append("Resume is too short")
        suggestions.append("Add more details about your experience, skills, and achievements")
        score -= 25
        details['length'] = 'Poor'
    elif len(text) < 800:
        issues.append("Resume could be more detailed")
        suggestions.append("Expand on your key achievements and responsibilities")
        score -= 10
        details['length'] = 'Fair'
    else:
        details['length'] = 'Good'

    text_lower = text.lower()
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    has_email = bool(re.search(email_pattern, text))
    has_phone = bool(re.search(phone_pattern, text))

    if not has_email:
        issues.append("Missing email address")
        suggestions.append("Add a professional email address")
        score -= 15
    if not has_phone:
        issues.append("Missing phone number")
        suggestions.append("Include your contact phone number")
        score -= 10
    details['contact_info'] = 'Complete' if (has_email and has_phone) else 'Incomplete'

    sections_found = []
    sections_missing = []
    section_keywords = {
        'Experience': ['experience', 'work history', 'employment', 'professional experience'],
        'Education': ['education', 'qualification', 'degree', 'academic', 'university', 'college'],
        'Skills': ['skills', 'technical skills', 'competencies', 'proficiencies'],
        'Summary': ['summary', 'objective', 'profile', 'about me']
    }
    for section, keywords in section_keywords.items():
        if any(k in text_lower for k in keywords):
            sections_found.append(section)
        else:
            sections_missing.append(section)
            if section in ['Experience', 'Education', 'Skills']:
                issues.append(f"Missing '{section}' section")
                suggestions.append(f"Add a clear '{section}' section")
                score -= 15

    details['sections'] = f"{len(sections_found)}/4 key sections found"

    action_verbs = ['developed', 'managed', 'led', 'created', 'implemented', 'designed',
                    'analyzed', 'improved', 'coordinated', 'achieved', 'executed',
                    'established', 'built', 'optimized', 'delivered', 'increased']
    verb_count = sum(1 for v in action_verbs if v in text_lower)
    if verb_count < 3:
        issues.append("Limited use of strong action verbs")
        suggestions.append("Use more action verbs like: developed, managed, led")
        score -= 12
        details['action_verbs'] = 'Poor'
    elif verb_count < 6:
        details['action_verbs'] = 'Fair'
    else:
        details['action_verbs'] = 'Good'

    special_char_ratio = len(re.findall(r'[^\w\s.,;:!?()\-\'/\n]', text)) / max(len(text), 1)
    if special_char_ratio > 0.08:
        issues.append("Excessive special characters detected")
        suggestions.append("Use simple bullet points, avoid tables and text boxes")
        score -= 12
        details['formatting'] = 'Complex (may cause ATS issues)'
    else:
        details['formatting'] = 'Simple (ATS-friendly)'

    score = max(0, min(100, score))
    is_ats_friendly = score >= 70

    if score >= 85:
        overall = "Excellent - Highly ATS-friendly"
    elif score >= 70:
        overall = "Good - ATS-friendly with minor improvements possible"
    elif score >= 50:
        overall = "Fair - Needs improvement"
    else:
        overall = "Poor - Major improvements needed"

    return {
        'is_ats_friendly': is_ats_friendly,
        'score': score,
        'overall': overall,
        'issues': issues,
        'suggestions': suggestions,
        'details': details
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API is running'})


@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = file.filename.lower()
        if not (filename.endswith('.pdf') or filename.endswith('.docx')):
            return jsonify({'error': 'Invalid file format. Please upload PDF or DOCX'}), 400

        try:
            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
            else:
                resume_text = extract_text_from_docx(file)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({'error': 'Could not extract sufficient text from file.'}), 400

        ats_result = check_ats_friendliness(resume_text)

         # Add smart NLP suggestions
        predicted_role = None
        if ats_result['is_ats_friendly']:
            try:
                prediction = predictor.predict(resume_text)
                predicted_role = normalize_role(prediction['predicted_role'])
            except Exception:
                pass

            smart = generate_smart_suggestions(resume_text, predicted_role)
            ats_result['issues'] = smart['issues']
            ats_result['suggestions'] = smart['suggestions']

        response = {'ats_check': ats_result, 'resume_text_length': len(resume_text), 'resume_text': resume_text}
        if ats_result['is_ats_friendly']:
            try:
                prediction = predictor.predict(resume_text)
                raw_role = prediction['predicted_role']

                # ✅ FIX: Normalize the role before storing in session
                normalized_role = normalize_role(raw_role)

                print(f"[RESUME] Raw predicted role: '{raw_role}'")
                print(f"[RESUME] Normalized role for DB: '{normalized_role}'")

                session['predicted_job_role'] = normalized_role
                session['raw_predicted_role'] = raw_role
                session['prediction_confidence'] = prediction['confidence']

                questions = interview_questions.get(raw_role, interview_questions['DEFAULT'])

                response['analysis'] = {
                    'predicted_role': raw_role,
                    'normalized_role': normalized_role,
                    'confidence': prediction['confidence'],
                    'top_3_roles': prediction['top_3_roles'],
                    'interview_questions': questions
                }
            except Exception as e:
                print(f"Analysis error: {str(e)}")
                response['analysis_error'] = f"Could not analyze resume: {str(e)}"

        return jsonify(response), 200

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume():
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        if not resume_text:
            return jsonify({'error': 'No resume text provided'}), 400

        ats_result = check_ats_friendliness(resume_text)
        response = {'ats_check': ats_result}

        if ats_result['is_ats_friendly']:
            try:
                prediction = predictor.predict(resume_text)
                raw_role = prediction['predicted_role']

                # ✅ FIX: Normalize role
                normalized_role = normalize_role(raw_role)

                session['predicted_job_role'] = normalized_role
                session['raw_predicted_role'] = raw_role
                session['prediction_confidence'] = prediction['confidence']

                questions = interview_questions.get(raw_role, interview_questions['DEFAULT'])

                response['analysis'] = {
                    'predicted_role': raw_role,
                    'normalized_role': normalized_role,
                    'confidence': prediction['confidence'],
                    'top_3_roles': prediction['top_3_roles'],
                    'interview_questions': questions
                }
            except Exception as e:
                response['analysis_error'] = f"Could not analyze resume: {str(e)}"

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================================
# MCQ TEST ENDPOINTS
# ========================================

@app.route('/api/get-mcq-test', methods=['GET'])
def get_mcq_test():
    try:
        # ✅ FIX: Role is already normalized when stored in session
        job_role = request.args.get('role') or session.get('predicted_job_role', 'DEFAULT')

        print(f"[MCQ] Fetching questions for role: '{job_role}'")

        questions = db.get_questions_by_role(job_role, limit=10)

        # ✅ FIX: If still no questions, fall back to DEFAULT
        if not questions:
            print(f"[MCQ] No questions for '{job_role}', trying DEFAULT")
            questions = db.get_questions_by_role('DEFAULT', limit=10)

        if not questions:
            return jsonify({'error': f'No questions found for role: {job_role}'}), 404

        session['test_question_ids'] = [q['id'] for q in questions]
        session['test_start_time'] = datetime.now().isoformat()

        safe_questions = []
        for q in questions:
            safe_questions.append({
                'id': q['id'],
                'question': q['question'],
                'options': q['options'],
                'difficulty': q.get('difficulty', 'medium')
            })

        return jsonify({
            'success': True,
            'job_role': session.get('raw_predicted_role', job_role),  # Show friendly name to user
            'db_role': job_role,
            'questions': safe_questions,
            'total_questions': len(safe_questions)
        })

    except Exception as e:
        print(f"MCQ Test Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/submit-test', methods=['POST'])
def submit_test():
    try:
        data = request.get_json()
        answers = data.get('answers', {})
        
        # ✅ Read from request body first, fallback to session
        question_ids = data.get('question_ids') or session.get('test_question_ids', [])

        if not question_ids:
            return jsonify({'error': 'No active test found. Please start a new test.'}), 400

        results = []
        correct_count = 0
        total_questions = len(question_ids)

        for question_id in question_ids:
            question_data = db.get_question_by_id(question_id)
            user_answer = answers.get(str(question_id))
            correct_answer = question_data['correct_answer']
            is_correct = (user_answer == correct_answer)
            if is_correct:
                correct_count += 1

            results.append({
                'question_id': question_id,
                'question': question_data['question'],
                'options': question_data['options'],
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'explanation': question_data.get('explanation', '')
            })

        score_percentage = (correct_count / total_questions) * 100
        job_role = data.get('role', session.get('predicted_job_role', 'DEFAULT'))
        
        test_result = {
            'job_role': job_role,
            'total_questions': total_questions,
            'correct_answers': correct_count,
            'score_percentage': score_percentage,
            'timestamp': datetime.now().isoformat()
        }
        db.save_test_result(test_result)

        return jsonify({
            'success': True,
            'score': score_percentage,
            'correct_answers': correct_count,
            'total_questions': total_questions,
            'results': results,
            'passed': score_percentage >= 60
        })

    except Exception as e:
        print(f"Submit Test Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-test-history', methods=['GET'])
def get_test_history():
    try:
        history = db.get_test_history(limit=10)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ NEW: Debug endpoint to check what role was detected
@app.route('/api/debug-role', methods=['GET'])
def debug_role():
    return jsonify({
        'raw_predicted_role': session.get('raw_predicted_role', 'Not set'),
        'normalized_role': session.get('predicted_job_role', 'Not set'),
        'confidence': session.get('prediction_confidence', 'Not set')
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Resume Analysis API with ATS Checker & MCQ Tests")
    print("=" * 60)
    print("📍 API running at: http://localhost:5000")
    print("\n🔗 Endpoints:")
    print("  GET  /api/health")
    print("  POST /api/upload-resume")
    print("  POST /api/analyze-resume")
    print("  GET  /api/get-mcq-test")
    print("  POST /api/submit-test")
    print("  GET  /api/get-test-history")
    print("  GET  /api/debug-role          ← Use this to debug role issues")
    print("=" * 60)
    app.run(debug=True, port=5000, host='0.0.0.0')