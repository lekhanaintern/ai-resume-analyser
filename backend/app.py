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
from supabase import create_client, Client
import pdfplumber   # ← add this line

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.predict import ResumePredictor
from database import Database

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app, supports_credentials=True)
app.secret_key = secrets.token_hex(16)

# ============================================================
# SUPABASE CLIENT
# ============================================================
SUPABASE_URL = "https://kxqzncqubkzxdjqkmstq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4cXpuY3F1Ymt6eGRqcWttc3RxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE2NDg3MDksImV4cCI6MjA4NzIyNDcwOX0.82b2UA_f3BPpwwvymmcXqBiBxDCvC1EYf7nvryUefPI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.pdf', '.docx']

predictor = ResumePredictor()

# MS SQL — only for question bank
db = Database(
    server='localhost\\SQLEXPRESS',
    use_windows_auth=True
)

questions_path = os.path.join(os.path.dirname(__file__), 'data', 'interview_questions.json')
with open(questions_path, 'r') as f:
    interview_questions = json.load(f)


# ============================================================
# ROLE NORMALIZATION MAP
# ============================================================
ROLE_NORMALIZATION_MAP = {
    'data scientist':        'DATA-SCIENCE',
    'data science':          'DATA-SCIENCE',
    'data-science':          'DATA-SCIENCE',
    'data_science':          'DATA-SCIENCE',
    'datascience':           'DATA-SCIENCE',
    'DATA SCIENTIST':        'DATA-SCIENCE',
    'DATA-SCIENTIST':        'DATA-SCIENCE',
    'web developer':         'WEB-DEVELOPER',
    'web development':       'WEB-DEVELOPER',
    'web-developer':         'WEB-DEVELOPER',
    'web_developer':         'WEB-DEVELOPER',
    'webdeveloper':          'WEB-DEVELOPER',
    'WEB DEVELOPER':         'WEB-DEVELOPER',
    'python developer':      'DATA-SCIENCE',
    'python':                'DATA-SCIENCE',
    'python dev':            'DATA-SCIENCE',
    'hr':                    'HR',
    'human resources':       'HR',
    'human resource':        'HR',
    'hr manager':            'HR',
    'designer':              'DESIGNER',
    'ui designer':           'DESIGNER',
    'ux designer':           'DESIGNER',
    'ui/ux designer':        'DESIGNER',
    'graphic designer':      'DESIGNER',
    'information technology': 'INFORMATION-TECHNOLOGY',
    'information-technology': 'INFORMATION-TECHNOLOGY',
    'it':                    'INFORMATION-TECHNOLOGY',
    'it professional':       'INFORMATION-TECHNOLOGY',
    'teacher':               'TEACHER',
    'educator':              'TEACHER',
    'professor':             'TEACHER',
    'instructor':            'TEACHER',
    'advocate':              'ADVOCATE',
    'lawyer':                'ADVOCATE',
    'attorney':              'ADVOCATE',
    'legal':                 'ADVOCATE',
    'business development':  'BUSINESS-DEVELOPMENT',
    'business-development':  'BUSINESS-DEVELOPMENT',
    'business developer':    'BUSINESS-DEVELOPMENT',
    'bd':                    'BUSINESS-DEVELOPMENT',
    'healthcare':            'HEALTHCARE',
    'health care':           'HEALTHCARE',
    'medical':               'HEALTHCARE',
    'doctor':                'HEALTHCARE',
    'nurse':                 'HEALTHCARE',
    'fitness':               'FITNESS',
    'fitness trainer':       'FITNESS',
    'personal trainer':      'FITNESS',
    'gym trainer':           'FITNESS',
    'agriculture':           'AGRICULTURE',
    'agriculturist':         'AGRICULTURE',
    'farmer':                'AGRICULTURE',
    'bpo':                   'BPO',
    'call center':           'BPO',
    'customer service':      'BPO',
    'sales':                 'SALES',
    'sales executive':       'SALES',
    'sales manager':         'SALES',
    'consultant':            'CONSULTANT',
    'consulting':            'CONSULTANT',
    'business consultant':   'CONSULTANT',
    'digital media':         'DIGITAL-MEDIA',
    'digital-media':         'DIGITAL-MEDIA',
    'digital marketing':     'DIGITAL-MEDIA',
    'social media':          'DIGITAL-MEDIA',
    'automobile':            'AUTOMOBILE',
    'automotive':            'AUTOMOBILE',
    'mechanic':              'AUTOMOBILE',
    'chef':                  'CHEF',
    'cook':                  'CHEF',
    'culinary':              'CHEF',
    'finance':               'FINANCE',
    'financial analyst':     'FINANCE',
    'finance manager':       'FINANCE',
    'apparel':               'APPAREL',
    'fashion':               'APPAREL',
    'fashion designer':      'APPAREL',
    'engineering':           'ENGINEERING',
    'engineer':              'ENGINEERING',
    'software engineer':     'ENGINEERING',
    'civil engineer':        'ENGINEERING',
    'mechanical engineer':   'ENGINEERING',
    'accountant':            'ACCOUNTANT',
    'accounting':            'ACCOUNTANT',
    'ca':                    'ACCOUNTANT',
    'chartered accountant':  'ACCOUNTANT',
    'construction':          'CONSTRUCTION',
    'civil':                 'CONSTRUCTION',
    'construction manager':  'CONSTRUCTION',
    'public relations':      'PUBLIC-RELATIONS',
    'public-relations':      'PUBLIC-RELATIONS',
    'pr':                    'PUBLIC-RELATIONS',
    'banking':               'BANKING',
    'bank':                  'BANKING',
    'banker':                'BANKING',
    'finance banking':       'BANKING',
    'arts':                  'ARTS',
    'artist':                'ARTS',
    'fine arts':             'ARTS',
    'aviation':              'AVIATION',
    'pilot':                 'AVIATION',
    'airline':               'AVIATION',
    'general':               'DEFAULT',
    'default':               'DEFAULT',
}

VALID_DB_ROLES = {
    'HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
    'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO',
    'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE',
    'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS',
    'BANKING', 'ARTS', 'AVIATION', 'DATA-SCIENCE', 'WEB-DEVELOPER', 'DEFAULT'
}


def normalize_role(predicted_role: str) -> str:
    if not predicted_role:
        return 'DEFAULT'
    if predicted_role in VALID_DB_ROLES:
        print(f"[ROLE] Direct match: '{predicted_role}'")
        return predicted_role
    lower = predicted_role.lower().strip()
    if lower in ROLE_NORMALIZATION_MAP:
        mapped = ROLE_NORMALIZATION_MAP[lower]
        print(f"[ROLE] Mapped: '{predicted_role}' → '{mapped}'")
        return mapped
    for key, value in ROLE_NORMALIZATION_MAP.items():
        if key in lower or lower in key:
            print(f"[ROLE] Partial match: '{predicted_role}' → '{value}'")
            return value
    upper = predicted_role.upper().replace(' ', '-')
    for valid_role in VALID_DB_ROLES:
        if valid_role in upper or upper in valid_role:
            print(f"[ROLE] Substring match: '{predicted_role}' → '{valid_role}'")
            return valid_role
    print(f"[ROLE] No match found for '{predicted_role}', falling back to DEFAULT")
    return 'DEFAULT'


def extract_text_from_pdf(file):
    text = ""

    # ATTEMPT 1: pdfplumber (handles modern/complex PDFs better)
    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            print("[PDF] Extracted successfully with pdfplumber.")
            if len(text) > 50000:
                text = text[:50000]
            return text
        print("[PDF] pdfplumber returned no text, trying PyPDF2...")
    except Exception as e:
        print(f"[PDF] pdfplumber failed: {e}, trying PyPDF2...")

    # ATTEMPT 2: PyPDF2 (fallback)
    try:
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = min(len(pdf_reader.pages), 50)
        for page_num in range(total_pages):
            try:
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"[PDF] PyPDF2 warning on page {page_num + 1}: {e}")
                continue
        if text.strip():
            print("[PDF] Extracted successfully with PyPDF2.")
            if len(text) > 50000:
                text = text[:50000]
            return text
        print("[PDF] PyPDF2 also returned no text.")
    except Exception as e:
        print(f"[PDF] PyPDF2 failed: {e}")

    raise Exception(
        "No text could be extracted from PDF. "
        "The file may be image-based or password protected. "
        "Please upload a text-based PDF."
    )

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

    word_count = len(text.split())
    if word_count < 150:
        issues.append(f"Resume is too short ({word_count} words). Aim for 300-700 words.")
        score -= 25
        details['length'] = 'Poor'
    elif word_count < 300:
        issues.append(f"Resume is slightly short ({word_count} words). Try to expand to 400+ words.")
        score -= 10
        details['length'] = 'Fair'
    elif word_count > 1000:
        issues.append(f"Resume is too long ({word_count} words). Keep it under 700 words.")
        score -= 15
        details['length'] = 'Too Long'
    else:
        details['length'] = 'Good'

    text_lower = text.lower()
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text))
    if not has_email:
        issues.append("No email address detected. Add your professional email.")
        score -= 15
    if not has_phone:
        issues.append("No phone number detected. Add your contact number.")
        score -= 10
    details['contact_info'] = 'Complete' if (has_email and has_phone) else 'Incomplete'

    sections_found = []
    sections_missing = []
    section_keywords = {
        'Experience': ['experience', 'work history', 'employment', 'professional experience'],
        'Education':  ['education', 'qualification', 'degree', 'academic', 'university', 'college'],
        'Skills':     ['skills', 'technical skills', 'competencies', 'proficiencies'],
        'Summary':    ['summary', 'objective', 'profile', 'about me']
    }
    for section, keywords in section_keywords.items():
        if any(k in text_lower for k in keywords):
            sections_found.append(section)
        else:
            sections_missing.append(section)
            if section in ['Experience', 'Education', 'Skills']:
                issues.append(f"Missing '{section}' section.")
                suggestions.append(f"Add a clearly labeled '{section}' section.")
                score -= 10
    details['sections'] = f"{len(sections_found)}/4 key sections found"

    action_verbs = ['developed', 'managed', 'led', 'created', 'implemented', 'designed',
                    'analyzed', 'improved', 'coordinated', 'achieved', 'executed',
                    'established', 'built', 'optimized', 'delivered', 'increased']
    verb_count = sum(1 for v in action_verbs if v in text_lower)
    if verb_count < 3:
        issues.append(f"Only {verb_count} action verbs found. Use stronger verbs like: developed, managed, led.")
        score -= 12
        details['action_verbs'] = 'Poor'
    elif verb_count < 6:
        suggestions.append("Add more action verbs to strengthen your resume.")
        details['action_verbs'] = 'Fair'
    else:
        details['action_verbs'] = 'Good'

    special_char_ratio = len(re.findall(r'[^\w\s.,;:!?()\-\'/\n]', text)) / max(len(text), 1)
    if special_char_ratio > 0.05:
        issues.append("Excessive special characters or symbols detected. ATS may misread these.")
        suggestions.append("Use plain text formatting. Avoid tables, text boxes, and decorative symbols.")
        score -= 15
        details['formatting'] = 'Complex (ATS risk)'
    else:
        details['formatting'] = 'Simple (ATS-friendly)'

    all_lines = [line.strip() for line in text.split('\n') if line.strip()]
    long_lines = [line for line in all_lines if len(line.split()) > 50]
    very_long_lines = [line for line in all_lines if len(line.split()) > 80]

    if very_long_lines:
        issues.append(f"{len(very_long_lines)} very long paragraph(s) detected (80+ words each). ATS systems struggle to parse dense paragraphs.")
        suggestions.append("Break long paragraphs into short bullet points (1-2 lines each).")
        score -= 25
        details['paragraphs'] = f'Poor — {len(very_long_lines)} very long paragraph(s)'
    elif long_lines:
        issues.append(f"{len(long_lines)} long paragraph(s) detected (50+ words each). Consider breaking these into bullet points.")
        suggestions.append("Use concise bullet points instead of long paragraphs.")
        score -= 15
        details['paragraphs'] = f'Fair — {len(long_lines)} long paragraph(s)'
    else:
        details['paragraphs'] = 'Good — concise lines detected'

    total_chars = len(text)
    alpha_chars = len(re.findall(r'[a-zA-Z]', text))
    alpha_ratio = alpha_chars / max(total_chars, 1)
    garbled_chars = len(re.findall(r'[^\x00-\x7F]', text))
    garbled_ratio = garbled_chars / max(total_chars, 1)
    symbol_lines = [line for line in all_lines if len(re.findall(r'[^\w\s]', line)) > len(line) * 0.3]

    if garbled_ratio > 0.05:
        issues.append(f"Garbled or non-readable characters detected ({int(garbled_ratio * 100)}% of content).")
        suggestions.append("Use a plain text-based PDF with no images, charts, or graphical elements.")
        score -= 30
        details['images_graphics'] = 'Detected — ATS cannot read image-based content'
    elif alpha_ratio < 0.55:
        issues.append("Low readable text ratio detected. Resume may contain images, charts, or heavy graphics.")
        suggestions.append("Remove images, skill bar charts, and graphics. ATS only reads plain text.")
        score -= 20
        details['images_graphics'] = 'Likely present — low text density'
    elif len(symbol_lines) > 3:
        issues.append("Multiple symbol-heavy lines detected — possibly from skill bars, charts, or icons.")
        suggestions.append("Replace graphical skill bars and icons with plain text lists.")
        score -= 15
        details['images_graphics'] = 'Possible graphical elements detected'
    else:
        details['images_graphics'] = 'None detected'

    short_lines = [line for line in all_lines if 1 <= len(line.split()) <= 4]
    short_line_ratio = len(short_lines) / max(len(all_lines), 1)
    if short_line_ratio > 0.55:
        issues.append("Possible multi-column layout detected. ATS reads columns left-to-right, mixing up content.")
        suggestions.append("Use a single-column layout for best ATS compatibility.")
        score -= 20
        details['layout'] = 'Multi-column (ATS risk)'
    else:
        details['layout'] = 'Single-column (ATS-friendly)'

    numbers = re.findall(r'\b\d+[\%\+]?\b', text)
    if len(numbers) < 2:
        issues.append("No quantified achievements found. Add numbers like '30% increase', 'team of 10'.")
        score -= 8
        details['quantification'] = 'Poor'
    else:
        details['quantification'] = f'Good — {len(numbers)} numbers/metrics found'

    encoding_issues = len(re.findall(r'[\x80-\x9F]', text))
    if encoding_issues > 10:
        issues.append("Font encoding issues detected. Use standard fonts like Arial, Calibri, or Times New Roman.")
        score -= 10
        details['fonts'] = 'Encoding issues detected'
    else:
        details['fonts'] = 'OK'

    score = max(0, min(100, score))
    is_ats_friendly = score >= 80

    if score >= 85:
        overall = "Excellent — Highly ATS-friendly"
    elif score >= 80:
        overall = "Good — ATS-friendly with minor improvements possible"
    elif score >= 60:
        overall = "Fair — Needs improvement before submitting"
    elif score >= 40:
        overall = "Poor — Major issues detected, significant rework needed"
    else:
        overall = "Very Poor — Resume will likely be rejected by ATS"

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
                if len(resume_text.strip()) < 200:
                    return jsonify({'error': 'Resume appears to be image-based. Please upload a text-based PDF.'}), 400
            elif filename.endswith('.docx'):
                resume_text = extract_text_from_docx(file)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({'error': 'Could not extract sufficient text from file.'}), 400

        ats_result = check_ats_friendliness(resume_text)

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
        job_role = request.args.get('role') or session.get('predicted_job_role', 'DEFAULT')
        print(f"[MCQ] Fetching questions for role: '{job_role}'")

        questions = db.get_questions_by_role(job_role, limit=10)

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
            'job_role': session.get('raw_predicted_role', job_role),
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

        return jsonify({
            'success': True,
            'score': score_percentage,
            'correct_answers': correct_count,
            'wrong_answers': total_questions - correct_count,
            'total_questions': total_questions,
            'results': results,
            'passed': score_percentage >= 60
        })

    except Exception as e:
        print(f"Submit Test Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ========================================
# MCQ HISTORY — fetch from Supabase
# ========================================

@app.route('/api/get-test-history', methods=['GET'])
def get_test_history():
    try:
        response = supabase.table("mcq_results") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        return jsonify({'success': True, 'history': response.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    print("  GET  /api/debug-role")
    print("=" * 60)
    app.run(debug=True, port=5000, host='0.0.0.0')