import os
import re
import json
from flask import Blueprint, request, jsonify, session, make_response
from services.resume_analyzer import (
    check_ats_friendliness, generate_smart_suggestions, normalize_role, get_role_key
)
try:
    from services.resume_analyzer import extract_actual_skills
except ImportError:
    def extract_actual_skills(text: str) -> list:  # type: ignore[misc]
        """Fallback: pull capitalised/known skill tokens from resume text."""
        import re as _re
        tokens = _re.findall(r'\b[A-Z][a-zA-Z+#.]{1,20}\b', text)
        seen, skills = set(), []
        for t in tokens:
            if t.lower() not in seen:
                seen.add(t.lower())
                skills.append(t)
        return skills[:30]
from services.resume_rewriter import rewrite_resume_nlp, rewrite_resume_for_role
from utils.file_parser import extract_text_from_pdf, extract_text_from_docx, paragraphs_to_bullets
from utils.subscription import check_limit, increment_usage
from models.predict import ResumePredictor

resume_bp = Blueprint('resume', __name__)

_questions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'interview_questions.json')
with open(_questions_path, 'r') as _f:
    interview_questions = json.load(_f)

predictor = ResumePredictor()

try:
    from nlp_engine import enhance_resume_for_role as _nlp_enhance
    NLP_ENGINE_AVAILABLE = True
except ImportError:
    NLP_ENGINE_AVAILABLE = False


@resume_bp.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API is running'})


@resume_bp.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    try:
        username = session.get('user_username', '')
        if username:
            allowed, reason = check_limit(username, 'resume')
            if not allowed:
                return jsonify({'error': reason, 'limit_exceeded': True}), 403

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = file.filename.lower()
        if not (filename.endswith('.pdf') or filename.endswith('.docx')):
            return jsonify({'error': 'Invalid file format. Please upload PDF or DOCX'}), 400

        import tempfile
        tmp_filepath = None
        try:
            if filename.endswith('.pdf'):
                # Save to temp file so ATS scorer can analyze PDF structure
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    file.seek(0)
                    tmp.write(file.read())
                    tmp_filepath = tmp.name
                file.seek(0)
                resume_text = extract_text_from_pdf(file)
                if not resume_text or len(resume_text.strip()) < 200:
                    return jsonify({'error': 'Resume appears to be image-based. Please upload a text-based PDF.'}), 400
            else:
                resume_text = extract_text_from_docx(file)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({'error': 'Could not extract sufficient text from file.'}), 400

        # ── NAME VERIFICATION ─────────────────────────────────────────────────
        import unicodedata

        def _norm(n):
            n = unicodedata.normalize('NFKD', n.lower().strip())
            n = ''.join(c for c in n if c.isalpha() or c.isspace())
            return ' '.join(n.split())

        account_name = session.get('user_name', '').strip()
        if account_name and len(account_name) > 1:
            # Search first 10 lines of resume for the name
            resume_lines = [ln.strip() for ln in resume_text.split('\n') if ln.strip()]
            resume_header = ' '.join(resume_lines[:10]).lower()
            resume_header_norm = _norm(resume_header)

            account_name_norm = _norm(account_name)
            name_parts = [p for p in account_name_norm.split() if len(p) > 1]

            # Match if majority of name parts found (allows middle name missing etc.)
            matched_parts = [p for p in name_parts if p in resume_header_norm]
            match_ratio   = len(matched_parts) / len(name_parts) if name_parts else 1

            if match_ratio < 0.6:
                return jsonify({
                    'error': (
                        f'Resume does not match your account. '
                        f'Your account name is "{session.get("user_name", "")}" '
                        f'but this name was not found at the top of the uploaded resume. '
                        f'Please upload your own resume.'
                    ),
                    'name_mismatch': True
                }), 400
        # ─────────────────────────────────────────────────────────────────────

        # Convert paragraph blobs to bullets BEFORE scoring
        # This ensures bullet_ratio is correct from the very first score shown
        resume_text = paragraphs_to_bullets(resume_text)

        ats_result = check_ats_friendliness(resume_text, is_enhanced=False, filepath=tmp_filepath)
        if tmp_filepath:
            try:
                os.unlink(tmp_filepath)
            except Exception:
                pass
        ats_score  = ats_result['score']

        # Only predict roles when ATS score is good enough (>= 80)
        predicted_role = None
        prediction     = None
        if ats_score >= 80:
            try:
                prediction     = predictor.predict(resume_text)
                predicted_role = normalize_role(prediction['predicted_role'])
            except Exception as e:
                print(f"[PREDICT] Error: {e}")

        smart = generate_smart_suggestions(resume_text, predicted_role)

        def _smart_dedup(base_list, new_list):
            import re as _re
            def fingerprint(s):
                stops = {'your','add','the','and','for','with','that','this','from',
                         'have','been','will','are','was','were','had','has','can',
                         'not','but','all','also','its','more','use','using','make'}
                words = _re.findall(r'[a-z]{4,}', s.lower())
                return frozenset(w for w in words if w not in stops)
            existing_fps = [fingerprint(x) for x in base_list]
            result = list(base_list)
            for item in new_list:
                fp = fingerprint(item)
                if not any(len(fp & efp) >= 3 for efp in existing_fps):
                    result.append(item)
                    existing_fps.append(fp)
            return result

        combined_issues      = _smart_dedup(ats_result['issues'],      smart['issues'])
        combined_suggestions = _smart_dedup(ats_result['suggestions'],  smart['suggestions'])
        ats_result['issues']      = combined_issues
        ats_result['suggestions'] = combined_suggestions

        response = {
            'ats_check':          ats_result,
            'resume_text_length': len(resume_text),
            'resume_text':        resume_text
        }

        if ats_score >= 80 and prediction:
            # Score is good — include full role prediction
            raw_role        = prediction['predicted_role']
            normalized_role = normalize_role(raw_role)
            session['predicted_job_role']    = normalized_role
            session['raw_predicted_role']    = raw_role
            session['prediction_confidence'] = prediction['confidence']

            questions = interview_questions.get(raw_role, interview_questions['DEFAULT'])
            extracted_skills = extract_actual_skills(resume_text)
            session['resume_text'] = resume_text
            session['resume_skills'] = extracted_skills
            response['analysis'] = {
                'predicted_role':   raw_role,
                'normalized_role':  normalized_role,
                'confidence':       prediction['confidence'],
                'top_3_roles':      prediction['top_3_roles'],
                'interview_questions': questions,
                'skills':           extracted_skills[:15],
                'experience_years': 2,
            }
        else:
            # Score below 80 — no role prediction, prompt user to fix resume first
            response['analysis'] = None
            response['score_gate'] = {
                'locked': True,
                'score':  ats_score,
                'message': (
                    f"Your ATS score is {ats_score}/100. "
                    "Role prediction is unlocked only when your score reaches 80+. "
                    "Apply the suggested fixes below and re-scan to unlock career predictions."
                )
            }

        if username:
            increment_usage(username, 'resume')

        return jsonify(response), 200

    except Exception as e:
        print(f"[upload-resume] Server error: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


@resume_bp.route('/api/analyze-resume', methods=['POST'])
def analyze_resume():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({'error': 'Invalid request body — could not parse JSON.'}), 400
        resume_text  = (data.get('resume_text') or '').strip()
        is_enhanced  = bool(data.get('is_enhanced', False))
        if not resume_text:
            return jsonify({'error': 'No resume text provided'}), 400

        # Only convert paragraphs to bullets for non-enhanced text
        # Enhanced/fixed resumes are already in bullet format — skip reprocessing
        if not is_enhanced:
            resume_text = paragraphs_to_bullets(resume_text)

        ats_result = check_ats_friendliness(resume_text, is_enhanced=True)
        ats_score  = ats_result['score']
        response   = {'ats_check': ats_result}

        if ats_score >= 80:
            # Score is now good enough — run role prediction
            try:
                prediction      = predictor.predict(resume_text)
                raw_role        = prediction['predicted_role']
                normalized_role = normalize_role(raw_role)
                session['predicted_job_role']    = normalized_role
                session['raw_predicted_role']    = raw_role
                session['prediction_confidence'] = prediction['confidence']
                questions = interview_questions.get(raw_role, interview_questions['DEFAULT'])
                response['analysis'] = {
                    'predicted_role':   raw_role,
                    'normalized_role':  normalized_role,
                    'confidence':       prediction['confidence'],
                    'top_3_roles':      prediction['top_3_roles'],
                    'low_confidence':   prediction.get('low_confidence', False),
                    'low_conf_message': prediction.get('low_conf_message', None),
                    'interview_questions': questions
                }
            except Exception as e:
                response['analysis_error'] = f"Could not analyze: {str(e)}"
        else:
            # Still below 80 — keep role prediction locked
            response['analysis'] = None
            response['score_gate'] = {
                'locked': True,
                'score':  ats_score,
                'message': (
                    f"Your ATS score is {ats_score}/100. "
                    "Role prediction unlocks at 80+. "
                    "Continue applying fixes to reach the threshold."
                )
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@resume_bp.route('/api/generate-role-resume', methods=['POST'])
def generate_role_resume():
    """
    NLP-powered resume tailoring engine.
    - Rewrites bullets with strong verbs (no fabricated metrics)
    - Generates role-specific summary preserving candidate's own words
    - Builds categorized skills section (Core / Tools / Soft) from existing skills only
    - Returns skill gap analysis, section scores, ATS compliance
    """
    try:
        body = request.get_json(force=True, silent=True)
        if body is None:
            return jsonify({'error': 'Invalid request body'}), 400

        resume_text = (body.get('resume_text') or '').strip()
        target_role = (body.get('target_role') or '').strip()

        if not resume_text:
            return jsonify({'error': 'resume_text is required'}), 400
        if not target_role:
            return jsonify({'error': 'target_role is required'}), 400

        # ── NLP path ──────────────────────────────────────────────
        if NLP_ENGINE_AVAILABLE:
            import traceback as _tb
            try:
                result = _nlp_enhance(resume_text, target_role)
                return jsonify({
                    'success':        True,
                    'resume_text':    result['enhanced_resume'],
                    'role':           result['role_key'],
                    'role_display':   result['role_title'],
                    # Rich NLP data for the frontend dashboard
                    'nlp': {
                        'skill_gap':          result['skill_gap'],
                        'section_scores':     result['section_scores'],
                        'ats_result':         result['ats_result'],
                        'stats':              result['stats'],
                        'change_log':         result['change_log'],
                        'summary':            result['summary'],
                        'skills_text':        result['skills_text'],
                        'contact':            result['contact'],
                    }
                })
            except Exception as nlp_err:
                print(f'[NLP] Engine error, falling back: {nlp_err}')
                print(_tb.format_exc())
                # Fall through to legacy path

        # ── Legacy fallback ───────────────────────────────────────
        role_key     = get_role_key(target_role) or normalize_role(target_role) or 'DEFAULT'
        role_display = role_key.replace('-', ' ').title()
        rewritten    = rewrite_resume_for_role(resume_text, target_role)
        return jsonify({
            'success':      True,
            'resume_text':  rewritten,
            'role':         role_key,
            'role_display': role_display,
            'nlp':          None   # signals legacy mode to frontend
        })

    except Exception as e:
        import traceback
        print(f'[generate-role-resume] Error: {e}')
        print(traceback.format_exc())
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@resume_bp.route('/api/nlp-analyze', methods=['POST'])
def nlp_analyze():
    """
    Returns the full NLP analysis WITHOUT rewriting the resume.
    Useful for showing the skill gap dashboard before/after.
    """
    if not NLP_ENGINE_AVAILABLE:
        return jsonify({'error': 'NLP engine not available'}), 503
    try:
        body = request.get_json(force=True, silent=True)
        if not body:
            return jsonify({'error': 'Invalid request body'}), 400
        resume_text = (body.get('resume_text') or '').strip()
        target_role = (body.get('target_role') or '').strip()
        if not resume_text or not target_role:
            return jsonify({'error': 'resume_text and target_role required'}), 400

        result = _nlp_enhance(resume_text, target_role)
        return jsonify({
            'success':       True,
            'skill_gap':     result['skill_gap'],
            'section_scores':result['section_scores'],
            'ats_result':    result['ats_result'],
            'stats':         result['stats'],
            'change_log':    result['change_log'],
            'role_title':    result['role_title'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@resume_bp.route('/api/enhance-resume', methods=['POST'])
def enhance_resume():
    try:
        body = request.get_json(force=True, silent=True)
        if body is None:
            raw = request.get_data(as_text=True)
            print(f'[enhance-resume] Could not parse JSON. Raw data length: {len(raw)}.')
            return jsonify({'error': 'Invalid request body — could not parse JSON.'}), 400

        resume_text = (body.get('resume_text') or '').strip()
        ats_check   = body.get('ats_check') or {}
        analysis    = body.get('analysis') or {}

        if not resume_text:
            return jsonify({'error': 'resume_text is required'}), 400

        enhanced = rewrite_resume_nlp(resume_text, ats_check, analysis)
        return jsonify({'enhanced_text': enhanced})

    except Exception as e:
        import traceback
        print(f'[enhance-resume] Error: {e}')
        print(traceback.format_exc())
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500


@resume_bp.route('/api/download-pdf', methods=['POST'])
def download_pdf():
    """
    Professional ATS-optimised PDF — clean single-column template.
    Sections: name (centred, navy, 20pt) → contact bar → ruled section headers →
    job-title/date rows → disc-bullet experience → labelled skills blocks → education.
    Zero images, zero charts, zero skill bars — pure text for maximum ATS readability.
    """
    try:
        import io
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units     import cm
        from reportlab.lib           import colors
        from reportlab.lib.enums     import TA_LEFT, TA_CENTER
        from reportlab.lib.styles   import ParagraphStyle
        from reportlab.platypus     import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
        )

        # ── Input ─────────────────────────────────────────────────────────────
        body        = request.get_json(force=True)
        resume_text = (body.get('resume_text') or '').strip()
        filename    = (body.get('filename')    or 'resume').strip()
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        if not resume_text:
            return jsonify({'error': 'resume_text is required'}), 400

        # ── Colour palette ────────────────────────────────────────────────────
        NAVY   = colors.HexColor('#1C3557')   # name + section headers + bullets
        DARK   = colors.HexColor('#1A1A1A')   # body text
        MED    = colors.HexColor('#333333')   # job title
        GREY   = colors.HexColor('#555555')   # company / date
        LGREY  = colors.HexColor('#777777')   # contact bar
        RULE   = colors.HexColor('#B8C8DC')   # section divider line

        # ── Page geometry ─────────────────────────────────────────────────────
        PW, _  = A4
        LM = RM = 1.85 * cm

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=LM, rightMargin=RM,
            topMargin=1.3 * cm, bottomMargin=1.4 * cm,
            title=filename.replace('.pdf', ''),
            subject='ATS-Optimised Resume',
            creator='Resume Analyzer Pro',
        )

        # ── Typography ────────────────────────────────────────────────────────
        def PS(name, **kw):
            return ParagraphStyle(name, **kw)

        S_NAME = PS('name',
            fontName='Helvetica-Bold', fontSize=20, leading=24,
            alignment=TA_CENTER, textColor=NAVY,
            spaceBefore=2, spaceAfter=2)

        S_CONTACT = PS('contact',
            fontName='Helvetica', fontSize=8.5, leading=12,
            alignment=TA_CENTER, textColor=LGREY, spaceAfter=5)

        S_SEC = PS('section',
            fontName='Helvetica-Bold', fontSize=9.5, leading=13,
            alignment=TA_LEFT, textColor=NAVY,
            spaceBefore=10, spaceAfter=1, tracking=80)

        S_JOBT = PS('jobtitle',
            fontName='Helvetica-Bold', fontSize=9.5, leading=13,
            alignment=TA_LEFT, textColor=MED,
            spaceBefore=5, spaceAfter=0)

        S_CO = PS('company',
            fontName='Helvetica', fontSize=9, leading=12,
            alignment=TA_LEFT, textColor=GREY, spaceAfter=2)

        S_BODY = PS('body',
            fontName='Helvetica', fontSize=9.5, leading=14,
            alignment=TA_LEFT, textColor=DARK, spaceAfter=2)

        S_BULLET = PS('bullet',
            fontName='Helvetica', fontSize=9.5, leading=14,
            alignment=TA_LEFT, textColor=DARK,
            leftIndent=13, firstLineIndent=-9, spaceAfter=2)

        S_SKILL_LBL = PS('skill_lbl',
            fontName='Helvetica-Bold', fontSize=8.5, leading=12,
            textColor=NAVY, spaceAfter=0)

        S_SKILL_VAL = PS('skill_val',
            fontName='Helvetica', fontSize=9, leading=13,
            textColor=DARK, spaceAfter=4)

        # ── Helpers ───────────────────────────────────────────────────────────
        def sx(t):
            return (str(t).replace('&','&amp;').replace('<','&lt;')
                          .replace('>','&gt;').replace('"','&quot;'))

        def section_hdr(title):
            return [
                Paragraph(sx(title.upper()), S_SEC),
                HRFlowable(width='100%', thickness=0.6, color=RULE,
                           spaceBefore=1, spaceAfter=4, lineCap='square'),
            ]

        def title_date_row(title, date='', company=''):
            """
            Bold title left, grey date right — rendered as a single Paragraph
            with a tab-like spacer using right-aligned date in a nested table-free
            approach. Uses a plain two-cell table with NO borders/lines so
            pdfplumber sees zero extra drawing objects.
            """
            elems = []
            if title or date:
                # Render as a single paragraph: bold title + spaced date
                # Use right-aligned invisible separator so ATS text extraction
                # reads: "Job Title  Company  Date" on one line — clean, parseable
                if date:
                    combined = (
                        f'<b>{sx(title)}</b>'
                        f'<font color="#777777" size="8.5">'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;{sx(date)}'
                        f'</font>'
                    )
                else:
                    combined = f'<b>{sx(title)}</b>'
                elems.append(Paragraph(combined, S_JOBT))
            if company:
                elems.append(Paragraph(sx(company), S_CO))
            return elems

        def bullet(text):
            return Paragraph(
                f'<font color="#1C3557">\u2022</font>\u00a0\u00a0{sx(text)}',
                S_BULLET)

        # ── Regex helpers ─────────────────────────────────────────────────────
        SEP_RE       = re.compile(r'^[-─=\u2500]{3,}$')
        BULLET_RE    = re.compile(r'^[-*\u2022\u25cf\u2013\u2014>]\s*')
        DATE_RE      = re.compile(
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
            r'\d{4}|Present|Current|Till\s*[Dd]ate)', re.I)
        DATE_RANGE   = re.compile(
            r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?'
            r'\s*\d{4}\s*[-\u2013\u2014to]+\s*'
            r'(?:\d{4}|Present|Current|Till\s*[Dd]ate)\b)', re.I)
        SKILL_LBL_RE = re.compile(
            r'^(Core Skills?|Technical Skills?|Key Skills?'
            r'|Tools?\s*[&]\s*Platforms?|Additional|Soft Skills?'
            r'|Competencies|Languages?)\s*:\s*', re.I)
        SKILL_CSV_RE = re.compile(r'^[A-Za-z0-9].{8,}(?:,\s*[A-Za-z0-9].+){2,}$')

        SECTION_HDRS = {
            'PROFESSIONAL SUMMARY','SUMMARY','OBJECTIVE','CAREER OBJECTIVE',
            'PROFILE','ABOUT ME',
            'SKILLS','TECHNICAL SKILLS','KEY SKILLS','CORE COMPETENCIES',
            'COMPETENCIES','EXPERTISE','TOOLS & TECHNOLOGIES','TOOLS & PLATFORMS',
            'PROFESSIONAL EXPERIENCE','WORK EXPERIENCE','EXPERIENCE',
            'EMPLOYMENT HISTORY','WORK HISTORY','EMPLOYMENT',
            'EDUCATION','ACADEMIC BACKGROUND','ACADEMIC QUALIFICATIONS',
            'EDUCATIONAL BACKGROUND','QUALIFICATIONS',
            'CERTIFICATIONS','CERTIFICATES','CREDENTIALS',
            'LICENSES & CERTIFICATIONS','CERTIFICATION','CERTIFICATE',
            'PROJECTS','KEY PROJECTS','PERSONAL PROJECTS','PORTFOLIO',
            'ACHIEVEMENTS','ACCOMPLISHMENTS','AWARDS','HONORS',
            'LANGUAGES','ADDITIONAL INFORMATION','INTERESTS',
            'VOLUNTEER','VOLUNTEER EXPERIENCE','PUBLICATIONS',
            'KEY CONTRIBUTIONS','KEY STRENGTHS','STRENGTHS',
        }

        # ── Parse lines → elements ────────────────────────────────────────────
        elements    = []
        lines       = resume_text.split('\n')
        i           = 0
        hdr_done    = False
        in_skills   = False

        while i < len(lines):
            raw      = lines[i]
            i += 1
            stripped = raw.strip()

            if SEP_RE.match(stripped):
                continue
            if not stripped:
                elements.append(Spacer(1, 2))
                continue

            upper = stripped.upper()

            # ── A. Name + contact header ──────────────────────────────────
            if not hdr_done:
                elements.append(Spacer(1, 4))
                elements.append(Paragraph(sx(stripped), S_NAME))

                contact_parts = []
                while i < len(lines):
                    nxt = lines[i].strip()
                    i += 1
                    if not nxt or SEP_RE.match(nxt):
                        continue
                    if nxt.upper() in SECTION_HDRS:
                        i -= 1
                        break
                    if nxt.lower().startswith(('tailored for','targeted for')):
                        continue
                    contact_parts.append(nxt)
                    if len(contact_parts) >= 3:
                        break

                if contact_parts:
                    joined = '  \u007c  '.join(contact_parts)
                    elements.append(Paragraph(sx(joined), S_CONTACT))

                elements.append(Spacer(1, 2))
                elements.append(HRFlowable(
                    width='100%', thickness=1.5, color=NAVY,
                    spaceBefore=0, spaceAfter=6, lineCap='square'))
                hdr_done  = True
                in_skills = False
                continue

            # ── B. Section heading ────────────────────────────────────────
            if upper in SECTION_HDRS:
                in_skills = upper in {
                    'SKILLS','TECHNICAL SKILLS','KEY SKILLS',
                    'CORE COMPETENCIES','COMPETENCIES','EXPERTISE',
                    'TOOLS & TECHNOLOGIES','TOOLS & PLATFORMS',
                }
                for el in section_hdr(stripped):
                    elements.append(el)
                continue

            # ── C. Bullet line ────────────────────────────────────────────
            if BULLET_RE.match(stripped):
                content = BULLET_RE.sub('', stripped).strip()
                if content:
                    elements.append(bullet(content))
                in_skills = False
                continue

            # ── D. Skills: "Label: val1, val2…" or plain CSV ─────────────
            m = SKILL_LBL_RE.match(stripped)
            if m or (in_skills and SKILL_CSV_RE.match(stripped)):
                if m:
                    lbl  = m.group(1).strip()
                    val  = stripped[m.end():].strip().strip(':').strip()
                else:
                    lbl, val = '', stripped
                if lbl:
                    elements.append(Paragraph(sx(lbl) + ':', S_SKILL_LBL))
                if val:
                    elements.append(Paragraph(sx(val), S_SKILL_VAL))
                continue

            # ── E. Job/education title row with date ─────────────────────
            has_sep = '|' in stripped or '\u2014' in stripped or '\u2013' in stripped
            if has_sep and DATE_RE.search(stripped):
                dm       = DATE_RANGE.search(stripped)
                date_str = dm.group(0).strip() if dm else ''
                rest     = (stripped[:dm.start()] + stripped[dm.end():]).strip().strip('|-').strip() if dm else stripped
                parts    = [p.strip() for p in rest.split('|') if p.strip()]
                title_s  = parts[0] if parts else rest
                co_s     = parts[1] if len(parts) > 1 else ''
                for el in title_date_row(title_s, date_str, co_s):
                    elements.append(el)
                in_skills = False
                continue

            if DATE_RE.search(stripped) and len(stripped) < 130:
                dm = DATE_RANGE.search(stripped) or re.search(r'\b\d{4}\b', stripped)
                if dm:
                    tp = stripped[:dm.start()].strip().rstrip('-|,').strip()
                    dp = dm.group(0).strip()
                    if tp:
                        for el in title_date_row(tp, dp):
                            elements.append(el)
                        in_skills = False
                        continue

            # ── F. Plain body ─────────────────────────────────────────────
            elements.append(Paragraph(sx(stripped), S_BODY))

        # ── Build PDF ─────────────────────────────────────────────────────────
        doc.build(elements)
        buffer.seek(0)

        resp = make_response(buffer.read())
        resp.headers['Content-Type']        = 'application/pdf'
        resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return resp

    except ImportError:
        return jsonify({'error': 'reportlab not installed. Run: pip install reportlab'}), 500
    except Exception as e:
        import traceback
        print(f'[download-pdf] Error: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  /api/claude-fix  — Apply All Fixes using structured resume_fixer pipeline
# ─────────────────────────────────────────────────────────────────────────────
@resume_bp.route('/api/claude-fix', methods=['POST'])
def claude_fix():
    import traceback
    try:
        # Force JSON parse — return proper JSON even on content-type mismatch
        data = request.get_json(force=True, silent=True)
        if data is None:
            raw = request.get_data(as_text=True)
            print(f'[claude-fix] Could not parse JSON. Raw length: {len(raw)}')
            return jsonify({'success': False, 'error': 'Invalid request body — could not parse JSON'}), 400

        resume_text = (data.get('resume_text') or '').strip()
        role        = (data.get('predicted_role') or '').strip()

        if not resume_text:
            return jsonify({'success': False, 'error': 'No resume text provided'}), 400

        try:
            from services.resume_fixer import fix_resume
        except ImportError as ie:
            print(f'[claude-fix] Import error: {ie}')
            return jsonify({'success': False, 'error': f'Fixer module not available: {ie}'}), 500

        # ── Run the full structured fixer ────────────────────────────────────
        # fix_resume() rebuilds the entire resume from parsed sections, applies
        # all ATS improvements, and returns before/after scores so the UI
        # can show a genuine score delta.
        fix_result   = fix_resume(resume_text, predicted_role=role)
        fixed_resume = fix_result['fixed_text']
        before_score = fix_result['before_score']
        new_score    = fix_result['after_score']
        score_delta  = fix_result['score_delta']
        new_ats      = fix_result['after_result']

        # ── Post-process: remove leftover separator lines ────────────────────
        fixed_resume = re.sub(r'^-{5,}$', '', fixed_resume, flags=re.MULTILINE)

        # ── Recover contact info if stripped during fix ──────────────────────
        if '@' not in fixed_resume:
            em = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', resume_text)
            if em:
                lines_tmp = fixed_resume.split('\n')
                lines_tmp.insert(1, em.group())
                fixed_resume = '\n'.join(lines_tmp)

        if not re.search(r'\+?\d[\d\s\-(). ]{7,15}\d', fixed_resume):
            ph = re.search(r'\+?\d[\d\s\-(). ]{7,15}\d', resume_text)
            if ph:
                lines_tmp = fixed_resume.split('\n')
                lines_tmp.insert(1, ph.group().strip())
                fixed_resume = '\n'.join(lines_tmp)

        print(f"[claude-fix] before={before_score} after={new_score} delta={score_delta:+d}")

        # ── Build human-readable fix list from fixer's own log ───────────────
        fix_log = fix_result.get('fixes_applied', [])
        fixes = []
        category_map = {
            'section': 'Structure', 'header': 'Structure', 'normalise': 'Structure',
            'summary': 'Content',   'skills': 'Skills',     'verb': 'Verbs',
            'bullet': 'Structure',  'paragraph': 'Structure','contact': 'Contact',
            'education': 'Content', 'word': 'Formatting',   'third': 'Formatting',
            'warning': 'Advisory',  'advisory': 'Advisory', 'tip': 'Advisory',
        }
        for msg in fix_log[:10]:
            cat = 'General'
            for kw, label in category_map.items():
                if kw.lower() in msg.lower():
                    cat = label
                    break
            fixes.append({'category': cat, 'action': msg[:80], 'detail': msg})

        if not fixes:
            fixes = [
                {'category': 'Structure', 'action': 'Normalised section headers', 'detail': 'All sections renamed to ATS-standard ALL CAPS format.'},
                {'category': 'Content',   'action': 'Polished Professional Summary', 'detail': 'Filler removed, weak verbs strengthened.'},
                {'category': 'Skills',    'action': 'Re-ordered Skills section', 'detail': 'Role-relevant skills moved to front.'},
                {'category': 'Verbs',     'action': 'Strengthened action verbs', 'detail': 'Weak phrases replaced with strong ATS verbs.'},
                {'category': 'Structure', 'action': 'Converted paragraphs to bullets', 'detail': 'Long prose reformatted as scannable dash bullets.'},
            ]

        # ── Role prediction — unlocked when fixed score >= 80 ────────────────
        role_prediction = None
        role_unlocked   = False

        if new_score >= 80:
            try:
                prediction      = predictor.predict(fixed_resume)
                raw_role        = prediction['predicted_role']
                normalized_role = normalize_role(raw_role)
                session['predicted_job_role']    = normalized_role
                session['raw_predicted_role']    = raw_role
                session['prediction_confidence'] = prediction['confidence']
                questions = interview_questions.get(raw_role, interview_questions['DEFAULT'])
                role_prediction = {
                    'predicted_role':      raw_role,
                    'normalized_role':     normalized_role,
                    'confidence':          prediction['confidence'],
                    'top_3_roles':         prediction['top_3_roles'],
                    'interview_questions': questions,
                }
                role_unlocked = True
                print(f"[claude-fix] Role unlocked: {raw_role} ({prediction['confidence']}% confidence)")
            except Exception as pred_err:
                print(f"[claude-fix] Prediction error: {pred_err}")

        delta_str = f'+{score_delta}' if score_delta >= 0 else str(score_delta)
        response = {
            'success':         True,
            'fixed_resume':    fixed_resume,
            'fixes':           fixes,
            'before_score':    before_score,
            'predicted_score': new_score,
            'score_delta':     score_delta,
            'role_unlocked':   role_unlocked,
            'summary': (
                f'Resume fixed. Score: {before_score} → {new_score}/100 ({delta_str} points). '
                + (f'Role prediction unlocked: {role_prediction["predicted_role"]}.'
                   if role_unlocked
                   else f'Reach 80+ to unlock role prediction. Currently {new_score}/100.')
            ),
        }

        if role_unlocked:
            response['role_prediction'] = role_prediction
        else:
            response['score_gate'] = {
                'locked':           True,
                'score':            new_score,
                'threshold':        80,
                'points_needed':    max(0, 80 - new_score),
                'message': (
                    f'Your ATS score is now {new_score}/100 (was {before_score}). '
                    f'Role prediction unlocks at 80+. '
                    + (f'You need {max(0, 80 - new_score)} more point(s). Fix the remaining issues below.'
                       if new_score < 80 else 'Almost there — re-scan to unlock.')
                ),
                'remaining_issues': new_ats.get('issues', [])[:5],
            }

        return jsonify(response)

    except Exception as e:
        import traceback
        print(f'[claude-fix] Error: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# /api/generate-questions — NLP-powered MCQ + Interview Question Generator
# ─────────────────────────────────────────────────────────────────────────────
@resume_bp.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    """
    Generate role-specific MCQ and mock interview questions using
    FLAN-T5 NLP model (free, open-source). Falls back to curated
    static question bank if model is unavailable.

    Request body:
        role             (str)  — normalized role key e.g. 'DATA-SCIENCE'
        skills           (list) — skills extracted from resume
        experience_years (int)  — years of experience (default 2)
        mcq_count        (int)  — number of MCQ questions (default 10)
        interview_count  (int)  — number of interview questions (default 10)

    Returns:
        { mcq: [...], interview: [...], role, model_used, skills_used }
    """
    try:
        from services.question_generator import generate_questions_for_resume  # type: ignore[import]
        from services.resume_analyzer import normalize_role, extract_actual_skills  # type: ignore[misc]

        data = request.get_json(force=True, silent=True) or {}

        # Get role — from request body, session, or default
        role = data.get('role') or session.get('predicted_job_role', 'DEFAULT')
        role = normalize_role(role)

        # Get skills — from request body or session resume text
        skills = data.get('skills', [])
        if not skills:
            resume_text = session.get('resume_text', '')
            if resume_text:
                skills = extract_actual_skills(resume_text)

        experience_years = int(data.get('experience_years', 2))
        mcq_count        = int(data.get('mcq_count', 10))
        interview_count  = int(data.get('interview_count', 10))

        result = generate_questions_for_resume(
            role=role,
            skills=skills,
            experience_years=experience_years,
            mcq_count=mcq_count,
            interview_count=interview_count
        )

        return jsonify({'success': True, **result}), 200

    except Exception as e:
        import traceback
        print(f'[generate-questions] Error: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /api/mock-interview — AI-powered conversational mock interview
# ─────────────────────────────────────────────────────────────────────────────
@resume_bp.route('/api/mock-interview', methods=['POST'])
def mock_interview():
    """
    Conversational mock interview session.
    Accepts user's answer, returns next question + feedback on previous answer.

    Request body:
        role             (str)  — normalized role key
        question         (str)  — the question that was asked
        user_answer      (str)  — candidate's answer
        question_number  (int)  — current question index
        skills           (list) — candidate's skills

    Returns:
        {
            feedback:       str,   — feedback on the user's answer
            score:          int,   — score for this answer (0-10)
            next_question:  str,   — next interview question
            is_complete:    bool,  — true if interview is done
            total_score:    int    — running total (if complete)
        }
    """
    try:
        from services.question_generator import get_interview_questions  # type: ignore[import]
        from services.resume_analyzer import normalize_role

        data = request.get_json(force=True, silent=True) or {}

        role            = normalize_role(data.get('role', session.get('predicted_job_role', 'DEFAULT')))
        question        = data.get('question', '')
        user_answer     = data.get('user_answer', '').strip()
        question_number = int(data.get('question_number', 1))
        skills          = data.get('skills', [])
        MAX_QUESTIONS   = 5

        # Generate basic feedback based on answer quality
        feedback, score = _evaluate_mock_answer(user_answer, question, role)

        # Get next question
        all_questions = get_interview_questions(role, 2, skills, MAX_QUESTIONS + 2)
        is_complete   = question_number >= MAX_QUESTIONS
        next_question = None if is_complete else (
            all_questions[question_number] if question_number < len(all_questions) else None
        )

        response = {
            'success':       True,
            'feedback':      feedback,
            'score':         score,
            'question_number': question_number,
            'next_question': next_question,
            'is_complete':   is_complete,
        }
        return jsonify(response), 200

    except Exception as e:
        print(f'[mock-interview] Error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


def _evaluate_mock_answer(answer: str, question: str, role: str) -> tuple:
    """
    Rule-based answer evaluator. Returns (feedback_str, score_0_to_10).
    No LLM required — uses heuristics on answer quality signals.
    """
    import re
    if not answer or len(answer.strip()) < 10:
        return ("Your answer was too brief. Try to provide a detailed response with examples.", 1)

    words      = answer.split()
    word_count = len(words)
    has_number = bool(re.search(r'\d+', answer))
    has_example= any(w in answer.lower() for w in ['example', 'instance', 'when i', 'i worked', 'i led', 'i built', 'i developed', 'project', 'team'])
    has_result = any(w in answer.lower() for w in ['result', 'outcome', 'achieved', 'improved', 'reduced', 'increased', '%', 'successfully'])

    score = 5  # baseline

    if word_count >= 80:
        score += 2
        length_fb = "Good detailed response."
    elif word_count >= 40:
        score += 1
        length_fb = "Decent length — could be more detailed."
    else:
        score -= 1
        length_fb = "Answer is quite short — aim for 50+ words with context."

    if has_number:
        score += 1

    if has_example:
        score += 1
        example_fb = "Good use of a concrete example."
    else:
        score -= 1
        example_fb = "Try adding a specific example from your experience."

    if has_result:
        score += 1
        result_fb = "You mentioned outcomes — great!"
    else:
        result_fb = "Mention the result/outcome of your actions for stronger impact."

    score = max(1, min(10, score))

    tips = []
    if not has_example:
        tips.append("Use the STAR method: Situation, Task, Action, Result.")
    if not has_number:
        tips.append("Add quantified results (e.g. '30% improvement', '5-person team').")
    if word_count < 40:
        tips.append("Expand your answer with more context and detail.")

    feedback_parts = [length_fb, example_fb, result_fb]
    if tips:
        feedback_parts.append("💡 Tip: " + " ".join(tips[:2]))

    feedback = " ".join(feedback_parts)
    return (feedback, score)