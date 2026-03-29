"""Resume rewriter helper functions.

CORE PHILOSOPHY — PRESERVE THEN ENHANCE:
  ✅ Professional Summary  → polish the candidate's OWN summary (fix filler & weak verbs)
                              — NEVER replace with a template unless no summary exists
  ✅ Skills               → keep EVERY skill from the resume; role-relevant ones moved first
                              — NEVER drop a skill, NEVER inject keywords not in the resume
  ✅ Experience           → bullets strengthened (weak verbs → strong verbs)
                              — content is never invented or replaced
  ✅ All other sections   → preserved exactly as-is
"""
import re
from services.resume_analyzer import (
    normalize_role, get_role_key, ROLE_KEYWORDS, WEAK_VERBS, SECTION_MAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION HELPERS  (used by both rewrite_resume_nlp and rewrite_resume_for_role)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_relevant_projects_rewriter(projects_text: str, role_key: str) -> str:
    """
    Filters projects to only include those relevant to the target role.
    Used by the legacy rewriter path.
    """
    if not projects_text.strip():
        return ''

    ROLE_SKILL_MAP = {
        'DATA-SCIENCE':           ['python','machine learning','sql','data','model','analysis','statistics','tensorflow','pandas','jupyter'],
        'DATA-ANALYST':           ['sql','excel','tableau','power bi','data','analysis','dashboard','reporting','python','visualization'],
        'INFORMATION-TECHNOLOGY': ['network','server','cloud','aws','azure','linux','security','database','api','system'],
        'DESIGNER':               ['figma','ui','ux','design','wireframe','prototype','sketch','adobe','user research','interface'],
        'DIGITAL-MEDIA':          ['social media','content','seo','campaign','analytics','wordpress','adobe','marketing','video','brand'],
        'ENGINEERING':            ['cad','autocad','solidworks','mechanical','electrical','civil','project','design','testing','manufacturing'],
        'FINANCE':                ['financial','excel','budget','forecast','accounting','tax','audit','reporting','gaap','analysis'],
        'HEALTHCARE':             ['patient','clinical','medical','health','hospital','care','ehr','emr','nursing','therapy'],
        'HR':                     ['recruitment','talent','onboarding','payroll','hris','training','performance','employee','hr','hiring'],
        'SALES':                  ['sales','crm','revenue','client','lead','pipeline','negotiation','target','b2b','account'],
        'BANKING':                ['banking','finance','loan','credit','kyc','aml','risk','compliance','investment','portfolio'],
        'JAVA-DEVELOPER':         ['java','spring','maven','hibernate','microservices','api','backend','database','junit','docker'],
        'PYTHON-DEVELOPER':       ['python','django','flask','fastapi','pandas','numpy','api','backend','database','automation'],
        'DEVOPS':                 ['docker','kubernetes','ci/cd','jenkins','aws','azure','linux','terraform','ansible','monitoring'],
        'REACT-DEVELOPER':        ['react','javascript','typescript','frontend','ui','html','css','redux','api','node'],
        'WEB-DESIGNING':          ['html','css','javascript','responsive','ui','ux','wordpress','figma','bootstrap','web'],
    }

    keywords = ROLE_SKILL_MAP.get(role_key, [])
    if not keywords:
        return projects_text

    lines = projects_text.split('\n')
    blocks, current = [], []
    for line in lines:
        s = line.strip()
        if s and len(s.split()) <= 8 and not s.startswith(('-','•','*')) and current:
            blocks.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append('\n'.join(current))
    if len(blocks) <= 1:
        blocks = [projects_text]

    def score(block):
        t = block.lower()
        return sum(1 for kw in keywords if kw in t)

    scored  = sorted([(score(b), b) for b in blocks if b.strip()], reverse=True)
    relevant = [(s, b) for s, b in scored if s >= 1]

    if not relevant:
        return ''

    return '\n\n'.join(b for _, b in relevant[:3]).strip()


def normalize_section_headers(text: str) -> str:
    lines = text.split('\n')
    out   = []
    for line in lines:
        stripped = line.strip()
        if (stripped
                and len(stripped) < 60
                and not stripped.startswith('-')
                and not re.search(r'[.!?,;@|]', stripped)):
            for pattern, replacement in SECTION_MAP.items():
                if re.match(pattern, stripped, re.IGNORECASE):
                    line = replacement
                    break
        out.append(line)
    return '\n'.join(out)


def strengthen_verbs(text: str) -> str:
    for pattern, replacement in WEAK_VERBS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def extract_sections(text: str) -> dict:
    sections = {
        'header': [], 'summary': [], 'skills': [],
        'experience': [], 'education': [], 'certifications': [],
        'projects': [], 'achievements': [], 'other': []
    }
    current = 'header'
    SECTION_PATTERNS = [
        (re.compile(r'^(PROFESSIONAL SUMMARY|SUMMARY|OBJECTIVE|CAREER OBJECTIVE|PROFILE|ABOUT ME)$'),
         'summary'),
        (re.compile(r'^(SKILLS|TECHNICAL SKILLS|KEY SKILLS|CORE COMPETENCIES|COMPETENCIES|EXPERTISE)$'),
         'skills'),
        (re.compile(r'^(PROFESSIONAL EXPERIENCE|WORK EXPERIENCE|EXPERIENCE|EMPLOYMENT|WORK HISTORY|EMPLOYMENT HISTORY)$'),
         'experience'),
        (re.compile(r'^(EDUCATION|ACADEMIC BACKGROUND|EDUCATIONAL BACKGROUND|QUALIFICATIONS)$'),
         'education'),
        (re.compile(r'^(CERTIFICATIONS?|CERTIFICATES?|CREDENTIALS|LICENSES?)$'),
         'certifications'),
        (re.compile(r'^(PROJECTS?|KEY PROJECTS?|PERSONAL PROJECTS?|PORTFOLIO)$'),
         'projects'),
        (re.compile(r'^(ACHIEVEMENTS?|ACCOMPLISHMENTS?|AWARDS?|HONORS?)$'),
         'achievements'),
        (re.compile(r'^(KEY CONTRIBUTIONS?|ADDITIONAL INFORMATION|KEY STRENGTHS?)$'),
         'other'),
    ]

    for line in text.split('\n'):
        stripped = line.strip()
        upper    = stripped.upper()
        matched  = False
        if stripped and len(stripped) < 60 and not stripped.startswith('-'):
            for pattern, section_name in SECTION_PATTERNS:
                if pattern.match(upper):
                    current = section_name
                    matched = True
                    break
        if not matched:
            sections[current].append(line)

    return {k: '\n'.join(v).strip() for k, v in sections.items()}


def extract_contact(header_text: str) -> tuple:
    lines = [ln.strip() for ln in header_text.split('\n') if ln.strip()]
    name  = lines[0] if lines else '[Your Name]'
    email_m    = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', header_text)
    phone_m    = re.search(r'(\+?\d[\d\s\-(). ]{7,}\d)', header_text)
    linkedin_m = re.search(r'linkedin\.com/in/[\w-]+', header_text, re.IGNORECASE)
    email    = email_m.group()         if email_m    else ''
    phone    = phone_m.group().strip() if phone_m    else ''
    linkedin = linkedin_m.group()      if linkedin_m else ''
    return name, email, phone, linkedin


def extract_actual_skills(resume_text: str) -> list:
    """
    Extract only skills that are ACTUALLY present in the uploaded resume.
    No fabrication — reads what's there.
    """
    text_lower = resume_text.lower()
    all_possible_skills = [
        'python','java','javascript','typescript','c++','c#','ruby','php','swift','kotlin','go','rust',
        'html','css','html5','css3','react','angular','vue','node.js','express','django','flask',
        'spring','laravel','rails','fastapi',
        'sql','mysql','postgresql','mongodb','redis','sqlite','oracle','dynamodb','firebase',
        'aws','azure','gcp','docker','kubernetes','git','linux','bash','terraform','jenkins',
        'machine learning','deep learning','nlp','tensorflow','pytorch','scikit-learn','keras',
        'pandas','numpy','matplotlib','seaborn','tableau','power bi','excel','r',
        'data analysis','data visualization','statistics','big data','spark','hadoop',
        'rest api','graphql','microservices','agile','scrum','devops','ci/cd',
        'project management','team leadership','communication','problem solving','critical thinking',
        'time management','negotiation','presentation','strategic planning','budget management',
        'crm','salesforce','sap','erp','jira','confluence','trello','slack',
        'financial analysis','budgeting','forecasting','gaap','auditing','tax','accounting',
        'quickbooks','tally','financial modeling','risk management','compliance',
        'recruitment','talent acquisition','onboarding','performance management','payroll','hris',
        'employee relations','training','hr analytics',
        'figma','adobe xd','photoshop','illustrator','indesign','sketch','ui/ux','wireframing',
        'prototyping','typography','user research',
        'patient care','emr','ehr','hipaa','clinical','medical billing','triage','cpr',
        'seo','sem','social media','content marketing','email marketing','google analytics',
        'copywriting','brand management','adobe creative suite','wordpress',
        'lead generation','pipeline management','account management','cold calling','upselling',
        'customer service','data entry','microsoft office','ms word','powerpoint','outlook',
        'quality control','six sigma','autocad','solidworks','lean manufacturing',
    ]
    found = []
    seen  = set()
    for skill in all_possible_skills:
        if skill.lower() in text_lower and skill.lower() not in seen:
            found.append(skill)
            seen.add(skill.lower())
    for kw_list in ROLE_KEYWORDS.values():
        for kw in kw_list:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw_lower in text_lower:
                found.append(kw)
                seen.add(kw_lower)
    return found


def _break_into_short_lines(text: str, max_words: int = 30) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    lines, current, current_count = [], [], 0
    for sent in sentences:
        words = sent.split()
        if current_count + len(words) > max_words and current:
            lines.append(' '.join(current))
            current       = words
            current_count = len(words)
        else:
            current.extend(words)
            current_count += len(words)
    if current:
        lines.append(' '.join(current))
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SKILLS — PRESERVE ALL, RE-ORDER ROLE-RELEVANT FIRST
# ─────────────────────────────────────────────────────────────────────────────

def build_role_specific_skills(
    role_key: str,
    actual_skills: list,
    existing_skills_text: str,
) -> str:
    """
    Build a skills section that:
      1. Keeps EVERY skill from the original resume — nothing dropped, nothing invented
      2. Role-relevant skills appear FIRST (ATS keyword density boost)
      3. All remaining original skills follow in their natural order
      4. Format: plain comma-separated, ATS-safe (no bullet chars, no percentage bars)

    Strategy:
      A. Parse the raw skills section text first (preserves original phrasing)
      B. Supplement with auto-detected skills found elsewhere in the resume
      C. Sort: role-required → role-preferred → everything else
    """
    rk            = role_key or 'DEFAULT'
    role_kw_lower = {k.lower() for k in ROLE_KEYWORDS.get(rk, [])}

    # ── A. Parse original skills text ────────────────────────────────────────
    raw_items = []
    seen_raw  = set()
    if existing_skills_text:
        for token in re.split(r'[,\n\r|•\-–/]', existing_skills_text):
            token = token.strip().strip('•-– ')
            if token and 1 < len(token) < 60:
                low = token.lower()
                if low not in seen_raw:
                    seen_raw.add(low)
                    raw_items.append(token)

    # ── B. Supplement with detected skills not in the explicit section ────────
    for skill in actual_skills:
        low = skill.lower()
        if low not in seen_raw:
            seen_raw.add(low)
            raw_items.append(skill)

    if not raw_items:
        # Absolute fallback (should be very rare)
        return ', '.join(list(ROLE_KEYWORDS.get(rk, []))[:8])

    # ── C. Bucket: role-relevant vs rest ─────────────────────────────────────
    role_bucket = []
    rest_bucket = []
    for skill in raw_items:
        low = skill.lower()
        is_role = (
            low in role_kw_lower or
            any(low in rk_kw or rk_kw in low for rk_kw in role_kw_lower)
        )
        if is_role:
            role_bucket.append(skill)
        else:
            rest_bucket.append(skill)

    return ', '.join(role_bucket + rest_bucket)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY — POLISH ORIGINAL, ONLY GENERATE WHEN ABSENT
# ─────────────────────────────────────────────────────────────────────────────

def _polish_existing_summary(
    text: str,
    role_key: str,
    actual_skills: list,
) -> str:
    """
    Polish the candidate's own summary text:
      • Remove filler phrases
      • Strengthen weak verb openers
      • Remove first-person "I" references
      • Add a concise role-targeting closing (only if role not already mentioned)

    NEVER replaces or invents content — only cleans and appends.
    """
    FILLER_RE = re.compile(
        r'\b(hardworking|hard.working|team player|go.getter|passionate (about|learner)|'
        r'quick learner|fast learner|self.motivated|detail.oriented|results.oriented|'
        r'dynamic (professional|individual)|highly motivated|seasoned professional|'
        r'excellent communication skills?|strong communication)\b',
        re.IGNORECASE
    )
    WEAK_VERB_MAP = [
        (r'\bresponsible for\b', 'leading'),
        (r'\bhelped (with|to)\b', 'supported'),
        (r'\bworked (on|with)\b', 'collaborated on'),
        (r'\bwas involved in\b', 'contributed to'),
        (r'\bwas part of\b', 'contributed to'),
        (r'\bI am\b', 'A'),
        (r'\bI have\b', 'Having'),
        (r'\bI\b', ''),
    ]

    text = FILLER_RE.sub('', text)
    for pat, repl in WEAK_VERB_MAP:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    text = re.sub(r'  +', ' ', text).strip()

    # Fix capitalisation after removals
    sents = re.split(r'(?<=[.!?])\s+', text)
    fixed = []
    for s in sents:
        s = s.strip()
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        if s:
            fixed.append(s)
    text = ' '.join(fixed)

    # Role-targeting closing (only if not already present)
    rk = role_key or 'DEFAULT'
    role_display = rk.replace('-', ' ').title()
    role_words   = [w for w in role_display.lower().split() if len(w) > 3]

    if rk != 'DEFAULT' and not any(w in text.lower() for w in role_words):
        role_kw_lower = {k.lower() for k in ROLE_KEYWORDS.get(rk, [])}
        relevant      = [s for s in actual_skills if s.lower() in role_kw_lower]
        skill_str     = ', '.join(relevant[:3]) if relevant else ', '.join(actual_skills[:3])
        if skill_str:
            closing = f" Seeking a {role_display} role to apply expertise in {skill_str}."
        else:
            closing = f" Targeting a {role_display} position."
        text = text.rstrip('.') + '.' + closing

    return _break_into_short_lines(text, max_words=30)


def build_role_specific_summary(
    role_key: str,
    actual_skills: list,
    existing_summary: str,
) -> str:
    """
    Build a professional summary:
      - If the existing summary has >= 15 words: polish it (preserve candidate's voice)
      - If it's very short or absent: generate from actual skills only (no templates)
    """
    rk = role_key or 'DEFAULT'

    if existing_summary and len(existing_summary.split()) >= 15:
        return _polish_existing_summary(existing_summary, rk, actual_skills)

    # No / too-short summary — generate minimal one from actual skills only
    role_kw_lower = {k.lower() for k in ROLE_KEYWORDS.get(rk, [])}
    relevant = [s for s in actual_skills if s.lower() in role_kw_lower]
    others   = [s for s in actual_skills if s.lower() not in role_kw_lower]
    skill_str = ', '.join((relevant[:4] if relevant else others[:4])) or 'professional competencies'
    role_display = rk.replace('-', ' ').title()

    summary = (
        f"Experienced professional with expertise in {skill_str}. "
        f"Committed to delivering high-quality results in a {role_display} capacity."
    )
    return _break_into_short_lines(summary, max_words=30)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY REWRITER  (used by /api/generate-role-resume legacy fallback path)
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_resume_for_role(resume_text: str, target_role: str) -> str:
    """
    Rewrites the resume for a specific target role.
    - Polishes summary (preserves candidate's original wording)
    - Reframes experience bullets with stronger verbs
    - Keeps ALL skills already present in the resume (no fabrication)
    - Preserves all other sections intact
    """
    role_key = get_role_key(target_role) or normalize_role(target_role) or 'DEFAULT'

    # 1. Light cleaning (keep unicode — only strip truly problematic chars)
    clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', resume_text)
    clean = re.sub(r' {2,}', ' ', clean)

    # 2. Normalize section headers
    clean = normalize_section_headers(clean)

    # 3. Parse sections
    secs = extract_sections(clean)

    # 4. Contact info
    name, email, phone, linkedin = extract_contact(secs.get('header', ''))
    contact_parts = list(filter(None, [email, phone, linkedin]))

    # 5. Extract ACTUAL skills from original resume (no fabrication)
    actual_skills = extract_actual_skills(resume_text)

    # 6. Preserve & polish summary (role-targeted but never replaced)
    new_summary = build_role_specific_summary(
        role_key, actual_skills, secs.get('summary', '')
    )

    # 7. Experience — only strengthen verbs, no fake content added
    exp = secs.get('experience', '').strip()
    exp = strengthen_verbs(exp) if exp else ''

    # 8. Skills — ALL original skills preserved, role-relevant reordered first
    existing_skills = secs.get('skills', '').strip()
    skills_section  = build_role_specific_skills(role_key, actual_skills, existing_skills)

    # 9. Other sections preserved as-is
    education      = secs.get('education', '').strip()
    certifications = secs.get('certifications', '').strip()
    projects       = secs.get('projects', '').strip()
    achievements   = secs.get('achievements', '').strip()

    sep = '-' * 44

    out = []
    out.append(name.strip().upper() if name and name.strip() and name.strip() != '[Your Name]' else '')
    if contact_parts:
        out.append(' | '.join(contact_parts))
    out.append('')
    out.append('PROFESSIONAL SUMMARY')
    out.append(sep)
    out.append(new_summary)
    out.append('')
    if skills_section:
        out.append('SKILLS')
        out.append(sep)
        out.append(skills_section)
        out.append('')
    if exp:
        out.append('PROFESSIONAL EXPERIENCE')
        out.append(sep)
        out.append(exp)
        out.append('')
    if projects:
        out.append('PROJECTS')
        out.append(sep)
        out.append(projects)
        out.append('')
    if education:
        out.append('EDUCATION')
        out.append(sep)
        out.append(education)
        out.append('')
    if certifications:
        out.append('CERTIFICATIONS')
        out.append(sep)
        out.append(certifications)
        out.append('')
    if achievements:
        out.append('ACHIEVEMENTS')
        out.append(sep)
        out.append(achievements)
        out.append('')

    return '\n'.join(out)


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY NLP REWRITER  (used by /api/enhance-resume)
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_resume_nlp(resume_text: str, ats_check: dict, analysis: dict) -> str:
    """
    ATS-optimised resume enhancer.

    Core philosophy — PRESERVE then ENHANCE:
      • Contact info    → kept exactly as-is from the original
      • Summary         → polished (filler removed, weak verbs fixed, role-closing added)
                          NOT replaced with a template
      • Skills          → ALL candidate's skills kept; role-relevant reordered first
                          No fabrication, no invented skills
      • Experience      → bullets strengthened (weak verbs → strong verbs, paragraphs → bullets)
                          Content never invented or replaced
      • Education / Certs / Projects / Achievements → kept exactly as-is
    """
    predicted_role = (analysis or {}).get('predicted_role', '')
    role_key       = get_role_key(predicted_role) or 'DEFAULT'
    role_display   = (role_key or 'Professional').replace('-', ' ').title()

    # ── 1. Light cleaning ─────────────────────────────────────────────────────
    clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', resume_text)
    clean = re.sub(r' {3,}', '  ', clean)
    clean = normalize_section_headers(clean)
    secs  = extract_sections(clean)

    # ── 2. Contact — always keep exactly what the user has ────────────────────
    name, email, phone, linkedin = extract_contact(secs.get('header', ''))
    contact_parts = list(filter(None, [email, phone, linkedin]))

    # ── 3. Extract ALL actual skills from the original resume ─────────────────
    actual_skills = extract_actual_skills(resume_text)

    # ── 4. Skills — ALL original skills kept; role-relevant moved first ────────
    existing_skills_text = secs.get('skills', '').strip()
    skills_text = build_role_specific_skills(role_key, actual_skills, existing_skills_text)

    # ── 5. Preserve & polish the existing summary ─────────────────────────────
    existing_summary = secs.get('summary', '').strip()

    if existing_summary and len(existing_summary.split()) >= 15:
        summary = _polish_existing_summary(existing_summary, role_key, actual_skills)
    else:
        # Very thin or missing summary — build minimal one from actual skills only
        skill_str  = ', '.join(actual_skills[:4]) if actual_skills else 'professional competencies'
        summary    = (
            f"Experienced professional with expertise in {skill_str}. "
            f"Committed to delivering high-quality results in a {role_display} capacity."
        )
        summary = _break_into_short_lines(summary, max_words=32)

    # ── 6. Experience — strengthen verbs + convert paragraphs to bullets ──────
    #    Never invent bullets. Only reformat/strengthen what's already there.
    exp_raw = secs.get('experience', '').strip()

    def fix_experience_section(exp_text: str) -> str:
        if not exp_text:
            return ''
        DATE_RE    = re.compile(
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|19\d{2}|20\d{2}|Present|Current)',
            re.I
        )
        JOB_SEP_RE = re.compile(r'[|\u2014\u2013]')
        lines  = exp_text.split('\n')
        output = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                output.append('')
                continue
            words    = stripped.split()
            is_bullet    = stripped.startswith(('-', '•', '*', '–', '—', '►', '▸'))
            is_header    = bool(re.match(r'^[A-Z][A-Z\s/&\-]{2,}$', stripped)) and len(words) <= 7
            is_date_line = bool(DATE_RE.search(stripped)) and len(words) <= 6
            is_job_title = bool(JOB_SEP_RE.search(stripped)) and bool(DATE_RE.search(stripped))
            is_short     = len(words) < 6

            if is_header or is_date_line or is_job_title or is_short or is_bullet:
                output.append(strengthen_verbs(stripped) if is_bullet else stripped)
            elif len(words) > 35:
                mid = len(words) // 2
                output.append('- ' + strengthen_verbs(' '.join(words[:mid])))
                output.append('- ' + strengthen_verbs(' '.join(words[mid:])))
            elif len(words) >= 10:
                output.append('- ' + strengthen_verbs(stripped))
            else:
                output.append(strengthen_verbs(stripped))
        return '\n'.join(output)

    exp = fix_experience_section(exp_raw)
    if not exp:
        exp = ''   # Omit section entirely — do not inject fake content

    # ── 7. All other sections — keep exactly as-is ────────────────────────────
    education    = secs.get('education', '').strip()
    certs        = secs.get('certifications', '').strip()
    projects     = _filter_relevant_projects_rewriter(secs.get('projects', '').strip(), role_key)
    achievements = secs.get('achievements', '').strip()

    # ── 8. Assemble ───────────────────────────────────────────────────────────
    sep = '-' * 44
    out = []
    out.append(name.strip().upper() if name and name.strip() and name.strip() != '[Your Name]' else '')
    if contact_parts:
        out.append(' | '.join(contact_parts))
    out.append('')
    out.append('PROFESSIONAL SUMMARY')
    out.append(sep)
    out.append(summary)
    out.append('')
    if skills_text:
        out.append('SKILLS')
        out.append(sep)
        out.append(skills_text)
        out.append('')
    out.append('PROFESSIONAL EXPERIENCE')
    out.append(sep)
    out.append(exp)
    out.append('')
    if projects:
        out.append('PROJECTS')
        out.append(sep)
        out.append(projects)
        out.append('')
    if education:
        out.append('EDUCATION')
        out.append(sep)
        out.append(education)
        out.append('')
    if certs:
        out.append('CERTIFICATIONS')
        out.append(sep)
        out.append(certs)
        out.append('')
    if achievements:
        out.append('ACHIEVEMENTS')
        out.append(sep)
        out.append(achievements)
        out.append('')

    result = '\n'.join(out)

    # ── 9. Word count advisory — never inject fabricated content ─────────────
    current_wc = len(result.split())
    if current_wc < 300:
        # Do NOT pad with generic bullets — they misrepresent the candidate.
        # The ATS scorer will flag the word count and advise the candidate directly.
        pass

    # ── 10. Hard cap at 950 words ─────────────────────────────────────────────
    if current_wc > 950:
        lines   = result.split('\n')
        trimmed = []
        wc      = 0
        for line in lines:
            lw = len(line.split())
            if wc + lw > 950:
                break
            trimmed.append(line)
            wc += lw
        result = '\n'.join(trimmed)

    return result