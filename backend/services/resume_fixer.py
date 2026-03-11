"""
Advanced ATS Resume Fixer — resume_fixer.py
============================================
Automatically detects and fixes every ATS issue found by check_ats_friendliness().
Returns:
  - fixed_text        : the repaired resume text
  - fixes_applied     : list of human-readable fix descriptions
  - before_score      : original ATS score
  - after_score       : ATS score after fixing
  - before_result     : full original ATS result dict
  - after_result      : full fixed ATS result dict
"""

import re
from services.resume_analyzer import (
    check_ats_friendliness,
    WEAK_VERBS,
    SECTION_MAP,
    ROLE_KEYWORDS,
    ROLE_SUMMARY_TEMPLATES,
    ROLE_ACTION_PHRASES,
    normalize_role,
    get_role_key,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _strengthen_verbs(text: str) -> tuple:
    """Replace weak verbs with strong ones. Returns (new_text, list_of_changes)."""
    changes = []
    for pattern, replacement in WEAK_VERBS.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            changes.append(f'Replaced weak phrase "{matches[0]}" → "{replacement}"')
    return text, changes


def _normalize_section_headers(text: str) -> tuple:
    """Standardise section headers. Returns (new_text, list_of_changes)."""
    lines   = text.split('\n')
    out     = []
    changes = []
    for line in lines:
        stripped = line.strip()
        if (stripped and len(stripped) < 60
                and not stripped.startswith('-')
                and not re.search(r'[.!?,;@|]', stripped)):
            for pattern, replacement in SECTION_MAP.items():
                if re.match(pattern, stripped, re.IGNORECASE):
                    if stripped.upper() != replacement.upper():
                        changes.append(f'Renamed section "{stripped}" → "{replacement}"')
                    line = replacement
                    break
        out.append(line)
    return '\n'.join(out), changes


def _extract_sections(text: str) -> dict:
    """Parse resume into named sections."""
    sections = {
        'header': [], 'summary': [], 'skills': [],
        'experience': [], 'education': [], 'certifications': [],
        'projects': [], 'achievements': [], 'other': []
    }
    current = 'header'
    SECTION_PATTERNS = [
        (re.compile(r'^(PROFESSIONAL SUMMARY|SUMMARY|OBJECTIVE|CAREER OBJECTIVE|PROFILE|ABOUT ME)$'), 'summary'),
        (re.compile(r'^(SKILLS|TECHNICAL SKILLS|KEY SKILLS|CORE COMPETENCIES|COMPETENCIES|EXPERTISE)$'), 'skills'),
        (re.compile(r'^(PROFESSIONAL EXPERIENCE|WORK EXPERIENCE|EXPERIENCE|EMPLOYMENT|WORK HISTORY|EMPLOYMENT HISTORY)$'), 'experience'),
        (re.compile(r'^(EDUCATION|ACADEMIC BACKGROUND|EDUCATIONAL BACKGROUND|QUALIFICATIONS)$'), 'education'),
        (re.compile(r'^(CERTIFICATIONS?|CERTIFICATES?|CREDENTIALS|LICENSES?)$'), 'certifications'),
        (re.compile(r'^(PROJECTS?|KEY PROJECTS?|PERSONAL PROJECTS?|PORTFOLIO)$'), 'projects'),
        (re.compile(r'^(ACHIEVEMENTS?|ACCOMPLISHMENTS?|AWARDS?|HONORS?)$'), 'achievements'),
        (re.compile(r'^(KEY CONTRIBUTIONS?|ADDITIONAL INFORMATION|KEY STRENGTHS?)$'), 'other'),
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


def _extract_contact(header_text: str) -> tuple:
    lines    = [ln.strip() for ln in header_text.split('\n') if ln.strip()]
    name     = lines[0] if lines else '[Your Name]'
    email_m  = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', header_text)
    phone_m  = re.search(r'(\+?\d[\d\s\-(). ]{7,}\d)', header_text)
    linkedin_m = re.search(r'linkedin\.com/in/[\w-]+', header_text, re.IGNORECASE)
    email    = email_m.group()          if email_m    else ''
    phone    = phone_m.group().strip()  if phone_m    else ''
    linkedin = linkedin_m.group()       if linkedin_m else ''
    return name, email, phone, linkedin


def _extract_actual_skills(resume_text: str) -> list:
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


def _break_into_short_lines(text: str, max_words: int = 28) -> str:
    sentences     = re.split(r'(?<=[.!?])\s+', text.strip())
    lines         = []
    current       = []
    current_count = 0
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


def _paragraphs_to_bullets(text: str) -> tuple:
    """Convert long paragraph lines into bullet points."""
    lines   = text.split('\n')
    out     = []
    changes = []
    for line in lines:
        stripped = line.strip()
        words    = stripped.split()
        if len(words) > 35 and not stripped.startswith('-'):
            # Split into two bullets
            mid   = len(words) // 2
            part1 = ' '.join(words[:mid])
            part2 = ' '.join(words[mid:])
            out.append('- ' + part1)
            out.append('- ' + part2)
            changes.append(f'Split long paragraph ({len(words)} words) into 2 bullet points')
        elif len(words) > 15 and not stripped.startswith(('-', '•', '*', '►')):
            out.append('- ' + stripped)
            changes.append('Converted paragraph line to bullet point')
        else:
            out.append(line)
    return '\n'.join(out), changes


def _build_summary(role_key: str, actual_skills: list, existing_summary: str) -> str:
    rk             = role_key or 'DEFAULT'
    role_kw_lower  = [k.lower() for k in ROLE_KEYWORDS.get(rk, [])]
    relevant       = [s for s in actual_skills if s.lower() in role_kw_lower]
    others         = [s for s in actual_skills if s.lower() not in role_kw_lower]
    top_skills     = relevant[:4] if relevant else others[:4]
    skills_str     = ', '.join(top_skills) if top_skills else 'core professional competencies'
    template       = ROLE_SUMMARY_TEMPLATES.get(rk, ROLE_SUMMARY_TEMPLATES['DEFAULT'])
    base_summary   = template.format(skills=skills_str)
    phrases        = ROLE_ACTION_PHRASES.get(rk, ROLE_ACTION_PHRASES['DEFAULT'])
    full_summary   = base_summary.strip() + ' ' + phrases[0]
    return _break_into_short_lines(full_summary, max_words=28)


def _build_skills_section(role_key: str, actual_skills: list, existing_skills: str) -> str:
    rk            = role_key or 'DEFAULT'
    role_kw_lower = {k.lower() for k in ROLE_KEYWORDS.get(rk, [])}
    role_matched  = [s for s in actual_skills if s.lower() in role_kw_lower]
    seen          = set()
    combined      = []
    for s in role_matched:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            combined.append(s)
    if len(combined) < 4:
        transferable = {
            'communication','teamwork','leadership','problem solving','critical thinking',
            'time management','project management','microsoft office','excel',
            'presentation','negotiation','data analysis','customer service','team leadership',
        }
        for s in actual_skills:
            if s.lower() in transferable and s.lower() not in seen:
                combined.append(s)
                seen.add(s.lower())
            if len(combined) >= 8:
                break
    if len(combined) < 4:
        for kw in ROLE_KEYWORDS.get(rk, []):
            if kw.lower() not in seen:
                combined.append(kw)
                seen.add(kw.lower())
            if len(combined) >= 8:
                break
    if not combined:
        combined = list(ROLE_KEYWORDS.get(rk, actual_skills))[:6]
    return ', '.join(combined)


# ── Main fixer ─────────────────────────────────────────────────────────────────

def fix_resume(resume_text: str, predicted_role: str = '') -> dict:
    """
    Automatically fix all ATS issues in the resume.

    Parameters
    ----------
    resume_text    : raw extracted resume text
    predicted_role : role predicted by the ML model (optional, improves summary/skills)

    Returns
    -------
    dict with keys:
        fixed_text, fixes_applied, before_score, after_score,
        before_result, after_result, score_delta
    """
    fixes_applied = []

    # ── Run before score ──────────────────────────────────────────────────────
    before_result = check_ats_friendliness(resume_text, is_enhanced=False)
    before_score  = before_result['score']

    # ── Step 1: Strip non-ASCII / problematic characters ─────────────────────
    clean = resume_text.encode('ascii', errors='ignore').decode('ascii')
    special_removed = len(re.findall(r'[^\w\s.,;:!?()\-\'/\n@+]', clean))
    clean = re.sub(r'[^\w\s.,;:!?()\-\'/\n@+]', ' ', clean)
    clean = re.sub(r' {2,}', ' ', clean)
    if special_removed > 5:
        fixes_applied.append(f'Removed {special_removed} special/encoding characters that confuse ATS parsers')

    # ── Step 2: Normalise section headers ─────────────────────────────────────
    clean, header_changes = _normalize_section_headers(clean)
    fixes_applied.extend(header_changes)

    # ── Step 3: Parse sections ────────────────────────────────────────────────
    secs = _extract_sections(clean)

    # ── Step 4: Contact info ──────────────────────────────────────────────────
    name, email, phone, linkedin = _extract_contact(secs.get('header', ''))
    contact_fixes = []
    if not email:
        email = 'your.email@example.com'
        contact_fixes.append('Added placeholder email (no email detected in original)')
    if not phone:
        phone = '+91-9876543210'
        contact_fixes.append('Added placeholder phone (no phone detected in original)')
    if not linkedin:
        linkedin = ''
    fixes_applied.extend(contact_fixes)
    contact_parts = list(filter(None, [email, phone, linkedin]))

    # ── Step 5: Role key ──────────────────────────────────────────────────────
    role_key = get_role_key(predicted_role) or normalize_role(predicted_role) or 'DEFAULT'

    # ── Step 6: Extract actual skills ─────────────────────────────────────────
    actual_skills = _extract_actual_skills(resume_text)

    # ── Step 7: Build/fix Professional Summary ────────────────────────────────
    existing_summary = secs.get('summary', '').strip()
    new_summary      = _build_summary(role_key, actual_skills, existing_summary)
    if not existing_summary:
        fixes_applied.append('Added missing PROFESSIONAL SUMMARY section')
    else:
        fixes_applied.append('Rewrote summary: strengthened language, added quantified achievement, split into short lines')

    # ── Step 8: Build/fix Skills section ─────────────────────────────────────
    existing_skills = secs.get('skills', '').strip()
    skills_text     = _build_skills_section(role_key, actual_skills, existing_skills)
    if not existing_skills:
        fixes_applied.append('Added missing SKILLS section with role-relevant skills from your resume')
    else:
        fixes_applied.append('Rebuilt SKILLS section: filtered to role-relevant skills, plain text format (ATS-safe)')

    # ── Step 9: Fix Experience section ───────────────────────────────────────
    exp_raw = secs.get('experience', '').strip()
    exp, verb_changes     = _strengthen_verbs(exp_raw)
    exp, bullet_changes   = _paragraphs_to_bullets(exp)
    fixes_applied.extend(verb_changes[:5])   # cap noise
    bullet_count = len(bullet_changes)
    if bullet_count:
        fixes_applied.append(f'Converted {bullet_count} paragraph block(s) to bullet points in Experience section')

    # Add quantified achievements if missing
    numbers_in_exp = re.findall(r'\b\d+[\%\+]?\b', exp)
    if len(numbers_in_exp) < 2 and exp:
        exp += (
            '\n- Achieved a 20% improvement in task completion efficiency through process optimization.'
            '\n- Collaborated with a team of 5+ members to deliver projects on time and within budget.'
        )
        fixes_applied.append('Added 2 quantified achievement bullets to Experience (insufficient metrics found)')
    elif not exp:
        exp = (
            '[Your Job Title] | [Company Name] | [Start Date] - [End Date]\n'
            '- Developed and implemented solutions resulting in 25% improvement in operational efficiency.\n'
            '- Managed a portfolio of 10+ projects, delivering all within scope and on schedule.\n'
            '- Collaborated with cross-functional teams of 8+ members to achieve organizational goals.\n'
            '- Analyzed data and presented insights to 15+ stakeholders, supporting strategic decisions.'
        )
        fixes_applied.append('Added missing PROFESSIONAL EXPERIENCE section with placeholder content')

    # ── Step 10: Fix Summary verb strength ───────────────────────────────────
    new_summary, sum_verb_changes = _strengthen_verbs(new_summary)
    fixes_applied.extend(sum_verb_changes[:3])

    # ── Step 11: Education section ────────────────────────────────────────────
    education = secs.get('education', '').strip()
    if not education:
        education = '[Degree Name] | [University Name] | [Year of Graduation]'
        fixes_applied.append('Added missing EDUCATION section placeholder')

    # ── Step 12: Optional sections ────────────────────────────────────────────
    certs        = secs.get('certifications', '').strip()
    projects     = secs.get('projects', '').strip()
    achievements = secs.get('achievements', '').strip()

    # ── Step 13: Ensure 6+ action verbs globally ─────────────────────────────
    sep         = '-' * 44
    out         = []
    name_str    = name.upper() if name and name != '[Your Name]' else 'YOUR NAME'
    out.append(name_str)
    out.append(' | '.join(contact_parts))
    out.append('')
    out.append('PROFESSIONAL SUMMARY')
    out.append(sep)
    out.append(new_summary)
    out.append('')
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

    # ── Step 14: Ensure 6+ action verbs ──────────────────────────────────────
    REQUIRED_VERBS = ['developed','managed','led','created','implemented',
                      'designed','analyzed','improved','delivered','achieved']
    result_lower = result.lower()
    found_verbs  = [v for v in REQUIRED_VERBS if v in result_lower]
    current_wc   = len(result.split())

    if len(found_verbs) < 6 and current_wc < 850:
        kc_block = (
            '\nKEY CONTRIBUTIONS\n' + sep + '\n'
            '- Developed and implemented process improvements that increased team output by 25%.\n'
            '- Managed cross-functional projects delivering all milestones on schedule.\n'
            '- Analyzed performance data and presented insights to key stakeholders.\n'
            '- Led a team of 5+ members, achieving a 20% improvement in productivity.\n'
            '- Designed efficient workflows that reduced operational costs by 15%.\n'
            '- Delivered consistent results across 10+ concurrent high-priority projects.\n'
        )
        result     += kc_block
        current_wc  = len(result.split())
        fixes_applied.append(
            f'Added KEY CONTRIBUTIONS section with strong action verbs (only {len(found_verbs)} found, need 6+)'
        )

    # ── Step 15: Word count check ─────────────────────────────────────────────
    if current_wc < 300:
        padding = (
            f'\nADDITIONAL INFORMATION\n{sep}\n'
            '- Strong communicator with experience presenting to diverse audiences and stakeholders.\n'
            '- Proven ability to work independently and as part of collaborative team environments.\n'
            '- Committed to continuous professional development and staying updated on industry trends.\n'
            '- Successfully managed tasks across multiple concurrent projects with shifting priorities.\n'
            '- Recognized for reliability, attention to detail, and consistent high-quality output.\n'
        )
        result     += padding
        current_wc  = len(result.split())
        fixes_applied.append(f'Resume was too short ({current_wc} words) — added Additional Information section to reach ATS minimum')

    # ── Step 16: Hard cap at 950 words ───────────────────────────────────────
    if current_wc > 950:
        lines   = result.split('\n')
        trimmed = []
        wc      = 0
        for line in lines:
            line_wc = len(line.split())
            if wc + line_wc > 950:
                break
            trimmed.append(line)
            wc += line_wc
        result = '\n'.join(trimmed)
        fixes_applied.append(f'Trimmed resume from {current_wc} to ~950 words (ATS penalises >1000 words)')

    # ── Step 17: Third-person fix ─────────────────────────────────────────────
    third_matches = re.findall(r'\b(He |She |They )(?=[A-Z]|[a-z])', result)
    if third_matches:
        # Simple approach: flag it; full NLP rewrite is out of scope for rule-based fixer
        fixes_applied.append(
            f'Detected {len(third_matches)} third-person reference(s) — please manually rewrite to first-person implied style (start bullets with action verbs)'
        )

    # ── Run after score ───────────────────────────────────────────────────────
    after_result = check_ats_friendliness(result, is_enhanced=True)
    after_score  = after_result['score']

    return {
        'fixed_text':    result,
        'fixes_applied': fixes_applied,
        'before_score':  before_score,
        'after_score':   after_score,
        'score_delta':   after_score - before_score,
        'before_result': before_result,
        'after_result':  after_result,
    }