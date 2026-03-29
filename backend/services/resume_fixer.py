"""
Advanced ATS Resume Fixer — resume_fixer.py
============================================
Automatically detects and fixes every ATS issue found by check_ats_friendliness().

CORE PHILOSOPHY — PRESERVE THEN ENHANCE:
  ✅ Professional Summary  → polish the ORIGINAL (filler removal, verb strengthening,
                              role-targeting closing) — NEVER replace with a template
  ✅ Skills Section        → keep ALL original skills, re-ordered so role-relevant
                              appear FIRST — NEVER drop a skill, NEVER invent one
  ✅ Experience            → bullets strengthened (weak verbs, paragraph→bullets)
  ✅ Contact / Education / Certifications / Projects / Achievements → preserved as-is

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
    """
    Extract only skills that are ACTUALLY present in the uploaded resume.
    Returns in the order they are detected (preserving original resume skill order).
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


def _extract_skills_from_text_raw(skills_text: str) -> list:
    """
    Parse the original skills section text and return all individual skill tokens
    in their original form. Used to preserve original skill phrasing.
    """
    if not skills_text.strip():
        return []
    items = []
    seen  = set()
    for token in re.split(r'[,\n\r|•\-–/]', skills_text):
        token = token.strip().strip('•-– ')
        if token and 1 < len(token) < 60:
            low = token.lower()
            if low not in seen:
                seen.add(low)
                items.append(token)
    return items


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


# ─────────────────────────────────────────────────────────────────────────────
# PRESERVE-FIRST SUMMARY POLISHER
# ─────────────────────────────────────────────────────────────────────────────

def _polish_and_preserve_summary(
    existing_summary: str,
    role_key: str,
    actual_skills: list,
) -> tuple:
    """
    PRESERVE the candidate's own professional summary.
    Only:
      1. Remove hollow filler phrases
      2. Replace weak verb openers with stronger ones
      3. Remove first-person "I" references
      4. Optionally append a concise role-targeting closing sentence
         (only when the role isn't already mentioned)

    Returns (polished_summary, description_of_changes)
    Never replaces the summary with a template.
    Never invents new content.
    """
    FILLER_RE = re.compile(
        r'\b(hardworking|hard.working|team player|go.getter|passionate (about|learner)|'
        r'quick learner|fast learner|self.motivated|detail.oriented|results.oriented|'
        r'dynamic (professional|individual)|highly motivated|seasoned professional|'
        r'excellent communication skills?|strong communication)\b',
        re.IGNORECASE
    )
    WEAK_VERB_MAP = [
        (r'\bresponsible for\b',   'leading'),
        (r'\bhelped (with|to)\b',  'supported'),
        (r'\bworked (on|with)\b',  'collaborated on'),
        (r'\bwas involved in\b',   'contributed to'),
        (r'\bwas part of\b',       'contributed to'),
        (r'\bI am\b',              'A'),
        (r'\bI have\b',            'Having'),
        (r'\bI\b',                 ''),
    ]

    text    = existing_summary
    changes = []

    # Strip filler
    filler_removed = FILLER_RE.sub('', text)
    if filler_removed != text:
        text = filler_removed
        changes.append('Removed hollow filler phrases (team player, quick learner, etc.)')

    # Strengthen verbs
    for pat, repl in WEAK_VERB_MAP:
        new = re.sub(pat, repl, text, flags=re.IGNORECASE)
        if new != text:
            text = new
            changes.append(f'Strengthened weak verb phrase → "{repl}"')

    # Clean up double-spaces and fix capitalisation
    text = re.sub(r'  +', ' ', text).strip()
    sents = re.split(r'(?<=[.!?])\s+', text)
    fixed = []
    for s in sents:
        s = s.strip()
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        if s:
            fixed.append(s)
    text = ' '.join(fixed)

    # Add role-targeting closing ONLY if role isn't already mentioned
    rk = role_key or 'DEFAULT'
    role_display = rk.replace('-', ' ').title()
    role_words   = [w for w in role_display.lower().split() if len(w) > 3]

    if rk != 'DEFAULT' and not any(w in text.lower() for w in role_words):
        # Use only the top 3 actual skills the candidate has for this role
        role_kw_lower = {k.lower() for k in ROLE_KEYWORDS.get(rk, [])}
        relevant = [s for s in actual_skills if s.lower() in role_kw_lower]
        skill_str = ', '.join(relevant[:3]) if relevant else ', '.join(actual_skills[:3])
        if skill_str:
            closing = f" Seeking a {role_display} role to apply expertise in {skill_str}."
        else:
            closing = f" Targeting a {role_display} position."
        text = text.rstrip('.') + '.' + closing
        changes.append(f'Added concise role-targeting closing for {role_display}')

    # Break into ATS-safe line lengths
    polished = _break_into_short_lines(text, max_words=28)
    return polished, changes


def _generate_minimal_summary_from_skills(
    role_key: str,
    actual_skills: list,
) -> str:
    """
    Only called when the resume has NO summary at all.
    Builds a brief summary from the candidate's ACTUAL detected skills only.
    Never uses role templates that might invent skills.
    """
    rk = role_key or 'DEFAULT'
    role_kw_lower = {k.lower() for k in ROLE_KEYWORDS.get(rk, [])}
    relevant = [s for s in actual_skills if s.lower() in role_kw_lower]
    others   = [s for s in actual_skills if s.lower() not in role_kw_lower]

    skill_sample = (relevant[:4] if relevant else others[:4])
    skill_str    = ', '.join(skill_sample) if skill_sample else 'professional competencies'
    role_display = rk.replace('-', ' ').title()

    summary = (
        f"Experienced professional with expertise in {skill_str}. "
        f"Committed to delivering high-quality results and contributing meaningfully "
        f"in a {role_display} capacity."
    )
    return _break_into_short_lines(summary, max_words=28)


# ─────────────────────────────────────────────────────────────────────────────
# PRESERVE-FIRST SKILLS BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_ordered_skills_section(
    role_key: str,
    actual_skills: list,
    existing_skills_text: str,
) -> tuple:
    """
    Build a skills section that:
      1. Keeps EVERY skill from the original resume — nothing is dropped
      2. Role-relevant skills appear FIRST (ATS sees them immediately)
      3. Remaining skills follow in their natural order
      4. Format: plain comma-separated, ATS-friendly

    Returns (skills_string, changes_description)
    """
    rk = role_key or 'DEFAULT'
    role_kw_lower = {k.lower() for k in ROLE_KEYWORDS.get(rk, [])}

    # ── Collect ALL original skills in their original form ──────────────────
    # Priority: raw skills section text → then auto-detected skills
    raw_original = _extract_skills_from_text_raw(existing_skills_text)

    # Merge: use raw_original as the canonical source (preserves original phrasing)
    # then supplement with any auto-detected skills not already present
    seen_lower = set()
    canonical  = []
    for skill in raw_original:
        low = skill.lower()
        if low not in seen_lower:
            seen_lower.add(low)
            canonical.append(skill)

    # Add detected skills that weren't in the explicit skills section
    # (e.g. skills mentioned only in the experience section)
    for skill in actual_skills:
        low = skill.lower()
        if low not in seen_lower:
            seen_lower.add(low)
            canonical.append(skill)

    if not canonical:
        # Absolute fallback — only role keywords as placeholders
        role_kws = list(ROLE_KEYWORDS.get(rk, []))[:8]
        return ', '.join(role_kws), ['Added role-relevant skill placeholders (no skills detected in original)']

    # ── Split into role-relevant and rest ────────────────────────────────────
    role_first = []
    rest       = []
    for skill in canonical:
        low = skill.lower()
        # Direct match OR substring match against role keywords
        is_role = (
            low in role_kw_lower or
            any(low in rk_kw or rk_kw in low for rk_kw in role_kw_lower)
        )
        if is_role:
            role_first.append(skill)
        else:
            rest.append(skill)

    ordered   = role_first + rest
    result    = ', '.join(ordered)
    changes   = []

    if role_first:
        changes.append(
            f'Re-ordered skills: {len(role_first)} role-relevant skill(s) moved to the front '
            f'({", ".join(role_first[:3])}{"..." if len(role_first) > 3 else ""}). '
            f'All {len(ordered)} original skills preserved.'
        )
    else:
        changes.append(f'Skills section preserved with all {len(ordered)} original skills (ATS plain-text format).')

    return result, changes


# ─────────────────────────────────────────────────────────────────────────────
# PARAGRAPHS → BULLETS
# ─────────────────────────────────────────────────────────────────────────────

def _paragraphs_to_bullets(text: str) -> tuple:
    """Convert long paragraph lines into bullet points."""
    lines   = text.split('\n')
    out     = []
    changes = []
    for line in lines:
        stripped = line.strip()
        words    = stripped.split()
        if len(words) > 35 and not stripped.startswith('-'):
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FIXER
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── Step 1: Strip only truly problematic characters ──────────────────────
    # KEEP: @ + | & % # " — all appear legitimately in resumes:
    #   |  → contact separator  (email | phone | linkedin)
    #   %  → achievements       (improved by 30%)
    #   &  → section names      (Tools & Platforms)
    #   #  → skills             (C#, F#)
    #   "  → quotes in summaries
    # REMOVE: box-drawing, emojis, null bytes, other non-ASCII decoratives.
    clean = resume_text.encode('ascii', errors='ignore').decode('ascii')
    special_removed = len(re.findall(r'[^\w\s.,;:!?()\-\'\/\n@+|&%#"]', clean))
    clean = re.sub(r'[^\w\s.,;:!?()\-\'\/\n@+|&%#"]', ' ', clean)
    clean = re.sub(r' {2,}', ' ', clean)
    if special_removed > 5:
        fixes_applied.append(
            f'Removed {special_removed} special/encoding characters that confuse ATS parsers'
        )

    # ── Step 2: Normalise section headers ─────────────────────────────────────
    clean, header_changes = _normalize_section_headers(clean)
    fixes_applied.extend(header_changes)

    # ── Step 3: Parse sections ────────────────────────────────────────────────
    secs = _extract_sections(clean)

    # ── Step 4: Contact info ──────────────────────────────────────────────────
    name, email, phone, linkedin = _extract_contact(secs.get('header', ''))
    contact_fixes = []
    if not email:
        # Do NOT inject a fake email — missing contact is flagged to the user
        contact_fixes.append(
            'Warning: No email address detected. Please add your professional email to the resume.'
        )
    if not phone:
        contact_fixes.append(
            'Warning: No phone number detected. Please add your contact number to the resume.'
        )
    fixes_applied.extend(contact_fixes)
    contact_parts = list(filter(None, [email, phone, linkedin]))

    # ── Step 5: Role key ──────────────────────────────────────────────────────
    role_key = get_role_key(predicted_role) or normalize_role(predicted_role) or 'DEFAULT'

    # ── Step 6: Extract ALL actual skills (from the original, unchanged text) ─
    actual_skills = _extract_actual_skills(resume_text)

    # ── Step 7: PRESERVE & POLISH Professional Summary ───────────────────────
    existing_summary = secs.get('summary', '').strip()

    if existing_summary and len(existing_summary.split()) >= 10:
        # Candidate has a summary — polish it, never replace it
        new_summary, sum_changes = _polish_and_preserve_summary(
            existing_summary, role_key, actual_skills
        )
        if sum_changes:
            fixes_applied.append(
                'Polished Professional Summary: ' + '; '.join(sum_changes)
            )
        else:
            fixes_applied.append('Professional Summary preserved as-is (already strong).')
    else:
        # No summary at all — generate minimal one from actual skills only
        new_summary = _generate_minimal_summary_from_skills(role_key, actual_skills)
        fixes_applied.append(
            'Added missing PROFESSIONAL SUMMARY using skills detected in your resume'
        )

    # ── Step 8: PRESERVE ALL Skills, re-ordered role-relevant first ──────────
    existing_skills_text = secs.get('skills', '').strip()
    skills_text, skill_changes = _build_ordered_skills_section(
        role_key, actual_skills, existing_skills_text
    )
    fixes_applied.extend(skill_changes)

    # ── Step 9: Fix Experience section ───────────────────────────────────────
    exp_raw = secs.get('experience', '').strip()
    exp, verb_changes    = _strengthen_verbs(exp_raw)
    exp, bullet_changes  = _paragraphs_to_bullets(exp)
    fixes_applied.extend(verb_changes[:5])   # cap noise
    bullet_count = len(bullet_changes)
    if bullet_count:
        fixes_applied.append(
            f'Converted {bullet_count} paragraph block(s) to bullet points in Experience section'
        )

    # Flag if quantified achievements are low — do NOT invent them
    numbers_in_exp = re.findall(r'\b\d+[\%\+]?\b', exp)
    if len(numbers_in_exp) < 2 and exp:
        fixes_applied.append(
            'Tip: Few quantified achievements detected in experience. '
            'Adding numbers (e.g. "Improved efficiency by 20%", "Managed a team of 8") '
            'significantly boosts your ATS score.'
        )
    elif not exp:
        # Do NOT inject fake experience bullets — flag it to the user
        fixes_applied.append(
            'Warning: No PROFESSIONAL EXPERIENCE section detected. '
            'Please add your work history with job title, company, dates and bullet-point responsibilities.'
        )

    # ── Step 10: Strengthen verbs in summary too ──────────────────────────────
    new_summary, sum_verb_changes = _strengthen_verbs(new_summary)
    fixes_applied.extend(sum_verb_changes[:3])

    # ── Step 11: Education section ────────────────────────────────────────────
    education = secs.get('education', '').strip()
    if not education:
        # Do NOT inject a fake education placeholder
        fixes_applied.append(
            'Warning: No EDUCATION section detected. '
            'Please add your degree, institution name, and graduation year.'
        )

    # ── Step 12: Optional sections (preserved as-is) ─────────────────────────
    certs        = secs.get('certifications', '').strip()
    projects     = secs.get('projects', '').strip()
    achievements = secs.get('achievements', '').strip()

    # ── Step 13: Assemble ─────────────────────────────────────────────────────
    sep      = '-' * 44
    out      = []
    name_str = name.strip().upper() if name and name.strip() and name.strip() != '[Your Name]' else ''
    if name_str:
        out.append(name_str)
    if contact_parts:
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

    # ── Step 14: Word count advisory ─────────────────────────────────────────
    # NEVER pad with generic bullets — fabricated content misrepresents the
    # candidate. Instead, flag short resumes so the person can add their own content.
    current_wc = len(result.split())
    if current_wc < 300:
        fixes_applied.append(
            f'Advisory: Resume is short ({current_wc} words). '
            'ATS systems score best with 300–700 words. '
            'Expand your experience bullets, add more skill detail, or include a Projects section.'
        )

    # ── Step 15: Hard cap at 950 words ────────────────────────────────────────
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
        fixes_applied.append(
            f'Trimmed resume from {current_wc} to ~950 words (ATS penalises >1000 words)'
        )

    # ── Step 16: Third-person fix flag ────────────────────────────────────────
    third_matches = re.findall(r'\b(He |She |They )(?=[A-Z]|[a-z])', result)
    if third_matches:
        fixes_applied.append(
            f'Detected {len(third_matches)} third-person reference(s) — '
            'manually rewrite to first-person implied style (start bullets with action verbs)'
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