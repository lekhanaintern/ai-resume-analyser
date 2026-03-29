"""
ATS scoring, smart suggestions, role normalization, and all role constants.
"""
import re

# ============================================================
# ROLE NORMALIZATION
# ============================================================
ROLE_NORMALIZATION_MAP = {
    'data scientist': 'DATA-SCIENCE', 'data science': 'DATA-SCIENCE',
    'data-science': 'DATA-SCIENCE', 'data_science': 'DATA-SCIENCE',
    'datascience': 'DATA-SCIENCE', 'DATA SCIENTIST': 'DATA-SCIENCE',
    'web developer': 'WEB-DEVELOPER', 'web development': 'WEB-DEVELOPER',
    'web-developer': 'WEB-DEVELOPER', 'webdeveloper': 'WEB-DEVELOPER',
    'hr': 'HR', 'human resources': 'HR', 'human resource': 'HR',
    'designer': 'DESIGNER', 'ui designer': 'DESIGNER', 'ux designer': 'DESIGNER',
    'information technology': 'INFORMATION-TECHNOLOGY', 'it': 'INFORMATION-TECHNOLOGY',
    'teacher': 'TEACHER', 'educator': 'TEACHER', 'professor': 'TEACHER',
    'advocate': 'ADVOCATE', 'lawyer': 'ADVOCATE', 'attorney': 'ADVOCATE',
    'business development': 'BUSINESS-DEVELOPMENT', 'bd': 'BUSINESS-DEVELOPMENT',
    'healthcare': 'HEALTHCARE', 'medical': 'HEALTHCARE', 'doctor': 'HEALTHCARE',
    'fitness': 'FITNESS', 'fitness trainer': 'FITNESS', 'personal trainer': 'FITNESS',
    'agriculture': 'AGRICULTURE', 'bpo': 'BPO', 'call center': 'BPO',
    'sales': 'SALES', 'sales executive': 'SALES',
    'consultant': 'CONSULTANT', 'consulting': 'CONSULTANT',
    'digital media': 'DIGITAL-MEDIA', 'digital marketing': 'DIGITAL-MEDIA',
    'automobile': 'AUTOMOBILE', 'mechanic': 'AUTOMOBILE',
    'chef': 'CHEF', 'cook': 'CHEF', 'culinary': 'CHEF',
    'finance': 'FINANCE', 'financial analyst': 'FINANCE',
    'apparel': 'APPAREL', 'fashion': 'APPAREL',
    'engineering': 'ENGINEERING', 'engineer': 'ENGINEERING',
    'accountant': 'ACCOUNTANT', 'accounting': 'ACCOUNTANT', 'ca': 'ACCOUNTANT',
    'construction': 'CONSTRUCTION', 'civil': 'CONSTRUCTION',
    'public relations': 'PUBLIC-RELATIONS', 'pr': 'PUBLIC-RELATIONS',
    'banking': 'BANKING', 'bank': 'BANKING', 'banker': 'BANKING',
    'arts': 'ARTS', 'artist': 'ARTS',
    'aviation': 'AVIATION', 'pilot': 'AVIATION',
    'general': 'DEFAULT', 'default': 'DEFAULT',
}

VALID_DB_ROLES = {
    'HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
    'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO',
    'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE',
    'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS',
    'BANKING', 'ARTS', 'AVIATION', 'DATA-SCIENCE', 'WEB-DEVELOPER', 'DEFAULT'
}



ROLE_KEYWORDS = {
    "DATA-SCIENCE":            ["Python","Machine Learning","SQL","TensorFlow","Pandas","NumPy",
                                "Data Analysis","Scikit-learn","Statistics","Deep Learning",
                                "Data Visualization","Jupyter","Feature Engineering","NLP","Spark"],
    "WEB-DEVELOPER":           ["HTML5","CSS3","JavaScript","React.js","Node.js","REST API","Git",
                                "Responsive Design","TypeScript","MongoDB","Bootstrap","Express.js",
                                "Webpack","Redux","jQuery"],
    "SOFTWARE-ENGINEER":       ["Python","Java","C++","Git","Agile","Docker","Kubernetes","REST API",
                                "System Design","Data Structures","Algorithms","SQL","CI/CD","Linux","OOP"],
    "HR":                      ["Talent Acquisition","Onboarding","HRIS","Performance Management",
                                "Employee Relations","Payroll","Recruitment","Training","HR Analytics",
                                "Conflict Resolution","ATS","Compensation","Benefits Administration"],
    "DESIGNER":                ["Figma","Adobe XD","UI/UX Design","Wireframing","Prototyping",
                                "Typography","User Research","Design Systems","Accessibility",
                                "Adobe Photoshop","Illustrator","InVision","Style Guide"],
    "ENGINEERING":             ["CAD","AutoCAD","SolidWorks","Project Management","Quality Control",
                                "Six Sigma","ISO Standards","Technical Documentation","FMEA",
                                "Root Cause Analysis","Manufacturing","Process Improvement"],
    "FINANCE":                 ["Financial Analysis","Excel","Budgeting","Forecasting","GAAP",
                                "Variance Analysis","Financial Modeling","DCF","P&L","Balance Sheet",
                                "Tableau","Power BI","Risk Assessment","Compliance"],
    "HEALTHCARE":              ["Patient Care","EMR/EHR","HIPAA Compliance","Clinical Procedures",
                                "Medical Records","Triage","CPR/BLS","Care Coordination",
                                "ICD-10","Medical Billing","Vital Signs","Medication Administration"],
    "SALES":                   ["CRM","Salesforce","Lead Generation","Pipeline Management",
                                "Revenue Growth","Client Retention","Negotiation","Cold Outreach",
                                "Quota Achievement","Account Management","B2B","B2C","Upselling"],
    "INFORMATION-TECHNOLOGY":  ["Network Administration","Cybersecurity","Cloud Computing","AWS",
                                "Azure","Linux","Active Directory","ITIL","Troubleshooting",
                                "Virtualization","VMware","Helpdesk","TCP/IP","Firewall"],
    "ACCOUNTANT":              ["GAAP","Tax Preparation","Auditing","QuickBooks","Tally",
                                "Financial Reporting","GST","TDS","Reconciliation","Cost Accounting",
                                "ERP","SAP","Accounts Payable","Accounts Receivable"],
    "TEACHER":                 ["Lesson Planning","Curriculum Development","Classroom Management",
                                "Student Assessment","Differentiated Instruction","EdTech","LMS",
                                "Google Classroom","Parent Communication","Rubrics"],
    "BANKING":                 ["KYC","AML","Risk Management","Financial Products","Regulatory Compliance",
                                "Credit Analysis","SWIFT","Trade Finance","Basel III","Treasury",
                                "Loan Processing","Customer Due Diligence"],
    "CHEF":                    ["Mise en Place","HACCP","Food Safety","Menu Planning","Kitchen Management",
                                "Culinary Arts","Inventory Management","Cost Control","Food Costing",
                                "Team Leadership","Plating","Dietary Requirements"],
    "ADVOCATE":                ["Legal Research","Litigation","Contract Drafting","Due Diligence",
                                "Client Counseling","Case Management","Legal Writing","Court Proceedings",
                                "Negotiation","Compliance","Arbitration"],
    "FITNESS":                 ["Personal Training","Nutrition Counseling","HIIT","Exercise Programming",
                                "Client Assessment","Workout Planning","Fitness Assessment","CPR/AED",
                                "Group Fitness","Strength Training","Injury Prevention"],
    "DIGITAL-MEDIA":           ["Content Strategy","SEO","Social Media Marketing","Google Analytics",
                                "Copywriting","Brand Management","Adobe Creative Suite","Email Marketing",
                                "Campaign Management","Content Creation","WordPress"],
    "BUSINESS-DEVELOPMENT":    ["Market Research","Lead Generation","Partnership Development",
                                "Revenue Growth","Client Acquisition","CRM","Negotiation",
                                "Strategic Planning","Proposal Writing","KPI Tracking"],
}


WEAK_VERBS = {
    r'\bresponsible for\b':  'Led',
    r'\bhelped to\b':        'Supported',
    r'\bhelped with\b':      'Supported',
    r'\bhelped\b':           'Supported',
    r'\bworked on\b':        'Developed',
    r'\bworked with\b':      'Collaborated with',
    r'\bwas involved in\b':  'Contributed to',
    r'\bdid\b':              'Executed',
    r'\bmade\b':             'Created',
    r'\bhandled\b':          'Managed',
    r'\bassisted in\b':      'Supported',
    r'\bworked under\b':     'Collaborated under',
    r'\bparticipated in\b':  'Contributed to',
    r'\bwas part of\b':      'Contributed to',
    r'\btried to\b':         'Worked to',
    r'\battempted to\b':     'Worked to',
    r'\bdid work on\b':      'Delivered',
}


SECTION_MAP = {
    r'^(objective|career objective|professional objective)$':              'PROFESSIONAL SUMMARY',
    r'^(summary|professional summary|profile|about me)$':                 'PROFESSIONAL SUMMARY',
    r'^(skills|technical skills|key skills|core competencies|expertise)$':'SKILLS',
    r'^(experience|work experience|professional experience|employment|work history)$':'PROFESSIONAL EXPERIENCE',
    r'^(education|academic|educational background|academic background)$':  'EDUCATION',
    r'^(certifications?|certificates?|credentials)$':                     'CERTIFICATIONS',
    r'^(projects?|key projects?|personal projects?)$':                    'PROJECTS',
    r'^(achievements?|accomplishments?|awards?)$':                        'ACHIEVEMENTS',
    r'^(languages?)$':                                                    'LANGUAGES',
}


ROLE_SUMMARY_TEMPLATES = {
    "DATA-SCIENCE": (
        "Results-driven data professional with hands-on experience in {skills}. "
        "Passionate about transforming raw data into actionable insights through statistical analysis, "
        "machine learning, and data visualization. Proven ability to build and deploy models that drive "
        "business decisions. Seeking to leverage analytical expertise as a Data Scientist."
    ),
    "WEB-DEVELOPER": (
        "Creative and detail-oriented web developer skilled in {skills}. "
        "Experienced in building responsive, user-friendly web applications from concept to deployment. "
        "Strong understanding of front-end and back-end development principles with a focus on "
        "performance and clean code. Looking to contribute as a Web Developer."
    ),
    "SOFTWARE-ENGINEER": (
        "Dedicated software engineer with expertise in {skills}. "
        "Adept at designing scalable systems, writing clean maintainable code, and solving complex "
        "technical challenges. Experienced in full software development lifecycle from requirements "
        "gathering to deployment. Eager to contribute as a Software Engineer."
    ),
    "HR": (
        "People-focused HR professional experienced in {skills}. "
        "Skilled at attracting top talent, fostering employee engagement, and building positive "
        "workplace cultures. Strong interpersonal and organizational skills with a track record of "
        "supporting organizational growth through effective human resource practices."
    ),
    "DESIGNER": (
        "Creative designer with a strong eye for aesthetics and user experience, skilled in {skills}. "
        "Passionate about crafting intuitive, visually compelling designs that solve real user problems. "
        "Experienced in translating briefs into polished, functional design solutions."
    ),
    "FINANCE": (
        "Detail-oriented finance professional with expertise in {skills}. "
        "Skilled at financial modeling, budgeting, and analysis to support strategic business decisions. "
        "Proven ability to interpret complex financial data and communicate insights to stakeholders."
    ),
    "HEALTHCARE": (
        "Compassionate healthcare professional with experience in {skills}. "
        "Committed to delivering high-quality patient care and maintaining the highest standards of "
        "clinical excellence. Strong ability to work under pressure in fast-paced medical environments."
    ),
    "SALES": (
        "Target-driven sales professional with a strong track record in {skills}. "
        "Skilled at building lasting client relationships, identifying opportunities, and consistently "
        "exceeding revenue targets. Passionate about delivering value to customers and driving growth."
    ),
    "INFORMATION-TECHNOLOGY": (
        "Skilled IT professional with experience in {skills}. "
        "Proven ability to manage infrastructure, troubleshoot complex issues, and implement secure, "
        "scalable technology solutions. Committed to keeping systems running smoothly and efficiently."
    ),
    "ENGINEERING": (
        "Methodical engineering professional with expertise in {skills}. "
        "Experienced in designing, testing, and improving engineering solutions with a focus on quality "
        "and precision. Strong analytical mindset with a commitment to technical excellence."
    ),
    "ACCOUNTANT": (
        "Meticulous accounting professional with expertise in {skills}. "
        "Skilled in maintaining accurate financial records, preparing reports, and ensuring regulatory "
        "compliance. Committed to supporting business financial health through precise accounting practices."
    ),
    "TEACHER": (
        "Dedicated educator skilled in {skills}. "
        "Passionate about creating engaging learning experiences that inspire students and support "
        "academic achievement. Strong classroom management and communication skills."
    ),
    "BANKING": (
        "Banking professional with expertise in {skills}. "
        "Experienced in financial products, customer relationship management, and regulatory compliance. "
        "Committed to delivering excellent service and supporting clients' financial goals."
    ),
    "BUSINESS-DEVELOPMENT": (
        "Strategic business development professional experienced in {skills}. "
        "Skilled at identifying growth opportunities, forging partnerships, and driving revenue. "
        "Strong negotiation and relationship-building skills with a results-oriented mindset."
    ),
    "DIGITAL-MEDIA": (
        "Creative digital media professional with expertise in {skills}. "
        "Experienced in crafting compelling content strategies and managing multi-channel campaigns "
        "that grow brand presence and engage target audiences."
    ),
    "DEFAULT": (
        "Motivated professional with experience in {skills}. "
        "Proven ability to deliver high-quality work, collaborate across teams, and drive measurable "
        "results. Adaptable and eager to contribute meaningfully in a dynamic environment."
    ),
}


def get_role_key(predicted_role: str) -> str:
    r = (predicted_role or '').upper().replace(' ', '-').replace('/', '-')
    if r in ROLE_KEYWORDS:
        return r
    for key in ROLE_KEYWORDS:
        if key in r or r in key:
            return key
    return None


def normalize_section_headers(text: str) -> str:
    lines = text.split('\n')
    out   = []
    for line in lines:
        stripped = line.strip()
        # Only attempt renaming on short, standalone lines with no bullet/sentence chars
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

    # Strict section header patterns — must be a SHORT standalone line (< 60 chars),
    # no bullet prefix, no sentence punctuation. This prevents content lines like
    # "- Managed cross-functional projects" from accidentally switching sections.
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
        # Absorb our own injected sections into 'other' so they don't contaminate
        (re.compile(r'^(KEY CONTRIBUTIONS?|ADDITIONAL INFORMATION|KEY STRENGTHS?)$'),
         'other'),
    ]

    for line in text.split('\n'):
        stripped = line.strip()
        upper    = stripped.upper()

        # Only try section matching on short lines without bullet/sentence structure
        matched = False
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

    # Large pool of possible skills to detect
    all_possible_skills = [
        # Tech
        'python','java','javascript','typescript','c++','c#','ruby','php','swift','kotlin','go','rust',
        'html','css','html5','css3','react','angular','vue','node.js','express','django','flask',
        'spring','laravel','rails','fastapi',
        'sql','mysql','postgresql','mongodb','redis','sqlite','oracle','dynamodb','firebase',
        'aws','azure','gcp','docker','kubernetes','git','linux','bash','terraform','jenkins',
        'machine learning','deep learning','nlp','tensorflow','pytorch','scikit-learn','keras',
        'pandas','numpy','matplotlib','seaborn','tableau','power bi','excel','r',
        'data analysis','data visualization','statistics','big data','spark','hadoop',
        'rest api','graphql','microservices','agile','scrum','devops','ci/cd',
        # Business
        'project management','team leadership','communication','problem solving','critical thinking',
        'time management','negotiation','presentation','strategic planning','budget management',
        'crm','salesforce','sap','erp','jira','confluence','trello','slack',
        # Finance/Accounting
        'financial analysis','budgeting','forecasting','gaap','auditing','tax','accounting',
        'quickbooks','tally','financial modeling','risk management','compliance',
        # HR
        'recruitment','talent acquisition','onboarding','performance management','payroll','hris',
        'employee relations','training','hr analytics',
        # Design
        'figma','adobe xd','photoshop','illustrator','indesign','sketch','ui/ux','wireframing',
        'prototyping','typography','user research',
        # Healthcare
        'patient care','emr','ehr','hipaa','clinical','medical billing','triage','cpr',
        # Marketing/Digital
        'seo','sem','social media','content marketing','email marketing','google analytics',
        'copywriting','brand management','adobe creative suite','wordpress',
        # Sales
        'lead generation','pipeline management','account management','cold calling','upselling',
        # Other
        'customer service','data entry','microsoft office','ms word','powerpoint','outlook',
        'quality control','six sigma','autocad','solidworks','lean manufacturing',
    ]

    found = []
    seen  = set()
    for skill in all_possible_skills:
        if skill.lower() in text_lower and skill.lower() not in seen:
            found.append(skill)
            seen.add(skill.lower())

    # Also scan ROLE_KEYWORDS pool so role-specific terms are caught with proper capitalisation
    for kw_list in ROLE_KEYWORDS.values():
        for kw in kw_list:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw_lower in text_lower:
                found.append(kw)
                seen.add(kw_lower)

    return found


def _break_into_short_lines(text: str, max_words: int = 30) -> str:
    """
    Split a paragraph into lines no longer than max_words words,
    breaking at sentence boundaries where possible.
    This prevents the ATS '50+ word paragraph' penalty.
    """
    # First split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    lines = []
    current = []
    current_count = 0
    for sent in sentences:
        words = sent.split()
        if current_count + len(words) > max_words and current:
            lines.append(' '.join(current))
            current = words
            current_count = len(words)
        else:
            current.extend(words)
            current_count += len(words)
    if current:
        lines.append(' '.join(current))
    return '\n'.join(lines)


# Per-role action phrases used to differentiate summaries
ROLE_ACTION_PHRASES = {
    "DATA-SCIENCE": [
        "Built and deployed machine learning models that improved prediction accuracy by 20%+.",
        "Conducted exploratory data analysis on datasets of 1M+ records to surface actionable insights.",
        "Designed data pipelines that reduced processing time by 35% across analytical workflows.",
    ],
    "WEB-DEVELOPER": [
        "Delivered 10+ responsive web applications with sub-2s load times and 99% uptime.",
        "Integrated RESTful APIs and third-party services to power real-time user experiences.",
        "Reduced page load time by 40% through code-splitting, caching, and lazy loading techniques.",
    ],
    "SOFTWARE-ENGINEER": [
        "Architected and shipped scalable microservices handling 500K+ daily requests.",
        "Reduced system downtime by 30% by implementing automated testing and CI/CD pipelines.",
        "Led code reviews for a team of 6 engineers, improving code quality and reducing bugs by 25%.",
    ],
    "HR": [
        "Recruited and onboarded 50+ candidates per quarter, reducing time-to-hire by 20%.",
        "Designed employee engagement programs that improved retention rates by 15%.",
        "Managed HR operations for a workforce of 200+ employees across 3 departments.",
    ],
    "DESIGNER": [
        "Delivered UI/UX designs for 15+ products with an average user satisfaction score of 4.7/5.",
        "Conducted 20+ user research sessions to drive data-informed design decisions.",
        "Reduced user onboarding drop-off by 30% through redesigned flows and prototypes.",
    ],
    "FINANCE": [
        "Managed financial reporting and forecasting for a $5M+ annual budget.",
        "Identified cost-saving opportunities that reduced operational expenses by 18%.",
        "Prepared quarterly P&L statements and variance analyses for C-suite stakeholders.",
    ],
    "HEALTHCARE": [
        "Delivered patient care across a caseload of 30+ patients daily with zero critical errors.",
        "Reduced patient wait times by 25% through optimized triage and workflow coordination.",
        "Maintained 100% compliance with HIPAA regulations and clinical documentation standards.",
    ],
    "SALES": [
        "Exceeded quarterly sales targets by 120%, generating $2M+ in new revenue.",
        "Built and managed a pipeline of 80+ prospects, converting 35% to closed deals.",
        "Retained 90% of key accounts through proactive relationship management strategies.",
    ],
    "INFORMATION-TECHNOLOGY": [
        "Managed IT infrastructure for 300+ endpoints with 99.9% system availability.",
        "Reduced incident resolution time by 40% by implementing an ITIL-based ticketing system.",
        "Deployed cloud migration for 3 critical systems, cutting infrastructure costs by 22%.",
    ],
    "ENGINEERING": [
        "Led 5+ engineering projects from design to delivery, all within scope and on budget.",
        "Improved production line efficiency by 20% through process redesign and automation.",
        "Maintained ISO 9001 compliance across all deliverables with zero non-conformances.",
    ],
    "ACCOUNTANT": [
        "Managed accounts for 50+ clients with 100% accuracy in financial reporting.",
        "Reduced month-end close cycle from 10 days to 5 days through process automation.",
        "Identified and recovered $120K in billing discrepancies through detailed reconciliation.",
    ],
    "TEACHER": [
        "Improved student test scores by 28% through differentiated instruction strategies.",
        "Designed curriculum for 5 courses serving 120+ students across 3 grade levels.",
        "Achieved 95% parent satisfaction rating through consistent communication and engagement.",
    ],
    "BANKING": [
        "Processed 100+ loan applications monthly with zero compliance violations.",
        "Grew client portfolio by 30% through proactive cross-selling of financial products.",
        "Maintained full AML and KYC compliance across a book of 500+ customer accounts.",
    ],
    "BUSINESS-DEVELOPMENT": [
        "Closed $3M+ in new partnerships and contracts within the first year of joining.",
        "Identified 20+ market expansion opportunities through competitor and market analysis.",
        "Built a partner network of 15+ organizations, increasing deal flow by 40%.",
    ],
    "DIGITAL-MEDIA": [
        "Grew social media following by 200% in 6 months through targeted content strategies.",
        "Managed $50K+ in ad spend with an average ROAS of 4.2x across campaigns.",
        "Produced 30+ pieces of high-performing content per month with measurable engagement uplift.",
    ],
    "DEFAULT": [
        "Delivered 10+ projects on time and within budget across cross-functional teams.",
        "Improved team efficiency by 20% through process improvements and workflow automation.",
        "Collaborated with 5+ departments to drive organizational goals and measurable outcomes.",
    ],
}




def normalize_role(predicted_role: str) -> str:
    if not predicted_role:
        return 'DEFAULT'
    if predicted_role in VALID_DB_ROLES:
        return predicted_role
    lower = predicted_role.lower().strip()
    if lower in ROLE_NORMALIZATION_MAP:
        return ROLE_NORMALIZATION_MAP[lower]
    for key, value in ROLE_NORMALIZATION_MAP.items():
        if key in lower or lower in key:
            return value
    upper = predicted_role.upper().replace(' ', '-')
    for valid_role in VALID_DB_ROLES:
        if valid_role in upper or upper in valid_role:
            return valid_role
    return 'DEFAULT'


def generate_smart_suggestions(resume_text, predicted_role=None):
    """
    Advanced suggestions focused ONLY on career-level and role-specific improvements.
    Intentionally avoids duplicating ATS checks (word count, contact, sections, verbs)
    since check_ats_friendliness() already covers those.
    """
    suggestions = []
    issues      = []
    text_lower  = resume_text.lower()

    # ── 1. Role-specific keyword gaps ────────────────────────────────────────
    if predicted_role and predicted_role in ROLE_KEYWORDS:
        role_kws     = ROLE_KEYWORDS[predicted_role]
        present      = [k for k in role_kws if k.lower() in text_lower]
        missing      = [k for k in role_kws if k.lower() not in text_lower]
        coverage_pct = int(len(present) / max(len(role_kws), 1) * 100)
        if coverage_pct >= 70:
            suggestions.append(
                f"Strong keyword match for {predicted_role.replace('-', ' ').title()} "
                f"({coverage_pct}% role keywords present)."
            )
        elif coverage_pct >= 40:
            issues.append(
                f"Moderate keyword coverage ({coverage_pct}%) for "
                f"{predicted_role.replace('-', ' ').title()}. "
                f"Add: {', '.join(missing[:5])}."
            )
        else:
            issues.append(
                f"Low keyword coverage ({coverage_pct}%) for "
                f"{predicted_role.replace('-', ' ').title()}. "
                f"Critical missing skills: {', '.join(missing[:6])}."
            )

    # ── 2. LinkedIn profile ───────────────────────────────────────────────────
    if 'linkedin.com' not in text_lower:
        suggestions.append(
            "Add your LinkedIn profile URL (linkedin.com/in/yourname) — "
            "recruiters check it 87% of the time."
        )

    # ── 3. GitHub / Portfolio (tech roles) ───────────────────────────────────
    tech_roles = {'DATA-SCIENCE', 'INFORMATION-TECHNOLOGY', 'ENGINEERING',
                  'DESIGNER', 'DIGITAL-MEDIA', 'DATA-ANALYST',
                  'JAVA-DEVELOPER', 'PYTHON-DEVELOPER', 'DEVOPS',
                  'WEB-DESIGNING', 'DOTNET-DEVELOPER', 'REACT-DEVELOPER'}
    if predicted_role in tech_roles:
        if 'github' not in text_lower and 'portfolio' not in text_lower:
            issues.append(
                "No GitHub or portfolio link found. Tech recruiters expect to see your work — "
                "add github.com/yourusername or a portfolio URL."
            )

    # ── 4. Certifications check ───────────────────────────────────────────────
    cert_keywords = ['certified', 'certification', 'certificate', 'aws', 'pmp',
                     'google', 'microsoft', 'coursera', 'udemy', 'cisco', 'comptia']
    has_certs = any(k in text_lower for k in cert_keywords)
    if not has_certs:
        suggestions.append(
            "No certifications detected. Adding 1-2 relevant certifications "
            "(e.g. AWS, Google, PMP, Coursera) significantly boosts ATS ranking."
        )

    # ── 5. Repetitive word detection ─────────────────────────────────────────
    words = resume_text.split()
    STOP = {'which','their','about','these','there','where','would','could',
            'should','experience','skills','resume','years','using','various',
            'multiple','different','ability','strong','excellent'}
    freq = {}
    for w in words:
        w = w.lower().strip('.,;:()')
        if len(w) > 4 and w not in STOP:
            freq[w] = freq.get(w, 0) + 1
    overused = sorted([w for w, c in freq.items() if c > 4], key=lambda x: -freq[x])
    if overused:
        suggestions.append(
            f"Overused words detected: '{'\', \''.join(overused[:3])}'. "
            "Use synonyms to vary your language and improve readability."
        )

    # ── 6. Passive voice detection ────────────────────────────────────────────
    passive_patterns = [
        r'\bwas responsible for\b', r'\bwere responsible for\b',
        r'\bwas involved in\b',     r'\bwere involved in\b',
        r'\bwas part of\b',         r'\bwere part of\b',
        r'\bwas tasked with\b',     r'\bwere tasked with\b',
        r'\bhas been\b',            r'\bhave been\b',
    ]
    passive_count = sum(1 for p in passive_patterns if re.search(p, text_lower))
    if passive_count >= 3:
        issues.append(
            f"Passive voice detected {passive_count} times. "
            "Replace with active verbs: 'Led', 'Designed', 'Built', 'Delivered'."
        )
    elif passive_count >= 1:
        suggestions.append(
            "Some passive voice detected. Replace 'was responsible for' → 'Led', "
            "'was involved in' → 'Contributed to'."
        )

    # ── 7. Summary quality check ──────────────────────────────────────────────
    summary_match = re.search(
        r'(professional summary|summary|objective|profile)[\s\S]{0,500}',
        text_lower
    )
    if summary_match:
        summary_text = summary_match.group()
        summary_words = len(summary_text.split())
        if summary_words < 30:
            issues.append(
                "Professional summary is too short (under 30 words). "
                "Write 3-5 sentences highlighting your expertise, key skills, and career goal."
            )
        elif summary_words > 120:
            suggestions.append(
                "Professional summary is too long (over 120 words). "
                "Keep it to 3-5 punchy sentences — recruiters spend 6 seconds on it."
            )

    # ── 8. Education depth ────────────────────────────────────────────────────
    if 'education' in text_lower:
        has_year   = bool(re.search(r'\b(19|20)\d{2}\b', resume_text))
        has_degree = any(d in text_lower for d in [
            'bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.e',
            'b.tech', 'm.tech', 'mba', 'diploma'
        ])
        if not has_degree:
            suggestions.append(
                "Spell out your full degree name (e.g. 'Bachelor of Technology') "
                "— abbreviations like B.E may not be parsed by ATS."
            )
        if not has_year:
            suggestions.append(
                "Add graduation year to your Education section — "
                "ATS systems use this to calculate experience."
            )

    # ── 9. Projects section (for tech/design roles) ───────────────────────────
    if predicted_role in tech_roles:
        if 'project' not in text_lower:
            suggestions.append(
                "No Projects section detected. Adding 2-3 relevant projects with "
                "tech stack and measurable outcomes greatly strengthens tech resumes."
            )

    # ── 10. Soft skills balance ───────────────────────────────────────────────
    soft_skills = ['communication', 'teamwork', 'leadership', 'problem solving',
                   'collaboration', 'adaptability', 'time management']
    found_soft = [s for s in soft_skills if s in text_lower]
    if len(found_soft) == 0:
        suggestions.append(
            "No soft skills mentioned. Add 2-3 like 'Team Leadership', "
            "'Cross-functional Collaboration', 'Problem Solving'."
        )

    return {'issues': issues, 'suggestions': suggestions}


# ============================================================
# HEALTH CHECK
# ============================================================



# ============================================================
# ATS SCORING — Advanced & Accurate v3.0
# ============================================================

def _analyze_pdf_structure(filepath: str) -> dict:
    """
    Analyzes raw PDF structure for drawing objects, images, and charts.
    Returns penalty signals that text-extraction cannot detect.
    Only called when filepath is available (not for extracted text analysis).

    NOTE: Standard PDF generators (ReportLab, Word, LibreOffice) produce a small
    number of lines/rects for page borders and section dividers. We only penalise
    when counts are clearly above what a clean single-column resume would produce.
    """
    result = {'drawing_objects': 0, 'has_chart': False, 'has_images': False, 'penalty': 0, 'reasons': []}
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            total_rects   = sum(len(p.rects)   for p in pdf.pages)
            total_curves  = sum(len(p.curves)  for p in pdf.pages)
            total_lines   = sum(len(p.lines)   for p in pdf.pages)
            total_images  = sum(len(p.images)  for p in pdf.pages)
            # Lines from section dividers & page borders are expected — subtract baseline
            n_pages       = max(len(pdf.pages), 1)
            baseline_lines = n_pages * 10   # ~10 lines/page is normal for a clean resume
            total_objects = max(0, (total_rects + total_curves + total_lines) - baseline_lines)

        result['drawing_objects'] = total_objects
        result['has_images']      = total_images > 0

        # Images always penalised — ATS cannot read them
        if total_images > 0:
            result['penalty'] += 25
            result['reasons'].append(f'{total_images} embedded image(s) detected — ATS cannot read images')

        # Only penalise drawing objects well above the baseline
        if total_objects > 120:
            result['penalty'] += 25
            result['reasons'].append(f'Heavy graphical elements ({total_objects} objects) — charts/skill bars/graphics present')
        elif total_objects > 60:
            result['penalty'] += 15
            result['reasons'].append(f'Moderate graphical elements ({total_objects} objects) — possible charts or decorative graphics')
        elif total_objects > 25:
            result['penalty'] += 5
            result['reasons'].append(f'Minor graphical elements ({total_objects} objects)')

        # Curves are a strong signal for charts (not produced by simple dividers)
        if total_curves > 40:
            result['has_chart'] = True
            result['penalty'] += 10
            result['reasons'].append('Chart/graph curves detected — ATS cannot parse graphical data')

        result['penalty'] = min(result['penalty'], 40)
    except Exception:
        pass
    return result


def _detect_skill_percentage_ratings(text: str) -> dict:
    """
    Detects skill percentage bars like 'React / Next.js 92%' or 'Python 85%'.
    These are ATS-unfriendly because they replace concrete experience with arbitrary numbers.
    """
    import re
    # Pattern: skill name followed directly by a percentage
    skill_pct_pattern = re.findall(
        r'[A-Za-z][A-Za-z0-9\.\+\#\/\s\-]{1,30}\s+(\d{2,3})%',
        text
    )
    # Filter to likely skill ratings (50-99%)
    skill_ratings = [p for p in skill_pct_pattern if 50 <= int(p) <= 99]

    # Also detect chart axis numbers (0 25 50 75 100 on separate lines = bar chart)
    has_chart_axis = bool(re.search(r'0\s+25\s+50\s+75\s+100', text))

    penalty = 0
    reasons = []
    if len(skill_ratings) >= 5:
        penalty += 20
        reasons.append(f'{len(skill_ratings)} skill percentage ratings detected (e.g. Python 85%) — replace with concrete experience')
    elif len(skill_ratings) >= 2:
        penalty += 10
        reasons.append(f'{len(skill_ratings)} skill percentage ratings — ATS ignores arbitrary self-ratings')

    if has_chart_axis:
        penalty += 15
        reasons.append('Proficiency bar chart detected — ATS cannot read graphical charts')

    return {'penalty': min(penalty, 30), 'reasons': reasons, 'count': len(skill_ratings)}


def _detect_third_person_writing(text: str) -> dict:
    """
    Detects third-person writing (He/She/They + person's name repeated as subject).
    Professional resumes should always use first-person implied (bullet points starting with verbs).
    """
    import re
    third_person_refs = re.findall(r'\b(He |She |They )(?=[A-Z]|[a-z])', text)

    # Also detect sentences starting with a name (Ryan serves..., John manages...)
    name_as_subject = re.findall(
        r'(?:^|\n)([A-Z][a-z]+ (?:serves|is|was|has|works|manages|leads|owns|builds|joined|began|developed|architected))', text, re.MULTILINE
    )

    total = len(third_person_refs) + len(name_as_subject)
    penalty = 0
    reasons = []

    if total >= 8:
        penalty = 15
        reasons.append(f'Resume written in third person ({total} references) — use first-person bullet points starting with action verbs')
    elif total >= 3:
        penalty = 8
        reasons.append(f'Some third-person writing detected ({total} references) — prefer first-person implied style')

    return {'penalty': penalty, 'reasons': reasons, 'count': total}


def _detect_career_timeline_graphic(text: str) -> dict:
    """Detects career timeline infographic blocks."""
    import re
    timeline = bool(re.search(r'career timeline|CAREER TIMELINE', text, re.IGNORECASE))
    year_row = bool(re.search(r'20\d\d\s+20\d\d\s+20\d\d', text))
    role_row = bool(re.search(r'Junior Dev.*Mid Dev|Mid Dev.*Senior Dev|Senior Dev.*Lead', text, re.IGNORECASE))

    if timeline and (year_row or role_row):
        return {'penalty': 10, 'reason': 'Career timeline infographic detected — replace with standard work experience dates'}
    return {'penalty': 0, 'reason': ''}


def _detect_paragraph_walls(text: str) -> dict:
    """
    Properly detects wall-of-text by stitching split PDF lines back into paragraphs,
    then measuring reconstructed paragraph lengths.
    """
    import re
    raw_lines = [ln.strip() for ln in text.split('\n') if ln.strip()]

    # Stitch adjacent long lines into paragraphs (PDF wraps long paragraphs)
    # Never stitch bullet lines or separator lines — they are already formatted correctly
    BULLET_RE = re.compile(r'^[\-\•\*\>\–\—\►\▸]')
    SEP_RE    = re.compile(r'^-{5,}$')

    paragraphs = []
    current    = []
    for ln in raw_lines:
        words = ln.split()
        # Bullets and separators always break the current paragraph group
        if BULLET_RE.match(ln) or SEP_RE.match(ln):
            if len(current) >= 2:
                paragraphs.append(' '.join(current))
            current = []
            continue
        if len(words) >= 6:
            current.append(ln)
        else:
            if len(current) >= 2:
                paragraphs.append(' '.join(current))
            current = []
    if len(current) >= 2:
        paragraphs.append(' '.join(current))

    # Analyze reconstructed paragraphs
    long_paras      = [p for p in paragraphs if len(p.split()) > 80]
    medium_paras    = [p for p in paragraphs if 40 < len(p.split()) <= 80]

    # Bullet point ratio
    bullet_lines       = [ln for ln in raw_lines if re.match(r'^[\-\•\*\>\–\—\►\▸]', ln)]
    para_content_lines = [ln for ln in raw_lines if len(ln.split()) > 3]
    bullet_ratio       = len(bullet_lines) / max(len(para_content_lines), 1)

    penalty = 0
    issues  = []

    if long_paras:
        penalty += min(30, len(long_paras) * 15)
        issues.append(f'{len(long_paras)} very long paragraph(s) detected ({long_paras[0].split().__len__()} words) — break into bullet points')
    if medium_paras:
        penalty += min(15, len(medium_paras) * 5)
        issues.append(f'{len(medium_paras)} medium paragraph(s) — consider converting to bullet points')
    if bullet_ratio < 0.15 and len(para_content_lines) > 8:
        penalty += 20
        issues.append(f'Only {int(bullet_ratio*100)}% of content uses bullet points — ATS strongly prefers bullet format')
    elif bullet_ratio < 0.30 and len(para_content_lines) > 8:
        penalty += 10
        issues.append(f'Low bullet usage ({int(bullet_ratio*100)}%) — add more bullet points')

    penalty = min(penalty, 40)

    if penalty >= 30:
        label = f'Poor — {len(long_paras)+len(medium_paras)} paragraph block(s) detected'
    elif penalty >= 15:
        label = 'Fair — paragraphs should be bullet points'
    elif penalty > 0:
        label = 'Acceptable — minor paragraph issues'
    else:
        label = 'Good — bullet-point format'

    return {
        'penalty': penalty, 'label': label, 'issues': issues,
        'bullet_ratio': bullet_ratio, 'long_para_count': len(long_paras)
    }


def _detect_graphics_from_text(text: str) -> dict:
    """Text-based graphics detection as fallback when no filepath available."""
    import re
    total_chars   = len(text)
    if total_chars == 0:
        return {'penalty': 40, 'reason': 'Empty resume', 'label': 'Unreadable'}

    garbled       = len(re.findall(r'[^\x00-\x7F]', text))
    garbled_ratio = garbled / total_chars
    alpha         = len(re.findall(r'[a-zA-Z]', text))
    alpha_ratio   = alpha / total_chars
    skill_bars    = len(re.findall(r'[●○■□◆◇★☆▪▸►•]{3,}|[█▓░]{2,}', text))

    if garbled_ratio > 0.08:
        return {'penalty': 35, 'reason': f'Heavy image content — {int(garbled_ratio*100)}% unreadable characters', 'label': 'Images detected'}
    if alpha_ratio < 0.45:
        return {'penalty': 30, 'reason': f'Very low text density ({int(alpha_ratio*100)}%) — large images present', 'label': 'Graphics detected'}
    if skill_bars >= 3:
        return {'penalty': 25, 'reason': f'{skill_bars} graphical skill bar(s) — ATS cannot read these', 'label': 'Skill bars detected'}
    if garbled_ratio > 0.03 or alpha_ratio < 0.55:
        return {'penalty': 20, 'reason': 'Possible graphical elements', 'label': 'Possible graphics'}
    return {'penalty': 0, 'reason': '', 'label': 'None detected — clean text'}


def check_ats_friendliness(text: str, is_enhanced: bool = False, filepath: str = None) -> dict:
    """
    Advanced ATS scoring v3.0 — strict and accurate.
    Detects: images, charts, skill bars, career timelines, paragraph walls,
             third-person writing, missing sections, weak verbs, and more.
    Optional filepath enables PDF structure analysis for maximum accuracy.
    """
    import re
    issues      = []
    suggestions = []
    details     = {}
    text_lower  = text.lower()
    word_count  = len(text.split())

    # ── DIMENSION 1: Content Length (max 10) ────────────────────────────────
    if word_count < 150:
        length_score = 0
        issues.append(f'Resume too short ({word_count} words) — aim for 300–700 words')
        details['length'] = f'Poor ({word_count} words)'
    elif word_count < 300:
        length_score = 5
        issues.append(f'Resume slightly short ({word_count} words) — expand to 400+ words')
        details['length'] = f'Fair ({word_count} words)'
    elif 300 <= word_count <= 800:
        length_score = 10
        details['length'] = f'Good ({word_count} words)'
    elif word_count <= 1100:
        length_score = 6
        suggestions.append(f'Resume slightly long ({word_count} words) — trim to under 800 words')
        details['length'] = f'Slightly long ({word_count} words)'
    else:
        length_score = 2
        issues.append(f'Resume too long ({word_count} words) — ATS prefers under 800 words')
        details['length'] = f'Too long ({word_count} words)'

    # ── DIMENSION 2: Contact Info (max 10) ──────────────────────────────────
    has_email    = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone    = bool(re.search(r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text))
    has_linkedin = bool(re.search(r'linkedin\.com|linkedin', text_lower))
    # Email and phone together give full 10 pts — LinkedIn is a bonus suggestion only
    contact_score = (5 if has_email else 0) + (5 if has_phone else 0)
    if not has_email:
        issues.append('No email address detected — add your professional email')
    if not has_phone:
        issues.append('No phone number detected — add your contact number')
    if not has_linkedin:
        suggestions.append('Add your LinkedIn profile URL (linkedin.com/in/yourname)')
    details['contact_info'] = 'Complete' if (has_email and has_phone) else 'Incomplete'

    # ── DIMENSION 3: Essential Sections (max 20) ────────────────────────────
    section_keywords = {
        'Experience': ['experience', 'work history', 'employment', 'professional experience'],
        'Education':  ['education', 'qualification', 'degree', 'university', 'college'],
        'Skills':     ['skills', 'technical skills', 'competencies', 'proficiencies'],
        'Summary':    ['summary', 'objective', 'profile', 'about me', 'professional summary'],
    }
    sections_found   = []
    sections_missing = []
    for section, keywords in section_keywords.items():
        if any(k in text_lower for k in keywords):
            sections_found.append(section)
        else:
            sections_missing.append(section)
            if section in ['Experience', 'Education', 'Skills']:
                issues.append(f"Missing '{section}' section")
    section_score = len(sections_found) * 5
    details['sections'] = f'{len(sections_found)}/4 key sections found'

    # ── DIMENSION 4: Action Verbs (max 10) ──────────────────────────────────
    action_verbs = ['developed','managed','led','created','implemented','designed',
                    'analyzed','improved','coordinated','achieved','executed',
                    'established','built','optimized','delivered','increased',
                    'reduced','launched','spearheaded','collaborated','mentored',
                    'trained','negotiated','streamlined','generated','drove',
                    'architected','deployed','migrated','automated','scaled']
    verb_count = sum(1 for v in action_verbs if v in text_lower)
    if verb_count >= 15:
        verb_score = 10
        details['action_verbs'] = f'Excellent ({verb_count} found)'
    elif verb_count >= 10:
        verb_score = 7
        details['action_verbs'] = f'Good ({verb_count} found)'
    elif verb_count >= 5:
        verb_score = 4
        suggestions.append('Add more action verbs: developed, managed, optimized, delivered, spearheaded')
        details['action_verbs'] = f'Fair ({verb_count} found)'
    elif verb_count >= 2:
        verb_score = 2
        issues.append(f'Only {verb_count} action verb(s) — use strong verbs like Led, Built, Delivered')
        details['action_verbs'] = f'Weak ({verb_count} found)'
    else:
        verb_score = 0
        issues.append('No action verbs detected — every bullet must start with a strong verb')
        details['action_verbs'] = f'Poor ({verb_count} found)'

    # ── DIMENSION 5: Paragraph Format (max 18) ──────────────────────────────
    para_result  = _detect_paragraph_walls(text)
    para_penalty = para_result['penalty']
    para_score   = max(0, 18 - para_penalty)
    details['paragraphs'] = para_result['label']
    for issue in para_result.get('issues', []):
        if para_penalty >= 15:
            issues.append(issue)
        else:
            suggestions.append(issue)

    # ── DIMENSION 6: Graphics & Charts (max 15) ─────────────────────────────
    graphics_penalty = 0
    graphics_reasons = []

    if True:  # always run graphics check — is_enhanced no longer bypasses this
        # Signal A: PDF structure analysis (most accurate)
        if filepath:
            pdf_result = _analyze_pdf_structure(filepath)
            graphics_penalty += pdf_result['penalty']
            graphics_reasons += pdf_result['reasons']

        # Signal B: Skill percentage ratings
        skill_pct_result = _detect_skill_percentage_ratings(text)
        graphics_penalty += skill_pct_result['penalty']
        graphics_reasons += skill_pct_result['reasons']

        # Signal C: Career timeline graphic
        timeline_result = _detect_career_timeline_graphic(text)
        graphics_penalty += timeline_result['penalty']
        if timeline_result['reason']:
            graphics_reasons.append(timeline_result['reason'])

        # Signal D: Text-based fallback
        if not filepath:
            txt_result = _detect_graphics_from_text(text)
            graphics_penalty += txt_result['penalty']
            if txt_result['reason']:
                graphics_reasons.append(txt_result['reason'])

        graphics_penalty = min(graphics_penalty, 40)

        for reason in graphics_reasons:
            if graphics_penalty >= 20:
                issues.append(reason)
            else:
                suggestions.append(reason)

        if graphics_penalty >= 20:
            suggestions.append('Remove all graphics: skill bars, charts, timelines, photos — replace with plain text')

        label_parts = []
        if graphics_penalty >= 30:
            label_parts.append('Heavy graphics detected')
        elif graphics_penalty >= 15:
            label_parts.append('Moderate graphics detected')
        elif graphics_penalty > 0:
            label_parts.append('Minor graphics detected')
        else:
            label_parts.append('Clean text — no graphics')
        details['images_graphics'] = ' | '.join(label_parts)
        graphics_score = max(0, 15 - graphics_penalty)

    # ── DIMENSION 7: Third-Person Writing (max 7) ───────────────────────────
    third_person_result = _detect_third_person_writing(text)
    third_penalty       = third_person_result['penalty']
    third_score         = max(0, 7 - third_penalty)
    details['writing_style'] = (
        'Third-person (poor ATS style)' if third_penalty >= 8
        else 'First-person implied (good)' if third_penalty == 0
        else 'Mixed style'
    )
    for reason in third_person_result['reasons']:
        issues.append(reason) if third_penalty >= 8 else suggestions.append(reason)

    # ── DIMENSION 8: Quantified Achievements (max 5) ────────────────────────
    # Exclude: years (19xx/20xx), long phone fragments (7+ digits),
    #          country code digits after '+', trivial small numbers, skill % bars
    skill_pct_vals  = set(re.findall(r'\b[5-9]\d%', text))
    phone_cc_digits = set(re.findall(r'(?<=\+)\d{1,3}(?=[-\s])', text))
    real_metrics = []
    for tok in re.findall(r'\b\d+[\%\+]?\b', text):
        clean = tok.rstrip('%+')
        if re.match(r'^(19|20)\d{2}$', clean):
            continue  # years
        if len(clean) >= 7 and clean.isdigit():
            continue  # phone fragments
        if clean in {'0', '1', '2', '3', '4'}:
            continue  # trivial
        if tok in skill_pct_vals:
            continue  # skill bars
        if clean in phone_cc_digits:
            continue  # +91 prefix
        real_metrics.append(tok)
    if len(real_metrics) >= 10:
        quant_score = 5
        details['quantification'] = f'Excellent — {len(real_metrics)} metrics found'
    elif len(real_metrics) >= 6:
        quant_score = 3
        details['quantification'] = f'Good — {len(real_metrics)} numbers found'
    elif len(real_metrics) >= 3:
        quant_score = 1
        suggestions.append("Add more quantified achievements — e.g. 'Improved performance by 30%', 'Managed team of 10'")
        details['quantification'] = f'Fair — {len(real_metrics)} numbers found'
    elif len(real_metrics) >= 1:
        quant_score = 0
        issues.append("Very few quantified achievements — add concrete numbers for every role")
        details['quantification'] = f'Weak — only {len(real_metrics)} metric(s) found'
    else:
        quant_score = 0
        issues.append("No quantified achievements — add numbers like '30% increase', 'team of 8', '$2M revenue'")
        details['quantification'] = 'Poor — no real metrics found'

    # ── DIMENSION 9: Special Characters / Encoding (max 5) ──────────────────
    # Allow chars that are normal in ATS-friendly plain-text resumes:
    # @ (email), + (phone), | (separator), & (section names), % (metrics), # (skills), "
    # Only penalise box-drawing, emojis, and non-ASCII decorative characters.
    special_ratio   = len(re.findall(r'[^\w\s.,;:!?()\\/\-\'\"@+|&%#\n]', text)) / max(len(text), 1)
    encoding_issues = len(re.findall(r'[\x80-\x9F]', text))
    if special_ratio > 0.05 or encoding_issues > 10:
        char_score = 0
        issues.append('Excessive special characters — use plain text formatting')
        details['formatting'] = 'Complex (ATS risk)'
    elif special_ratio > 0.02:
        char_score = 3
        details['formatting'] = 'Acceptable'
    else:
        char_score = 5
        details['formatting'] = 'Clean (ATS-friendly)'

    # ── DIMENSION 10: Keyword Density & Role Alignment (max 10) ─────────────
    # Uses specific professional terms that distinguish quality resumes.
    # Generic words like 'experience', 'skills', 'team' deliberately excluded —
    # every resume has them and they provide zero signal of quality.
    domain_keywords = [
        # Metrics & business impact vocabulary
        'revenue', 'efficiency', 'productivity', 'cost', 'budget', 'growth',
        'roi', 'kpi', 'targets', 'objectives', 'stakeholders', 'deliverables',
        # Strong professional verbs (more specific)
        'spearheaded', 'orchestrated', 'negotiated', 'streamlined', 'optimized',
        'launched', 'automated', 'mentored', 'deployed', 'architected',
        'onboarded', 'reduced', 'increased', 'generated', 'drove',
        # Scope / collaboration indicators
        'cross-functional', 'enterprise', 'end-to-end', 'scalable',
        'roadmap', 'milestone', 'pipeline', 'compliance', 'governance',
    ]
    domain_hits = sum(1 for k in domain_keywords if k in text_lower)
    if domain_hits >= 18:
        keyword_density_score = 10
        details['keyword_density'] = f'Excellent ({domain_hits} professional terms)'
    elif domain_hits >= 12:
        keyword_density_score = 7
        details['keyword_density'] = f'Good ({domain_hits} professional terms)'
    elif domain_hits >= 6:
        keyword_density_score = 4
        suggestions.append('Use more industry-standard professional vocabulary: KPIs, ROI, stakeholders, optimized, spearheaded.')
        details['keyword_density'] = f'Fair ({domain_hits} professional terms)'
    elif domain_hits >= 2:
        keyword_density_score = 2
        issues.append('Low professional vocabulary — add impact-focused terms: revenue, efficiency, stakeholders, optimized.')
        details['keyword_density'] = f'Weak ({domain_hits} professional terms)'
    else:
        keyword_density_score = 0
        issues.append('Resume lacks professional impact vocabulary — add terms like KPIs, ROI, optimized, stakeholders.')
        details['keyword_density'] = f'Poor ({domain_hits} professional terms)'

    # ── DIMENSION 11: Consistency & Completeness Penalty ────────────────────
    # Checks structural completeness — dates, bullets, company names, content depth,
    # actual degree name, and minimum bullet count.
    lines_with_content = [ln for ln in text.split('\n') if len(ln.strip()) > 20]
    has_dates    = bool(re.search(r'\b(19|20)\d{2}\b', text))
    has_bullets  = bool(re.search(r'^\s*[\-\•\*\–]', text, re.MULTILINE))
    has_company  = bool(re.search(r'\b(pvt|ltd|inc|corp|llc|company|technologies|solutions|services|institute|university|college)\b', text_lower))
    has_degree   = any(d in text_lower for d in ['bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.e', 'b.tech', 'm.tech', 'mba', 'diploma', 'b.com', 'b.a', 'm.a'])
    bullet_lines = [ln for ln in text.split('\n') if re.match(r'^\s*[\-\•\*\–\►]', ln)]

    completeness_score = 0
    if has_dates:
        completeness_score += 2
    else:
        issues.append('No dates found — add employment and education dates (e.g. Jan 2022 – Present).')
    if has_bullets:
        completeness_score += 1
    else:
        suggestions.append('Use bullet points for experience and achievements.')
    if len(bullet_lines) >= 8:
        completeness_score += 2
    elif len(bullet_lines) >= 4:
        completeness_score += 1
    else:
        issues.append(f'Only {len(bullet_lines)} bullet point(s) found — add at least 8 bullets across experience and skills.')
    if has_company:
        completeness_score += 1
    else:
        suggestions.append('Mention company/institution names explicitly.')
    if has_degree:
        completeness_score += 1
    else:
        suggestions.append('Spell out your full degree name (e.g. Bachelor of Technology) — ATS may miss abbreviations.')
    if len(lines_with_content) >= 25:
        completeness_score += 2
    elif len(lines_with_content) >= 15:
        completeness_score += 1
    else:
        issues.append('Resume content is very thin — expand all sections with more detail.')

    # Hard cap: if degree missing AND no company, cap completeness low
    if not has_degree and not has_company:
        completeness_score = min(completeness_score, 2)

    details['completeness'] = f'{completeness_score}/9'

    # ── FINAL SCORE ──────────────────────────────────────────────────────────
    # Rebalanced weights — max ~100 points across 11 dimensions
    # Stricter thresholds: domain keywords harder, verbs harder, quant harder
    length_score_adj   = int(length_score   * 0.8)   # max 8
    contact_score_adj  = int(contact_score  * 0.8)   # max 8
    section_score_adj  = int(section_score  * 0.8)   # max 16
    para_score_adj     = int(para_score     * 0.78)  # max ~14
    graphics_score_adj = int(graphics_score * 0.8)   # max 12
    third_score_adj    = int(third_score    * 0.71)  # max ~5
    quant_score_adj    = int(quant_score    * 1.2)   # max 6
    # verb_score: max 10, char_score: max 5, keyword_density: max 10, completeness: max 9

    raw_score = (length_score_adj + contact_score_adj + section_score_adj +
                 verb_score + para_score_adj + graphics_score_adj +
                 third_score_adj + quant_score_adj + char_score +
                 keyword_density_score + completeness_score)

    score = min(100, int(raw_score))

    # ── Hard penalties for critical quality signals ───────────────────────
    if quant_score == 0:
        score = max(0, score - 12)   # no metrics = heavy penalty
    elif quant_score <= 1:
        score = max(0, score - 5)    # very few metrics
    if not has_dates:
        score = max(0, score - 7)
    if keyword_density_score <= 2:
        score = max(0, score - 5)    # resume reads as generic/vague
    if verb_count < 5:
        score = max(0, score - 5)    # not enough action verbs

    # Hard caps for critical failures
    critical_failures = []
    if graphics_penalty >= 25:
        critical_failures.append('heavy graphics/charts detected')
    if para_penalty >= 25:
        critical_failures.append('wall-of-text paragraphs')
    if section_score <= 5:
        critical_failures.append('missing essential sections')
    if contact_score < 5:
        critical_failures.append('missing contact info')
    if third_penalty >= 8:
        critical_failures.append('third-person writing style')

    if critical_failures:
        score = min(score, 65)

    is_ats_friendly = score >= 80 and len(critical_failures) == 0

    if score >= 90:
        overall = 'Excellent — Highly ATS-friendly. Role prediction unlocked.'
    elif score >= 80:
        overall = 'Good — ATS-friendly. Role prediction unlocked.'
    elif score >= 65:
        overall = 'Fair — Needs improvement. Reach 80 to unlock role prediction.'
    elif score >= 45:
        overall = 'Poor — Major issues detected. Fix these to reach 80+.'
    else:
        overall = 'Very Poor — Resume will likely be rejected by ATS. Significant rework needed.'

    if critical_failures:
        overall += f' (Critical: {", ".join(critical_failures)})'

    return {
        'is_ats_friendly': is_ats_friendly,
        'score':           score,
        'overall':         overall,
        'issues':          issues,
        'suggestions':     suggestions,
        'details':         details,
        'score_breakdown': {
            'content_length':    length_score_adj,
            'contact_info':      contact_score_adj,
            'sections':          section_score_adj,
            'action_verbs':      verb_score,
            'paragraph_format':  para_score_adj,
            'graphics_charts':   graphics_score_adj,
            'writing_style':     third_score_adj,
            'quantification':    quant_score_adj,
            'special_chars':     char_score,
            'keyword_density':   keyword_density_score,
            'completeness':      completeness_score,
        }
    }