import requests

# Test resume text
test_resume = """
JOHN DOE
Software Engineer

SKILLS:
- Python, Django, Flask, FastAPI
- React.js, JavaScript, HTML, CSS
- PostgreSQL, MongoDB, Redis
- Docker, Kubernetes, AWS
- Git, CI/CD, Agile

EXPERIENCE:
Senior Software Engineer - Tech Corp (2020-2024)
- Built scalable web applications using Python and React
- Designed RESTful APIs and microservices architecture
- Implemented CI/CD pipelines using Jenkins and GitHub Actions
- Managed PostgreSQL and MongoDB databases

Full Stack Developer - StartupXYZ (2018-2020)
- Developed responsive web applications
- Integrated third-party APIs
- Optimized application performance

EDUCATION:
Bachelor of Science in Computer Science
University of Technology (2014-2018)
"""

# API endpoint
url = "http://localhost:5000/api/analyze-resume"

# Make request
response = requests.post(
    url,
    json={"resume_text": test_resume},
    headers={"Content-Type": "application/json"}
)

# Print results
print("="*60)
print("RESUME ANALYSIS RESULTS")
print("="*60)

if response.status_code == 200:
    result = response.json()
    
    print(f"\n🎯 Predicted Role: {result['predicted_role']}")
    print(f"📊 Confidence: {result['confidence']*100:.2f}%")
    
    print("\n🏆 Top 3 Predictions:")
    for i, (role, prob) in enumerate(result['top_3_roles'], 1):
        print(f"  {i}. {role}: {prob*100:.2f}%")
    
    print("\n❓ Interview Questions:")
    for i, question in enumerate(result['interview_questions'], 1):
        print(f"  {i}. {question}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)