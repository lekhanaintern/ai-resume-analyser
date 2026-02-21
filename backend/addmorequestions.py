"""
Add More MCQ Questions to Database
Run this file to add 50+ questions for all job roles
"""

from database import Database

db = Database()

# Comprehensive question bank
all_questions = [
    # ========================================
    # DATA SCIENTIST QUESTIONS (15 questions)
    # ========================================
    {
        'job_role': 'Data Scientist',
        'question': 'Which library is primarily used for data manipulation in Python?',
        'options': ['NumPy', 'Pandas', 'Matplotlib', 'Scikit-learn'],
        'correct_answer': 'Pandas',
        'difficulty': 'easy',
        'explanation': 'Pandas is the primary library for data manipulation and analysis in Python.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What does overfitting mean in machine learning?',
        'options': [
            'Model performs well on training but poorly on test data',
            'Model performs poorly on both datasets',
            'Model performs well on both datasets',
            'Model performs poorly on training but well on test data'
        ],
        'correct_answer': 'Model performs well on training but poorly on test data',
        'difficulty': 'medium',
        'explanation': 'Overfitting occurs when a model learns training data too well, including noise.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'Which algorithm is used for classification problems?',
        'options': ['Linear Regression', 'K-Means', 'Logistic Regression', 'PCA'],
        'correct_answer': 'Logistic Regression',
        'difficulty': 'medium',
        'explanation': 'Logistic Regression is a classification algorithm.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What is the purpose of cross-validation?',
        'options': [
            'To train faster',
            'To evaluate model performance on unseen data',
            'To clean data',
            'To visualize results'
        ],
        'correct_answer': 'To evaluate model performance on unseen data',
        'difficulty': 'medium',
        'explanation': 'Cross-validation helps assess how well a model generalizes to unseen data.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'Which metric is best for imbalanced classification problems?',
        'options': ['Accuracy', 'F1 Score', 'MSE', 'R-squared'],
        'correct_answer': 'F1 Score',
        'difficulty': 'medium',
        'explanation': 'F1 Score balances precision and recall, making it suitable for imbalanced datasets.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What is gradient descent used for?',
        'options': [
            'Data visualization',
            'Optimizing model parameters',
            'Feature selection',
            'Data cleaning'
        ],
        'correct_answer': 'Optimizing model parameters',
        'difficulty': 'medium',
        'explanation': 'Gradient descent is an optimization algorithm to minimize loss functions.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What does NLP stand for in Data Science?',
        'options': [
            'Natural Language Processing',
            'Neural Learning Process',
            'Network Layer Protocol',
            'None of these'
        ],
        'correct_answer': 'Natural Language Processing',
        'difficulty': 'easy',
        'explanation': 'NLP is the field of AI focused on understanding human language.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'Which is a supervised learning algorithm?',
        'options': ['K-Means', 'Decision Tree', 'DBSCAN', 'Apriori'],
        'correct_answer': 'Decision Tree',
        'difficulty': 'easy',
        'explanation': 'Decision Trees are supervised learning algorithms used for classification and regression.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What is feature engineering?',
        'options': [
            'Creating new features from existing data',
            'Removing features',
            'Scaling features',
            'Encoding categorical features'
        ],
        'correct_answer': 'Creating new features from existing data',
        'difficulty': 'medium',
        'explanation': 'Feature engineering creates new features to improve model performance.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'Which library is used for deep learning in Python?',
        'options': ['Pandas', 'NumPy', 'TensorFlow', 'Matplotlib'],
        'correct_answer': 'TensorFlow',
        'difficulty': 'easy',
        'explanation': 'TensorFlow is a popular deep learning framework.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What is the purpose of regularization?',
        'options': [
            'Prevent overfitting',
            'Increase training speed',
            'Clean data',
            'Visualize results'
        ],
        'correct_answer': 'Prevent overfitting',
        'difficulty': 'medium',
        'explanation': 'Regularization techniques like L1/L2 help prevent overfitting.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What does RMSE stand for?',
        'options': [
            'Root Mean Square Error',
            'Relative Mean Standard Error',
            'Random Model Selection Error',
            'Rapid Model System Evaluation'
        ],
        'correct_answer': 'Root Mean Square Error',
        'difficulty': 'easy',
        'explanation': 'RMSE is a common metric for regression problems.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'Which algorithm is used for dimensionality reduction?',
        'options': ['Random Forest', 'PCA', 'K-NN', 'SVM'],
        'correct_answer': 'PCA',
        'difficulty': 'medium',
        'explanation': 'Principal Component Analysis (PCA) reduces feature dimensionality.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What is a confusion matrix used for?',
        'options': [
            'Visualizing data',
            'Evaluating classification models',
            'Feature selection',
            'Data preprocessing'
        ],
        'correct_answer': 'Evaluating classification models',
        'difficulty': 'easy',
        'explanation': 'Confusion matrix shows true/false positives and negatives.'
    },
    {
        'job_role': 'Data Scientist',
        'question': 'What is the curse of dimensionality?',
        'options': [
            'Too many features make models perform poorly',
            'Not enough training data',
            'Slow training time',
            'Incorrect labels'
        ],
        'correct_answer': 'Too many features make models perform poorly',
        'difficulty': 'hard',
        'explanation': 'High dimensionality can lead to sparse data and poor model performance.'
    },

    # ========================================
    # WEB DEVELOPER QUESTIONS (15 questions)
    # ========================================
    {
        'job_role': 'Web Developer',
        'question': 'What does DOM stand for?',
        'options': ['Document Object Model', 'Data Object Model', 'Document Oriented Model', 'Display Object Management'],
        'correct_answer': 'Document Object Model',
        'difficulty': 'easy',
        'explanation': 'DOM is the programming interface for HTML documents.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'Which JavaScript method adds an element at the end of an array?',
        'options': ['push()', 'pop()', 'shift()', 'unshift()'],
        'correct_answer': 'push()',
        'difficulty': 'easy',
        'explanation': 'push() adds elements to the end of an array.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'What is CSS used for?',
        'options': [
            'Styling web pages',
            'Database queries',
            'Server logic',
            'Network protocols'
        ],
        'correct_answer': 'Styling web pages',
        'difficulty': 'easy',
        'explanation': 'CSS (Cascading Style Sheets) styles HTML elements.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'What does CORS stand for?',
        'options': [
            'Cross-Origin Resource Sharing',
            'Cross-Origin Request Security',
            'Central Origin Resource System',
            'Cross-Object Resource Sharing'
        ],
        'correct_answer': 'Cross-Origin Resource Sharing',
        'difficulty': 'medium',
        'explanation': 'CORS is a security feature for cross-origin requests.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'Which HTML tag is used for creating hyperlinks?',
        'options': ['<link>', '<a>', '<href>', '<url>'],
        'correct_answer': '<a>',
        'difficulty': 'easy',
        'explanation': 'The <a> tag creates hyperlinks in HTML.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'What is the box model in CSS?',
        'options': [
            'A model describing element layout with margin, border, padding, content',
            'A grid system',
            'A flexbox container',
            'A positioning system'
        ],
        'correct_answer': 'A model describing element layout with margin, border, padding, content',
        'difficulty': 'medium',
        'explanation': 'The CSS box model includes content, padding, border, and margin.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'Which JavaScript keyword declares a constant?',
        'options': ['var', 'let', 'const', 'static'],
        'correct_answer': 'const',
        'difficulty': 'easy',
        'explanation': 'const declares constants that cannot be reassigned.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'What is REST API?',
        'options': [
            'Representational State Transfer API',
            'Remote State Transfer API',
            'Restful Service Transfer API',
            'Resource State Transaction API'
        ],
        'correct_answer': 'Representational State Transfer API',
        'difficulty': 'medium',
        'explanation': 'REST is an architectural style for web services.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'Which method is used to prevent default form submission?',
        'options': [
            'preventDefault()',
            'stopPropagation()',
            'stopDefault()',
            'preventSubmit()'
        ],
        'correct_answer': 'preventDefault()',
        'difficulty': 'medium',
        'explanation': 'preventDefault() stops the default action of an event.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'What is the purpose of async/await in JavaScript?',
        'options': [
            'Handle asynchronous operations',
            'Create loops',
            'Define variables',
            'Style elements'
        ],
        'correct_answer': 'Handle asynchronous operations',
        'difficulty': 'medium',
        'explanation': 'async/await makes asynchronous code easier to write and read.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'Which CSS property controls text size?',
        'options': ['font-size', 'text-size', 'size', 'font-weight'],
        'correct_answer': 'font-size',
        'difficulty': 'easy',
        'explanation': 'font-size controls the size of text.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'What is JSON used for?',
        'options': [
            'Data interchange format',
            'Styling pages',
            'Database management',
            'Server configuration'
        ],
        'correct_answer': 'Data interchange format',
        'difficulty': 'easy',
        'explanation': 'JSON is a lightweight data interchange format.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'What does HTTP stand for?',
        'options': [
            'HyperText Transfer Protocol',
            'HyperText Transmission Protocol',
            'High Transfer Text Protocol',
            'HyperText Transport Protocol'
        ],
        'correct_answer': 'HyperText Transfer Protocol',
        'difficulty': 'easy',
        'explanation': 'HTTP is the protocol for transferring web pages.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'What is the virtual DOM in React?',
        'options': [
            'A lightweight copy of the real DOM',
            'A database',
            'A server',
            'A styling framework'
        ],
        'correct_answer': 'A lightweight copy of the real DOM',
        'difficulty': 'medium',
        'explanation': 'Virtual DOM improves performance by minimizing real DOM updates.'
    },
    {
        'job_role': 'Web Developer',
        'question': 'Which HTTP method is used to retrieve data?',
        'options': ['GET', 'POST', 'PUT', 'DELETE'],
        'correct_answer': 'GET',
        'difficulty': 'easy',
        'explanation': 'GET is used to retrieve data from a server.'
    },

    # ========================================
    # PYTHON DEVELOPER QUESTIONS (15 questions)
    # ========================================
    {
        'job_role': 'Python Developer',
        'question': 'What is the output of: print(type([]) is list)?',
        'options': ['True', 'False', 'None', 'Error'],
        'correct_answer': 'True',
        'difficulty': 'easy',
        'explanation': 'type([]) returns list class, and "is" checks identity.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'Which symbol defines a decorator in Python?',
        'options': ['@', '#', '$', '&'],
        'correct_answer': '@',
        'difficulty': 'medium',
        'explanation': 'Decorators are defined using the @ symbol.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What is PEP 8?',
        'options': [
            'Python style guide',
            'Python version',
            'Python library',
            'Python compiler'
        ],
        'correct_answer': 'Python style guide',
        'difficulty': 'easy',
        'explanation': 'PEP 8 is the official Python style guide.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What does the __init__ method do?',
        'options': [
            'Initialize class attributes',
            'Destroy objects',
            'Define static methods',
            'Create class variables'
        ],
        'correct_answer': 'Initialize class attributes',
        'difficulty': 'medium',
        'explanation': '__init__ is a constructor that initializes object attributes.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'Which method adds an element to a set?',
        'options': ['add()', 'append()', 'insert()', 'push()'],
        'correct_answer': 'add()',
        'difficulty': 'easy',
        'explanation': 'add() method adds elements to a set.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What is a lambda function?',
        'options': [
            'Anonymous function',
            'Named function',
            'Class method',
            'Static method'
        ],
        'correct_answer': 'Anonymous function',
        'difficulty': 'easy',
        'explanation': 'Lambda creates small anonymous functions.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What is list comprehension?',
        'options': [
            'Concise way to create lists',
            'List sorting method',
            'List deletion method',
            'List copying method'
        ],
        'correct_answer': 'Concise way to create lists',
        'difficulty': 'medium',
        'explanation': 'List comprehension provides a concise syntax for creating lists.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What is the difference between list and tuple?',
        'options': [
            'Lists are mutable, tuples are immutable',
            'Lists are immutable, tuples are mutable',
            'Both are mutable',
            'Both are immutable'
        ],
        'correct_answer': 'Lists are mutable, tuples are immutable',
        'difficulty': 'easy',
        'explanation': 'Lists can be changed, tuples cannot.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What does GIL stand for?',
        'options': [
            'Global Interpreter Lock',
            'General Interface Library',
            'Graphical Integration Layer',
            'Generic Input Library'
        ],
        'correct_answer': 'Global Interpreter Lock',
        'difficulty': 'hard',
        'explanation': 'GIL allows only one thread to execute Python bytecode at a time.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'Which keyword is used for exception handling?',
        'options': ['try', 'catch', 'throw', 'handle'],
        'correct_answer': 'try',
        'difficulty': 'easy',
        'explanation': 'Python uses try-except for exception handling.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What is pip?',
        'options': [
            'Package installer for Python',
            'Python interpreter',
            'Python IDE',
            'Python debugger'
        ],
        'correct_answer': 'Package installer for Python',
        'difficulty': 'easy',
        'explanation': 'pip is used to install Python packages.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What does *args allow in a function?',
        'options': [
            'Variable number of arguments',
            'Keyword arguments',
            'Default arguments',
            'Required arguments'
        ],
        'correct_answer': 'Variable number of arguments',
        'difficulty': 'medium',
        'explanation': '*args allows passing variable number of positional arguments.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What is a generator in Python?',
        'options': [
            'Function that yields values',
            'Class constructor',
            'Loop iterator',
            'Data structure'
        ],
        'correct_answer': 'Function that yields values',
        'difficulty': 'medium',
        'explanation': 'Generators yield values one at a time using yield keyword.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'What is the purpose of self in Python?',
        'options': [
            'Reference to instance of class',
            'Global variable',
            'Function parameter',
            'Module import'
        ],
        'correct_answer': 'Reference to instance of class',
        'difficulty': 'easy',
        'explanation': 'self refers to the instance of the class.'
    },
    {
        'job_role': 'Python Developer',
        'question': 'Which module is used for regular expressions?',
        'options': ['re', 'regex', 'regexp', 'pattern'],
        'correct_answer': 're',
        'difficulty': 'easy',
        'explanation': 'The re module provides regular expression operations.'
    },

    # ========================================
    # GENERAL QUESTIONS (10 questions)
    # ========================================
    {
        'job_role': 'General',
        'question': 'What does API stand for?',
        'options': [
            'Application Programming Interface',
            'Advanced Programming Interface',
            'Application Process Interface',
            'Automated Programming Interface'
        ],
        'correct_answer': 'Application Programming Interface',
        'difficulty': 'easy',
        'explanation': 'API allows different software to communicate.'
    },
    {
        'job_role': 'General',
        'question': 'Which data structure uses LIFO principle?',
        'options': ['Queue', 'Stack', 'Array', 'Linked List'],
        'correct_answer': 'Stack',
        'difficulty': 'easy',
        'explanation': 'Stack follows Last In First Out principle.'
    },
    {
        'job_role': 'General',
        'question': 'What is the time complexity of binary search?',
        'options': ['O(n)', 'O(log n)', 'O(n²)', 'O(1)'],
        'correct_answer': 'O(log n)',
        'difficulty': 'medium',
        'explanation': 'Binary search divides the search space in half each time.'
    },
    {
        'job_role': 'General',
        'question': 'What does SQL stand for?',
        'options': [
            'Structured Query Language',
            'Simple Query Language',
            'Standard Query Language',
            'System Query Language'
        ],
        'correct_answer': 'Structured Query Language',
        'difficulty': 'easy',
        'explanation': 'SQL is used for managing relational databases.'
    },
    {
        'job_role': 'General',
        'question': 'What is version control?',
        'options': [
            'System to track code changes',
            'Software versioning',
            'Code compiler',
            'Testing framework'
        ],
        'correct_answer': 'System to track code changes',
        'difficulty': 'easy',
        'explanation': 'Version control tracks changes to code over time.'
    },
    {
        'job_role': 'General',
        'question': 'What is Git?',
        'options': [
            'Distributed version control system',
            'Programming language',
            'Database',
            'Web framework'
        ],
        'correct_answer': 'Distributed version control system',
        'difficulty': 'easy',
        'explanation': 'Git tracks changes in source code.'
    },
    {
        'job_role': 'General',
        'question': 'What is agile methodology?',
        'options': [
            'Iterative software development approach',
            'Testing framework',
            'Programming language',
            'Database system'
        ],
        'correct_answer': 'Iterative software development approach',
        'difficulty': 'medium',
        'explanation': 'Agile emphasizes iterative development and collaboration.'
    },
    {
        'job_role': 'General',
        'question': 'What does OOP stand for?',
        'options': [
            'Object-Oriented Programming',
            'Object-Oriented Process',
            'Operational Object Programming',
            'Optimized Object Protocol'
        ],
        'correct_answer': 'Object-Oriented Programming',
        'difficulty': 'easy',
        'explanation': 'OOP is a programming paradigm based on objects.'
    },
    {
        'job_role': 'General',
        'question': 'What is debugging?',
        'options': [
            'Finding and fixing errors in code',
            'Writing code',
            'Testing code',
            'Deploying code'
        ],
        'correct_answer': 'Finding and fixing errors in code',
        'difficulty': 'easy',
        'explanation': 'Debugging is the process of finding and fixing bugs.'
    },
    {
        'job_role': 'General',
        'question': 'What is cloud computing?',
        'options': [
            'Delivering computing services over internet',
            'Local server hosting',
            'Desktop applications',
            'Mobile apps'
        ],
        'correct_answer': 'Delivering computing services over internet',
        'difficulty': 'easy',
        'explanation': 'Cloud computing provides on-demand computing resources.'
    },
]

# Add all questions
print("\n" + "="*60)
print("Adding Questions to Database")
print("="*60)

added_count = 0
for q in all_questions:
    try:
        db.add_question(
            job_role=q['job_role'],
            question=q['question'],
            options=q['options'],
            correct_answer=q['correct_answer'],
            difficulty=q['difficulty'],
            explanation=q['explanation']
        )
        added_count += 1
        print(f"✅ Added: {q['question'][:60]}...")
    except Exception as e:
        print(f"❌ Error adding question: {e}")

print("\n" + "="*60)
print(f"🎉 Successfully added {added_count} questions!")
print("="*60)

# Show summary
print("\nQuestions by Role:")
print("-" * 40)
print("Data Scientist: 15 questions")
print("Web Developer: 15 questions")
print("Python Developer: 15 questions")
print("General: 10 questions")
print("-" * 40)
print("Total: 55 questions")
print("\n✅ Your database is now ready with enough questions for testing!")