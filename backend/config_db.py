# config_db.py
# Database Configuration for Resume Analyzer MCQ System

"""
CONFIGURATION INSTRUCTIONS:
1. Find your SQL Server instance name (usually 'localhost\\SQLEXPRESS')
2. Choose authentication method (Windows Auth recommended for local)
3. Update the DB_CONFIG dictionary below
"""

# ========================================
# DATABASE CONFIGURATION
# ========================================

DB_CONFIG = {
    # SQL Server instance name
    # Common values:
    # - 'localhost\\SQLEXPRESS' (most common)
    # - '.\\SQLEXPRESS'
    # - 'localhost' (default instance)
    # - 'YOUR-PC-NAME\\SQLEXPRESS'
    'server': 'HP-VICTUS\SQLEXPRESS',
    
    # Database name (will be created automatically)
    'database': 'ResumeAnalyzerDB',
    
    # Authentication method
    # True = Windows Authentication (recommended for local)
    # False = SQL Server Authentication (requires username/password)
    'use_windows_auth': True,
    
    # SQL Server credentials (only if use_windows_auth = False)
    'username': 'sa',
    'password': 'YourPassword123!',
}

# ========================================
# HOW TO FIND YOUR SQL SERVER NAME
# ========================================
"""
Method 1: Using SQL Server Management Studio (SSMS)
- Open SSMS
- The server name is shown in the connection dialog
- Example: localhost\SQLEXPRESS

Method 2: Using Command Prompt
- Open CMD
- Run: sqlcmd -L
- This lists all SQL Server instances on your network

Method 3: Windows Services
- Press Win + R
- Type: services.msc
- Look for "SQL Server (SQLEXPRESS)" or "SQL Server (MSSQLSERVER)"
- The name in parentheses is your instance name
"""

# ========================================
# AUTHENTICATION OPTIONS
# ========================================

# Option 1: Windows Authentication (Recommended for local development)
EXAMPLE_WINDOWS_AUTH = {
    'server': 'localhost\\SQLEXPRESS',
    'database': 'ResumeAnalyzerDB',
    'use_windows_auth': True,
}

# Option 2: SQL Server Authentication
EXAMPLE_SQL_AUTH = {
    'server': 'localhost\\SQLEXPRESS',
    'database': 'ResumeAnalyzerDB',
    'use_windows_auth': False,
    'username': 'sa',
    'password': 'YourStrongPassword123!',
}

# Option 3: Remote SQL Server
EXAMPLE_REMOTE = {
    'server': '192.168.1.100\\SQLEXPRESS',
    'database': 'ResumeAnalyzerDB',
    'use_windows_auth': False,
    'username': 'remote_user',
    'password': 'RemotePassword123!',
}

# Option 4: Azure SQL Database
EXAMPLE_AZURE = {
    'server': 'yourserver.database.windows.net',
    'database': 'ResumeAnalyzerDB',
    'use_windows_auth': False,
    'username': 'azureuser',
    'password': 'AzurePassword123!',
}

# ========================================
# USAGE IN CODE
# ========================================
"""
# In database.py, import and use like this:

from config_db import DB_CONFIG

class Database:
    def __init__(self):
        self.server = DB_CONFIG['server']
        self.database = DB_CONFIG['database']
        self.use_windows_auth = DB_CONFIG['use_windows_auth']
        self.username = DB_CONFIG.get('username', '')
        self.password = DB_CONFIG.get('password', '')
"""