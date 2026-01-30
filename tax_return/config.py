"""
Configuration for Tax Return Tool
"""

import os
from datetime import date

# Flask Configuration
DEBUG = True
SECRET_KEY = 'change_this_in_production'
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx', 'xls', 'xlsx'}

# Tax Configuration
CURRENT_TAX_YEAR = date.today().year
EARLIEST_TAX_YEAR = 2000

# Standard Deductions (2024)
STANDARD_DEDUCTIONS = {
    'single': 14600,
    'married_filing_jointly': 29200,
    'married_filing_separately': 14600,
    'head_of_household': 21900,
    'qualifying_widow_widower': 29200,
}

# Tax Brackets (2024 - Single Filer)
TAX_BRACKETS = [
    (11600, 0.10),
    (47150, 0.12),
    (100525, 0.22),
    (191950, 0.24),
    (243725, 0.32),
    (609350, 0.35),
    (float('inf'), 0.37),
]

# Tax Credits
TAX_CREDITS = {
    'child_tax_credit': {
        'amount': 2000,
        'phase_out_start': 400000,
        'phase_out_end': 440000,
        'description': 'Child Tax Credit (age 0-16)',
    },
    'education_credit': {
        'amount': 2500,
        'phase_out_start': 80000,
        'phase_out_end': 90000,
        'description': 'American Opportunity Tax Credit',
    },
    'earned_income_credit': {
        'amount': 3995,  # Single filer, no children
        'phase_out_start': 40360,
        'phase_out_end': 50162,
        'description': 'Earned Income Tax Credit',
    },
}

# Application Settings
ITEMS_PER_PAGE = 10
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'tax_return_tool.log'

# Email Configuration (for notifications)
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

# Database Configuration (if using database)
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///tax_return.db'
SQLALCHEMY_TRACK_MODIFICATIONS = False
