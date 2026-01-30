"""
Tax Return Tool - Flask Web Application

Main application entry point with web UI for tax return filing assistance.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from datetime import date, datetime
from werkzeug.utils import secure_filename

from models import (
    TaxReturn, PersonalInfo, IncomeSource, Deduction, TaxCredit, Dependent
)
from calculator import TaxCalculator
from upload_handler import UploadHandler
from materials_checklist import MaterialsChecklist
from utils import (
    format_currency, format_percentage, is_valid_ssn, is_valid_email,
    validate_tax_year, format_dict_as_json, TaxHelper
)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here_change_in_production'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize handlers
upload_handler = UploadHandler(app.config['UPLOAD_FOLDER'])


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/start', methods=['GET', 'POST'])
def start_filing():
    """Start a new tax return"""
    if request.method == 'POST':
        data = request.get_json()
        
        # Validate input
        errors = []
        
        if not data.get('first_name'):
            errors.append('First name is required')
        if not data.get('last_name'):
            errors.append('Last name is required')
        if not data.get('ssn') or not is_valid_ssn(data['ssn']):
            errors.append('Valid SSN is required')
        if not data.get('tax_year') or not validate_tax_year(int(data['tax_year'])):
            errors.append('Valid tax year is required')
        
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Create tax return object
        try:
            personal_info = PersonalInfo(
                first_name=data['first_name'],
                last_name=data['last_name'],
                ssn=data['ssn'],
                date_of_birth=datetime.strptime(data['dob'], '%Y-%m-%d').date(),
                street_address=data.get('address', ''),
                city=data.get('city', ''),
                state=data.get('state', ''),
                zip_code=data.get('zip', ''),
                email=data.get('email', ''),
                phone=data.get('phone', ''),
                filing_status=data.get('filing_status', 'single'),
            )
            
            tax_return = TaxReturn(
                tax_year=int(data['tax_year']),
                personal_info=personal_info,
                filing_status=data.get('filing_status', 'single'),
            )
            
            # Store in session (in production, use database)
            session['tax_return'] = {
                'tax_year': tax_return.tax_year,
                'filer_name': f"{personal_info.first_name} {personal_info.last_name}",
                'filing_status': tax_return.filing_status,
                'created_date': datetime.now().isoformat(),
            }
            
            # Store personal info separately for display
            session['personal_info'] = {
                'first_name': personal_info.first_name,
                'last_name': personal_info.last_name,
                'ssn': personal_info.ssn,
                'date_of_birth': personal_info.date_of_birth.strftime('%Y-%m-%d'),
                'street_address': personal_info.street_address,
                'city': personal_info.city,
                'state': personal_info.state,
                'zip_code': personal_info.zip_code,
                'email': personal_info.email,
                'phone': personal_info.phone,
                'filing_status': personal_info.filing_status,
            }
            
            return jsonify({'success': True, 'redirect': url_for('materials_checklist')})
        
        except Exception as e:
            return jsonify({'success': False, 'errors': [str(e)]}), 500
    
    return render_template('start.html')


@app.route('/materials-checklist')
def materials_checklist():
    """Materials checklist page"""
    if 'tax_return' not in session:
        return redirect(url_for('start_filing'))
    
    checklist = MaterialsChecklist()
    required = checklist.get_required_checklist()
    optional = checklist.get_optional_checklist()
    
    # Get personal info from session
    personal_info = session.get('personal_info', {})
    
    return render_template(
        'materials_checklist.html',
        required_materials=required,
        optional_materials=optional,
        filer_name=session['tax_return']['filer_name'],
        personal_info=personal_info,
    )


@app.route('/upload', methods=['POST'])
def upload_document():
    """Handle document upload"""
    if 'tax_return' not in session:
        return jsonify({'success': False, 'error': 'Session expired'}), 401
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    category = request.form.get('category', 'other')
    
    success, doc_id, result = upload_handler.upload_file(file, category)
    
    if success:
        return jsonify({
            'success': True,
            'document_id': doc_id,
            'file_path': result,
            'message': f'File uploaded successfully',
        })
    else:
        return jsonify({'success': False, 'error': result}), 400


@app.route('/income')
def income_input():
    """Income input page"""
    if 'tax_return' not in session:
        return redirect(url_for('start_filing'))
    
    income_types = [
        ('w2', 'W-2 Wage/Salary'),
        ('1099-nec', '1099-NEC (Self-Employment)'),
        ('1099-misc', '1099-MISC'),
        ('1099-int', 'Interest Income'),
        ('1099-div', 'Dividend Income'),
        ('1099-b', 'Capital Gains/Losses'),
        ('rental', 'Rental Income'),
        ('other', 'Other Income'),
    ]
    
    return render_template(
        'income.html',
        income_types=income_types,
        filer_name=session['tax_return']['filer_name'],
    )


@app.route('/api/income', methods=['GET', 'POST'])
def api_income():
    """API for income management"""
    if request.method == 'POST':
        data = request.get_json()
        
        # Validate
        if not data.get('amount') or float(data['amount']) < 0:
            return jsonify({'error': 'Invalid amount'}), 400
        
        income = {
            'source_type': data['source_type'],
            'description': data.get('description', ''),
            'amount': float(data['amount']),
            'document_id': data.get('document_id'),
        }
        
        # Store in session income list
        if 'income_sources' not in session:
            session['income_sources'] = []
        
        session['income_sources'].append(income)
        session.modified = True
        
        return jsonify({'success': True, 'income': income})
    
    # GET - return all income sources
    return jsonify({'income_sources': session.get('income_sources', [])})


@app.route('/deductions')
def deductions_input():
    """Deductions input page"""
    if 'tax_return' not in session:
        return redirect(url_for('start_filing'))
    
    deduction_types = [
        ('mortgage_interest', 'Mortgage Interest'),
        ('property_tax', 'Property Tax'),
        ('state_local_tax', 'State & Local Tax (SALT)'),
        ('charitable', 'Charitable Contributions'),
        ('medical', 'Medical & Dental'),
        ('student_loan', 'Student Loan Interest'),
        ('education', 'Education Expenses'),
        ('business', 'Business Expenses'),
        ('other', 'Other Deductions'),
    ]
    
    standard_deduction = {
        'single': 14600,
        'married_filing_jointly': 29200,
        'head_of_household': 21900,
    }
    
    filing_status = session['tax_return'].get('filing_status', 'single')
    
    return render_template(
        'deductions.html',
        deduction_types=deduction_types,
        standard_deduction=standard_deduction.get(filing_status, 14600),
        filer_name=session['tax_return']['filer_name'],
    )


@app.route('/credits')
def credits_input():
    """Tax credits input page"""
    if 'tax_return' not in session:
        return redirect(url_for('start_filing'))
    
    credit_types = [
        ('child_tax_credit', 'Child Tax Credit'),
        ('education_credit', 'American Opportunity/Lifetime Learning'),
        ('earned_income_credit', 'Earned Income Credit'),
        ('child_dependent_care', 'Child & Dependent Care'),
        ('adoption_credit', 'Adoption Credit'),
        ('retirement_savings', 'Retirement Savings Contribution'),
        ('other', 'Other Credits'),
    ]
    
    return render_template(
        'credits.html',
        credit_types=credit_types,
        filer_name=session['tax_return']['filer_name'],
    )


@app.route('/dependents')
def dependents_input():
    """Dependents input page"""
    if 'tax_return' not in session:
        return redirect(url_for('start_filing'))
    
    return render_template(
        'dependents.html',
        filer_name=session['tax_return']['filer_name'],
    )


@app.route('/calculate')
def calculate_tax():
    """Tax calculation page"""
    if 'tax_return' not in session:
        return redirect(url_for('start_filing'))
    
    try:
        # Reconstruct tax return from session data
        # This is simplified - in production, you'd load from database
        
        calculator = TaxCalculator(int(session['tax_return']['tax_year']))
        
        # Create a minimal tax return for display
        result = {
            'tax_year': session['tax_return']['tax_year'],
            'filer_name': session['tax_return']['filer_name'],
            'income_sources': session.get('income_sources', []),
            'deductions': session.get('deductions', []),
            'credits': session.get('credits', []),
        }
        
        return render_template('calculate.html', result=result)
    
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/summary')
def summary():
    """Summary and review page"""
    if 'tax_return' not in session:
        return redirect(url_for('start_filing'))
    
    return render_template('summary.html', filer_name=session['tax_return']['filer_name'])


@app.route('/help')
def help_page():
    """Help and FAQ page"""
    faqs = [
        {
            'question': 'What documents do I need to file my taxes?',
            'answer': 'You\'ll need W-2 forms, 1099 forms for various income, receipts for deductions, and identification.'
        },
        {
            'question': 'What\'s the difference between standard and itemized deductions?',
            'answer': 'Standard deduction is a fixed amount. Itemized deductions are your actual expenses. Use whichever is larger.'
        },
        {
            'question': 'Can I claim my adult child as a dependent?',
            'answer': 'Only if they meet specific requirements including age, residency, and income limits.'
        },
        {
            'question': 'What credits can I claim?',
            'answer': 'Common credits include Child Tax Credit, Education credits, Earned Income Credit, and Childcare credit.'
        },
    ]
    
    return render_template('help.html', faqs=faqs)


@app.route('/api/session-status')
def session_status():
    """Get current session status"""
    if 'tax_return' in session:
        return jsonify({
            'logged_in': True,
            'filer_name': session['tax_return'].get('filer_name'),
            'tax_year': session['tax_return'].get('tax_year'),
        })
    return jsonify({'logged_in': False})


@app.route('/logout')
def logout():
    """End session and logout"""
    session.clear()
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error='Server error'), 500


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
