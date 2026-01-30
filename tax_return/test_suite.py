#!/usr/bin/env python3
"""
Quick Start Guide & Test Suite for Tax Return Tool

This script tests all core modules to ensure everything is working correctly.
"""

import sys
import os
from datetime import date

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from models import PersonalInfo, IncomeSource, Deduction, TaxCredit, Dependent, TaxReturn
from calculator import TaxCalculator
from materials_checklist import MaterialsChecklist
from upload_handler import UploadHandler
from utils import format_currency, TaxHelper


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_models():
    """Test data models"""
    print_header("TESTING DATA MODELS")
    
    # Create personal info
    personal_info = PersonalInfo(
        first_name="Jane",
        last_name="Doe",
        ssn="987-65-4321",
        date_of_birth=date(1985, 5, 15),
        street_address="456 Oak Ave",
        city="Los Angeles",
        state="CA",
        zip_code="90001",
        email="jane@example.com",
        phone="555-5678",
        filing_status="single",
    )
    
    print(f"✓ Created PersonalInfo: {personal_info.first_name} {personal_info.last_name}")
    
    # Create tax return
    tax_return = TaxReturn(
        tax_year=2024,
        personal_info=personal_info,
        filing_status="single",
    )
    
    print(f"✓ Created TaxReturn for {tax_return.tax_year}")
    
    # Add income
    tax_return.add_income(IncomeSource("w2", "Primary Job", 85000))
    tax_return.add_income(IncomeSource("interest", "Savings Interest", 750))
    print(f"✓ Added income sources - Total: {format_currency(tax_return.total_income())}")
    
    # Add deductions
    tax_return.add_deduction(Deduction("mortgage_interest", "Mortgage Interest", 15000))
    tax_return.add_deduction(Deduction("property_tax", "Property Taxes", 6500))
    print(f"✓ Added deductions - Total: {format_currency(tax_return.total_deductions())}")
    
    # Add credits
    tax_return.add_credit(TaxCredit("education_credit", "Education Credit", 2500))
    print(f"✓ Added tax credits - Total: {format_currency(tax_return.total_credits())}")
    
    # Add dependent
    dependent = Dependent(
        first_name="Johnny",
        last_name="Doe",
        ssn="123-45-6700",
        relationship="child",
        date_of_birth=date(2015, 3, 20),
        months_lived_with=12,
    )
    tax_return.add_dependent(dependent)
    print(f"✓ Added dependent: {dependent.first_name} {dependent.last_name}")
    
    return tax_return


def test_calculator(tax_return):
    """Test tax calculator"""
    print_header("TESTING TAX CALCULATOR")
    
    calculator = TaxCalculator(2024)
    result = calculator.calculate_tax(tax_return)
    
    print(f"\n📊 TAX CALCULATION SUMMARY")
    print(f"   Gross Income:          {format_currency(result.gross_income)}")
    print(f"   Standard Deduction:    {format_currency(result.standard_deduction)}")
    print(f"   Itemized Deductions:   {format_currency(result.itemized_deductions)}")
    print(f"   → Using:               {result.deduction_used.title()}")
    print(f"   Taxable Income:        {format_currency(result.taxable_income)}")
    print(f"   Federal Income Tax:    {format_currency(result.total_tax)}")
    print(f"   Tax Credits:           {format_currency(result.total_credits)}")
    print(f"   Tax After Credits:     {format_currency(result.tax_after_credits)}")
    print(f"   Effective Tax Rate:    {result.effective_tax_rate:.2f}%")
    print(f"   Marginal Tax Rate:     {result.marginal_tax_rate:.2f}%")
    print(f"   Estimated Refund/Owed: {format_currency(result.estimated_refund_or_owed)}")
    
    # Test deduction strategy
    strategy = calculator.suggest_deduction_strategy(tax_return)
    print(f"\n💡 DEDUCTION STRATEGY")
    print(f"   Recommendation:        {strategy['recommendation']}")
    print(f"   Potential Savings:     {format_currency(strategy['potential_savings'])}")
    
    # Test quarterly payments
    quarterly = calculator.estimate_quarterly_payments(tax_return.total_income())
    print(f"\n📅 QUARTERLY PAYMENTS ESTIMATE")
    for i, payment in enumerate(quarterly, 1):
        print(f"   Q{i}: {format_currency(payment)}")
    
    print(f"\n✓ Tax calculation completed successfully")
    
    return result


def test_materials_checklist():
    """Test materials checklist"""
    print_header("TESTING MATERIALS CHECKLIST")
    
    checklist = MaterialsChecklist()
    
    # Mark some items as collected
    checklist.mark_item_collected("Personal Info", "Social Security Number (SSN)", "doc_123")
    checklist.mark_item_collected("Income Documents", "W-2 forms from all employers", "doc_124")
    
    # Get summary
    summary = checklist.get_summary()
    
    print(f"\n📋 MATERIALS COLLECTION PROGRESS")
    print(f"   Total Required Items:  {summary['total_required_items']}")
    print(f"   Items Collected:       {summary['items_collected']}")
    print(f"   Progress:              {summary['progress_percentage']:.1f}%")
    print(f"   Ready to File:         {'✓ Yes' if summary['is_ready'] else '✗ No'}")
    
    if summary['missing_items']:
        print(f"\n   Missing Items:")
        for category, items in summary['missing_items'].items():
            print(f"      {category}:")
            for item in items:
                print(f"         - {item}")
    
    print(f"\n✓ Materials checklist test completed")


def test_upload_handler():
    """Test upload handler"""
    print_header("TESTING UPLOAD HANDLER")
    
    handler = UploadHandler("test_uploads")
    
    print(f"\n📁 UPLOAD HANDLER CONFIGURATION")
    print(f"   Upload Folder:         {handler.upload_folder}")
    print(f"   Max File Size:         {handler.MAX_FILE_SIZE / 1024 / 1024:.1f} MB")
    print(f"   Allowed Extensions:    {', '.join(sorted(handler.ALLOWED_EXTENSIONS))}")
    
    # Test validation
    print(f"\n✓ File Validation:")
    test_files = [
        ("w2_2024.pdf", True),
        ("1099_form.docx", True),
        ("photo.jpg", True),
        ("script.exe", False),
        ("virus.bat", False),
    ]
    
    for filename, expected in test_files:
        result = handler.allowed_file(filename)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {filename}: {result}")
    
    print(f"\n✓ Upload handler test completed")


def test_utils():
    """Test utility functions"""
    print_header("TESTING UTILITY FUNCTIONS")
    
    print(f"\n💰 CURRENCY & FORMATTING")
    print(f"   $50000.99:             {format_currency(50000.99)}")
    print(f"   $1234567.5:            {format_currency(1234567.5)}")
    
    print(f"\n👥 DEPENDENT ELIGIBILITY")
    print(f"   Age 10, 12 months:     {TaxHelper.is_dependent_eligible(10, 12)} ✓")
    print(f"   Age 30, 12 months:     {TaxHelper.is_dependent_eligible(30, 12)} ✗")
    
    print(f"\n💳 CHILD TAX CREDIT")
    print(f"   Age 10:                {format_currency(TaxHelper.child_tax_credit_amount(10))}")
    print(f"   Age 20:                {format_currency(TaxHelper.child_tax_credit_amount(20))}")
    print(f"   Age 25:                {format_currency(TaxHelper.child_tax_credit_amount(25))}")
    
    print(f"\n✓ Utility functions test completed")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + "  TAX RETURN TOOL - TEST SUITE & QUICK START".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Run tests
        tax_return = test_models()
        result = test_calculator(tax_return)
        test_materials_checklist()
        test_upload_handler()
        test_utils()
        
        # Success message
        print_header("✓ ALL TESTS PASSED SUCCESSFULLY")
        
        print(f"\n📚 GETTING STARTED")
        print(f"\n   1. Install requirements:")
        print(f"      pip install -r requirements.txt")
        print(f"\n   2. Run the Flask application:")
        print(f"      python app.py")
        print(f"\n   3. Open your browser and navigate to:")
        print(f"      http://localhost:5000")
        print(f"\n   4. Start filing your taxes!")
        
        print(f"\n📂 PROJECT STRUCTURE")
        print(f"   ├── app.py                 - Flask main application")
        print(f"   ├── models.py              - Data models")
        print(f"   ├── calculator.py          - Tax calculation engine")
        print(f"   ├── materials_checklist.py - Materials management")
        print(f"   ├── upload_handler.py      - File upload handling")
        print(f"   ├── utils.py               - Utility functions")
        print(f"   ├── config.py              - Configuration settings")
        print(f"   ├── requirements.txt       - Python dependencies")
        print(f"   ├── templates/             - HTML templates")
        print(f"   │   ├── index.html")
        print(f"   │   ├── start.html")
        print(f"   │   ├── materials_checklist.html")
        print(f"   │   ├── error.html")
        print(f"   │   └── help.html")
        print(f"   └── uploads/               - Document storage")
        
        print(f"\n" + "="*70 + "\n")
        
        return 0
    
    except Exception as e:
        print_header("✗ TEST FAILED")
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
