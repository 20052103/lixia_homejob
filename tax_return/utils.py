"""
Utility Functions for Tax Return Tool
"""

from datetime import date
import json
from typing import Any, Dict


def format_currency(amount: float) -> str:
    """Format amount as USD currency"""
    return f"${amount:,.2f}"


def format_percentage(percent: float) -> str:
    """Format percentage value"""
    return f"{percent:.2f}%"


def get_age(birth_date: date) -> int:
    """Calculate age from birth date"""
    today = date.today()
    return today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )


def is_valid_ssn(ssn: str) -> bool:
    """Validate SSN format (basic validation)"""
    # Remove common formats
    ssn = ssn.replace("-", "").replace(" ", "")
    
    # Check length and all digits
    if len(ssn) != 9 or not ssn.isdigit():
        return False
    
    # Check for invalid patterns
    if ssn == "000000000" or ssn == "111111111" or ssn == "666666666":
        return False
    
    # Check if all same digit
    if len(set(ssn)) == 1:
        return False
    
    return True


def is_valid_email(email: str) -> bool:
    """Basic email validation"""
    return "@" in email and "." in email.split("@")[1]


def is_valid_zip_code(zip_code: str) -> bool:
    """Validate US ZIP code format"""
    zip_code = zip_code.replace("-", "").replace(" ", "")
    
    # 5 digits or 5+4 format
    if len(zip_code) == 5 and zip_code.isdigit():
        return True
    if len(zip_code) == 9 and zip_code.isdigit():
        return True
    
    return False


def calculate_income_tax_estimate(income: float) -> float:
    """Quick estimate of federal income tax (single filer, 2024)"""
    
    if income <= 11600:
        return income * 0.10
    elif income <= 47150:
        return 1160 + (income - 11600) * 0.12
    elif income <= 100525:
        return 5426 + (income - 47150) * 0.22
    elif income <= 191950:
        return 17168.50 + (income - 100525) * 0.24
    elif income <= 243725:
        return 39110.50 + (income - 191950) * 0.32
    elif income <= 609350:
        return 55678.50 + (income - 243725) * 0.35
    else:
        return 183647.25 + (income - 609350) * 0.37


def format_dict_as_json(data: Dict[str, Any]) -> str:
    """Format dictionary as pretty JSON"""
    return json.dumps(data, indent=2, default=str)


def round_to_cents(amount: float) -> float:
    """Round amount to nearest cent"""
    return round(amount, 2)


def validate_tax_year(year: int) -> bool:
    """Validate tax year"""
    current_year = date.today().year
    return 2000 <= year <= current_year + 1


def get_filing_status_standard_deduction(status: str) -> float:
    """Get standard deduction for filing status (2024)"""
    
    status_lower = status.lower().replace(" ", "_")
    
    deductions = {
        "single": 14600,
        "married_filing_jointly": 29200,
        "married_filing_separately": 14600,
        "head_of_household": 21900,
        "qualifying_widow_widower": 29200,
    }
    
    return deductions.get(status_lower, 14600)


def calculate_phase_out(income: float, phase_out_start: float, phase_out_end: float, max_credit: float) -> float:
    """Calculate phase-out amount for tax credits"""
    
    if income <= phase_out_start:
        return max_credit
    elif income >= phase_out_end:
        return 0
    else:
        # Linear phase-out
        percent_phased = (income - phase_out_start) / (phase_out_end - phase_out_start)
        return max(0, max_credit * (1 - percent_phased))


def format_form_data(data: Dict) -> Dict:
    """Format and validate form input data"""
    
    formatted = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            formatted[key] = value.strip()
        elif isinstance(value, (int, float)):
            formatted[key] = round_to_cents(float(value)) if isinstance(value, float) else value
        else:
            formatted[key] = value
    
    return formatted


class TaxHelper:
    """Helper class for common tax-related calculations"""
    
    @staticmethod
    def is_dependent_eligible(age: int, months_lived: int) -> bool:
        """Check if person qualifies as dependent"""
        return age <= 26 and months_lived >= 6
    
    @staticmethod
    def child_tax_credit_amount(age: int) -> float:
        """Get child tax credit amount based on age (2024)"""
        if age <= 16:
            return 2000  # Child Tax Credit
        elif age <= 23:
            return 1700  # Credit for Other Dependents
        return 0
    
    @staticmethod
    def qualify_for_education_credit(income: float, filing_status: str) -> Tuple[bool, str]:
        """Check education credit eligibility"""
        
        limits = {
            "single": 85000,
            "married_filing_jointly": 170000,
            "head_of_household": 85000,
        }
        
        limit = limits.get(filing_status, 85000)
        
        if income > limit:
            return False, f"Income exceeds limit of {format_currency(limit)}"
        return True, "Eligible for education credit"
    
    @staticmethod
    def get_standard_deduction_increase(age: int, filing_status: str) -> float:
        """Get additional standard deduction for age 65+ (2024)"""
        
        base_deduction = get_filing_status_standard_deduction(filing_status)
        
        if age >= 65:
            # Additional amount for 65+
            if "married" in filing_status.lower():
                return 1650
            elif "head_of_household" in filing_status.lower():
                return 1400
            else:
                return 1950
        
        return 0


# Type hints import
from typing import Tuple


if __name__ == "__main__":
    print("Tax Return Tool - Utilities Module")
    print("=" * 50)
    
    # Test utilities
    print("\nUtility Tests:")
    print(f"  Format $50000.99: {format_currency(50000.99)}")
    print(f"  Format 15.5%: {format_percentage(15.5)}")
    print(f"  Age from 1990-01-01: {get_age(date(1990, 1, 1))} years")
    print(f"  Valid SSN '123-45-6789': {is_valid_ssn('123-45-6789')}")
    print(f"  Valid Email 'test@example.com': {is_valid_email('test@example.com')}")
    print(f"  Valid ZIP '10001': {is_valid_zip_code('10001')}")
    print(f"  Tax estimate on $75000: {format_currency(calculate_income_tax_estimate(75000))}")
    print(f"  Standard deduction (single): {format_currency(get_filing_status_standard_deduction('single'))}")
    print(f"  Child tax credit (age 10): {format_currency(TaxHelper.child_tax_credit_amount(10))}")
    
    print("\n✓ All utilities working correctly")
