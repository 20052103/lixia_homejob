"""
Tax Calculation Engine

Performs tax calculations based on income, deductions, and credits.
Includes 2024 US Federal Tax Rates.
"""

from models import TaxReturn, TaxCalculation
from typing import Tuple


class TaxCalculator:
    """Calculates federal income tax"""
    
    # 2024 Federal Tax Brackets (single filer)
    TAX_BRACKETS_2024 = {
        "single": [
            (11600, 0.10),
            (47150, 0.12),
            (100525, 0.22),
            (191950, 0.24),
            (243725, 0.32),
            (609350, 0.35),
            (float('inf'), 0.37),
        ],
        "married_filing_jointly": [
            (23200, 0.10),
            (94300, 0.12),
            (201050, 0.22),
            (383900, 0.24),
            (487450, 0.32),
            (731200, 0.35),
            (float('inf'), 0.37),
        ],
        "head_of_household": [
            (16550, 0.10),
            (63100, 0.12),
            (100500, 0.22),
            (191950, 0.24),
            (243700, 0.32),
            (609350, 0.35),
            (float('inf'), 0.37),
        ],
    }
    
    # 2024 Standard Deductions
    STANDARD_DEDUCTIONS_2024 = {
        "single": 14600,
        "married_filing_jointly": 29200,
        "married_filing_separately": 14600,
        "head_of_household": 21900,
    }
    
    def __init__(self, tax_year: int = 2024):
        self.tax_year = tax_year
    
    def calculate_tax(self, tax_return: TaxReturn) -> TaxCalculation:
        """Calculate complete tax return"""
        
        # Get filing status
        filing_status = tax_return.filing_status.lower().replace(" ", "_")
        
        # Calculate gross income
        gross_income = tax_return.total_income()
        
        # Get standard deduction
        standard_deduction = self.STANDARD_DEDUCTIONS_2024.get(filing_status, 14600)
        
        # Calculate itemized deductions
        itemized_deductions = tax_return.total_deductions()
        
        # Determine which deduction to use
        if itemized_deductions > standard_deduction:
            deduction_used = "itemized"
            total_deductions = itemized_deductions
        else:
            deduction_used = "standard"
            total_deductions = standard_deduction
        
        # Calculate taxable income
        taxable_income = max(0, gross_income - total_deductions)
        
        # Calculate federal income tax
        total_tax = self._calculate_federal_tax(taxable_income, filing_status)
        
        # Apply tax credits
        total_credits = tax_return.total_credits()
        tax_after_credits = max(0, total_tax - total_credits)
        
        # Estimate total withholding and refund/owed
        # (simplified - assumes no withholding or estimated payments for now)
        total_withholding = 0
        estimated_refund_or_owed = total_withholding - tax_after_credits
        
        # Calculate effective tax rate
        effective_tax_rate = (tax_after_credits / gross_income * 100) if gross_income > 0 else 0
        
        # Calculate marginal tax rate
        marginal_tax_rate = self._get_marginal_rate(taxable_income, filing_status)
        
        return TaxCalculation(
            tax_year=self.tax_year,
            gross_income=gross_income,
            total_deductions=total_deductions,
            standard_deduction=standard_deduction,
            itemized_deductions=itemized_deductions,
            deduction_used=deduction_used,
            taxable_income=taxable_income,
            total_tax=total_tax,
            total_credits=total_credits,
            tax_after_credits=tax_after_credits,
            total_withholding=total_withholding,
            estimated_refund_or_owed=estimated_refund_or_owed,
            effective_tax_rate=effective_tax_rate,
            marginal_tax_rate=marginal_tax_rate,
        )
    
    def _calculate_federal_tax(self, taxable_income: float, filing_status: str) -> float:
        """Calculate federal income tax based on brackets"""
        
        brackets = self.TAX_BRACKETS_2024.get(filing_status, self.TAX_BRACKETS_2024["single"])
        
        tax = 0.0
        previous_limit = 0
        
        for limit, rate in brackets:
            if taxable_income <= previous_limit:
                break
            
            # Calculate tax for this bracket
            income_in_bracket = min(taxable_income, limit) - previous_limit
            tax += income_in_bracket * rate
            previous_limit = limit
        
        return tax
    
    def _get_marginal_rate(self, taxable_income: float, filing_status: str) -> float:
        """Get marginal tax rate for given income"""
        
        brackets = self.TAX_BRACKETS_2024.get(filing_status, self.TAX_BRACKETS_2024["single"])
        
        for limit, rate in brackets:
            if taxable_income <= limit:
                return rate * 100
        
        return brackets[-1][1] * 100
    
    def suggest_deduction_strategy(self, tax_return: TaxReturn) -> dict:
        """Provide deduction strategy recommendations"""
        
        standard_deduction = self.STANDARD_DEDUCTIONS_2024.get(
            tax_return.filing_status.lower().replace(" ", "_"), 14600
        )
        itemized_deductions = tax_return.total_deductions()
        
        return {
            "standard_deduction": standard_deduction,
            "itemized_deductions": itemized_deductions,
            "difference": itemized_deductions - standard_deduction,
            "recommendation": "Use itemized deductions" if itemized_deductions > standard_deduction else "Use standard deduction",
            "potential_savings": abs(itemized_deductions - standard_deduction),
        }
    
    def estimate_quarterly_payments(self, annual_income: float) -> list:
        """Estimate quarterly estimated tax payments"""
        
        # Simplified calculation (should be more complex in real scenario)
        # Assumes 25% effective tax rate on self-employment income
        estimated_annual_tax = annual_income * 0.25
        quarterly_payment = estimated_annual_tax / 4
        
        return [quarterly_payment] * 4


def test_calculator():
    """Test tax calculation"""
    from models import PersonalInfo, IncomeSource, Deduction, TaxCredit
    from datetime import date
    
    # Create test return
    personal_info = PersonalInfo(
        first_name="John",
        last_name="Doe",
        ssn="123-45-6789",
        date_of_birth=date(1990, 1, 1),
        street_address="123 Main St",
        city="New York",
        state="NY",
        zip_code="10001",
        email="john@example.com",
        phone="555-1234",
        filing_status="single",
    )
    
    tax_return = TaxReturn(
        tax_year=2024,
        personal_info=personal_info,
        filing_status="single",
    )
    
    # Add income
    tax_return.add_income(IncomeSource("w2", "Salary", 75000))
    tax_return.add_income(IncomeSource("interest", "Interest Income", 500))
    
    # Add deductions
    tax_return.add_deduction(Deduction("mortgage_interest", "Mortgage Interest", 12000))
    tax_return.add_deduction(Deduction("property_tax", "Property Tax", 5000))
    
    # Add credits
    tax_return.add_credit(TaxCredit("child_tax_credit", "Child Tax Credit", 2000))
    
    # Calculate
    calculator = TaxCalculator(2024)
    result = calculator.calculate_tax(tax_return)
    
    print("\n" + "="*60)
    print("TAX CALCULATION RESULTS (2024)")
    print("="*60)
    print(f"Gross Income:             ${result.gross_income:>12,.2f}")
    print(f"Standard Deduction:       ${result.standard_deduction:>12,.2f}")
    print(f"Itemized Deductions:      ${result.itemized_deductions:>12,.2f}")
    print(f"Deduction Used:           {result.deduction_used:>20}")
    print(f"Taxable Income:           ${result.taxable_income:>12,.2f}")
    print(f"Federal Income Tax:       ${result.total_tax:>12,.2f}")
    print(f"Tax Credits:              ${result.total_credits:>12,.2f}")
    print(f"Tax After Credits:        ${result.tax_after_credits:>12,.2f}")
    print(f"Effective Tax Rate:       {result.effective_tax_rate:>20.2f}%")
    print(f"Marginal Tax Rate:        {result.marginal_tax_rate:>20.2f}%")
    print(f"Estimated Refund/Owed:    ${result.estimated_refund_or_owed:>12,.2f}")
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    test_calculator()
