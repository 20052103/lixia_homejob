"""
Data Models for Tax Return Tool

Defines the structure for user information, tax data, and filing details.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import date


@dataclass
class PersonalInfo:
    """User's personal information"""
    first_name: str
    last_name: str
    ssn: str  # Social Security Number
    date_of_birth: date
    street_address: str
    city: str
    state: str
    zip_code: str
    email: str
    phone: str
    filing_status: str  # "single", "married", "head_of_household", etc.


@dataclass
class IncomeSource:
    """Individual income source"""
    source_type: str  # "w2", "1099", "interest", "dividends", "capital_gains", etc.
    description: str
    amount: float
    document_id: Optional[str] = None
    
    def __post_init__(self):
        self.amount = round(self.amount, 2)


@dataclass
class Deduction:
    """Individual deduction"""
    category: str  # "mortgage_interest", "property_tax", "charitable", "medical", etc.
    description: str
    amount: float
    document_id: Optional[str] = None
    
    def __post_init__(self):
        self.amount = round(self.amount, 2)


@dataclass
class TaxCredit:
    """Individual tax credit"""
    credit_type: str  # "child_tax_credit", "education_credit", "earned_income", etc.
    description: str
    amount: float
    
    def __post_init__(self):
        self.amount = round(self.amount, 2)


@dataclass
class Dependent:
    """Tax dependent information"""
    first_name: str
    last_name: str
    ssn: str
    relationship: str  # "child", "parent", "sibling", etc.
    date_of_birth: date
    months_lived_with: int  # 0-12
    
    
@dataclass
class TaxReturn:
    """Complete tax return information"""
    tax_year: int
    personal_info: PersonalInfo
    income_sources: List[IncomeSource] = field(default_factory=list)
    deductions: List[Deduction] = field(default_factory=list)
    tax_credits: List[TaxCredit] = field(default_factory=list)
    dependents: List[Dependent] = field(default_factory=list)
    uploaded_documents: Dict[str, str] = field(default_factory=dict)  # document_id -> file_path
    filing_status: str = "single"
    created_date: date = field(default_factory=date.today)
    
    def total_income(self) -> float:
        """Calculate total income"""
        return sum(source.amount for source in self.income_sources)
    
    def total_deductions(self) -> float:
        """Calculate total deductions"""
        return sum(deduction.amount for deduction in self.deductions)
    
    def total_credits(self) -> float:
        """Calculate total credits"""
        return sum(credit.amount for credit in self.tax_credits)
    
    def add_income(self, source: IncomeSource):
        """Add income source"""
        self.income_sources.append(source)
    
    def add_deduction(self, deduction: Deduction):
        """Add deduction"""
        self.deductions.append(deduction)
    
    def add_credit(self, credit: TaxCredit):
        """Add tax credit"""
        self.tax_credits.append(credit)
    
    def add_dependent(self, dependent: Dependent):
        """Add dependent"""
        self.dependents.append(dependent)


@dataclass
class TaxCalculation:
    """Calculated tax return results"""
    tax_year: int
    gross_income: float
    total_deductions: float
    standard_deduction: float
    itemized_deductions: float
    deduction_used: str  # "standard" or "itemized"
    taxable_income: float
    total_tax: float
    total_credits: float
    tax_after_credits: float
    total_withholding: float
    estimated_refund_or_owed: float  # positive = refund, negative = owed
    effective_tax_rate: float
    marginal_tax_rate: float
    
    def __post_init__(self):
        # Round all values to 2 decimal places
        for field_name in self.__dataclass_fields__:
            if isinstance(getattr(self, field_name), float):
                setattr(self, field_name, round(getattr(self, field_name), 2))
