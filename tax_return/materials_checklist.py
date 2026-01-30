"""
Materials Checklist for Tax Return

Provides checklist of required and optional documents for tax filing.
"""

from typing import List, Dict


class MaterialsChecklist:
    """Manages tax material requirements and checklists"""
    
    # 2024 Tax Materials Checklist
    REQUIRED_MATERIALS = {
        "Personal Info": [
            "Social Security Number (SSN)",
            "Valid ID (Driver's License, Passport, etc.)",
            "Date of Birth",
            "Current Address",
        ],
        "Income Documents": [
            "W-2 forms from all employers",
            "1099-NEC or 1099-MISC (self-employment/contractor income)",
            "1099-INT (interest income)",
            "1099-DIV (dividend income)",
            "1099-B (capital gains/losses)",
        ],
        "Deductions": [
            "Mortgage interest statements (1098)",
            "Property tax statements",
            "State and local tax (SALT) records",
            "Charitable contribution receipts",
            "Medical expense records (if itemizing)",
        ],
        "Credits": [
            "Child Tax Credit documentation (SSN of children)",
            "Education expense records (tuition, fees)",
            "Childcare expense receipts",
        ],
        "Other": [
            "Prior year tax return (if available)",
            "Payment receipts for estimated taxes",
            "Dependent information (SSN, relationship, months lived)",
        ],
    }
    
    OPTIONAL_MATERIALS = {
        "Education": [
            "Student loan interest statements (1098-E)",
            "Education savings account statements",
        ],
        "Business": [
            "Business expense records",
            "Equipment purchase receipts",
            "Home office expense records",
        ],
        "Investments": [
            "Brokerage statements",
            "Dividend/interest statements",
            "Stock options/RSU documentation",
        ],
        "Other Deductions": [
            "Unreimbursed employee expense records",
            "Tax preparation fee receipts",
            "Mortgage points documentation",
        ],
    }
    
    def __init__(self):
        self.collected_items = {}
        self.notes = {}
    
    def get_required_checklist(self) -> Dict[str, List[str]]:
        """Get all required materials"""
        return self.REQUIRED_MATERIALS.copy()
    
    def get_optional_checklist(self) -> Dict[str, List[str]]:
        """Get all optional materials"""
        return self.OPTIONAL_MATERIALS.copy()
    
    def get_all_checklist(self) -> Dict[str, Dict[str, bool]]:
        """Get complete checklist with status"""
        checklist = {}
        
        # Add required items
        for category, items in self.REQUIRED_MATERIALS.items():
            if category not in checklist:
                checklist[category] = {}
            for item in items:
                item_key = f"{category}:{item}"
                checklist[category][item] = self.collected_items.get(item_key, False)
        
        # Add optional items
        for category, items in self.OPTIONAL_MATERIALS.items():
            if category not in checklist:
                checklist[category] = {}
            for item in items:
                item_key = f"{category}:{item}"
                checklist[category][item] = self.collected_items.get(item_key, False)
        
        return checklist
    
    def mark_item_collected(self, category: str, item: str, document_id: str = None):
        """Mark an item as collected"""
        item_key = f"{category}:{item}"
        self.collected_items[item_key] = True
        if document_id:
            self.notes[item_key] = document_id
    
    def mark_item_missing(self, category: str, item: str):
        """Mark an item as not collected"""
        item_key = f"{category}:{item}"
        self.collected_items[item_key] = False
        if item_key in self.notes:
            del self.notes[item_key]
    
    def get_collection_progress(self) -> Dict[str, float]:
        """Get progress percentage by category"""
        progress = {}
        
        for category in self.REQUIRED_MATERIALS.keys():
            items = self.REQUIRED_MATERIALS[category]
            collected = sum(1 for item in items if self.collected_items.get(f"{category}:{item}", False))
            progress[category] = (collected / len(items)) * 100 if items else 0
        
        return progress
    
    def get_missing_required_items(self) -> Dict[str, List[str]]:
        """Get list of missing required items"""
        missing = {}
        
        for category, items in self.REQUIRED_MATERIALS.items():
            missing_items = []
            for item in items:
                item_key = f"{category}:{item}"
                if not self.collected_items.get(item_key, False):
                    missing_items.append(item)
            
            if missing_items:
                missing[category] = missing_items
        
        return missing
    
    def is_ready_to_file(self) -> bool:
        """Check if all required materials are collected"""
        missing = self.get_missing_required_items()
        return len(missing) == 0
    
    def get_summary(self) -> Dict:
        """Get summary of collection status"""
        progress = self.get_collection_progress()
        missing = self.get_missing_required_items()
        total_collected = sum(1 for v in self.collected_items.values() if v)
        total_items = len(self.REQUIRED_MATERIALS)
        
        return {
            "total_required_items": total_items,
            "items_collected": total_collected,
            "progress_percentage": (total_collected / total_items * 100) if total_items > 0 else 0,
            "missing_items": missing,
            "is_ready": self.is_ready_to_file(),
            "progress_by_category": progress,
        }


# Pre-built checklist instance
standard_checklist = MaterialsChecklist()
