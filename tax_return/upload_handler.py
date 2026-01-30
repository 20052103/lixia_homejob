"""
Upload Handler for Tax Documents

Manages file uploads and document processing.
"""

import os
import uuid
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path


class UploadHandler:
    """Handles document uploads for tax return tool"""
    
    ALLOWED_EXTENSIONS = {
        'pdf', 'jpg', 'jpeg', 'png', 'gif',
        'doc', 'docx', 'xls', 'xlsx', 'csv',
        'txt', 'zip'
    }
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    
    def __init__(self, upload_folder: str = "uploads"):
        self.upload_folder = upload_folder
        self._create_upload_folder()
    
    def _create_upload_folder(self):
        """Create upload folder if it doesn't exist"""
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)
        
        # Create category folders
        categories = [
            "income", "deductions", "credits", "personal",
            "dependents", "other"
        ]
        for category in categories:
            Path(os.path.join(self.upload_folder, category)).mkdir(
                parents=True, exist_ok=True
            )
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def get_file_extension(self, filename: str) -> str:
        """Get file extension"""
        return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    def generate_document_id(self) -> str:
        """Generate unique document ID"""
        return f"doc_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def upload_file(self, file_obj, category: str = "other") -> Tuple[bool, str, str]:
        """
        Upload a file and return success status, document ID, and file path.
        
        Args:
            file_obj: File object from Flask request
            category: Document category (income, deductions, etc.)
        
        Returns:
            Tuple of (success: bool, document_id: str, file_path: str or error_message: str)
        """
        
        # Check if file exists
        if not file_obj or file_obj.filename == '':
            return False, "", "No file selected"
        
        # Check file extension
        if not self.allowed_file(file_obj.filename):
            return False, "", f"File type not allowed. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
        
        # Check file size
        file_obj.seek(0, os.SEEK_END)
        file_size = file_obj.tell()
        file_obj.seek(0)
        
        if file_size > self.MAX_FILE_SIZE:
            return False, "", f"File size exceeds maximum {self.MAX_FILE_SIZE / 1024 / 1024:.1f} MB"
        
        try:
            # Generate document ID and filename
            document_id = self.generate_document_id()
            extension = self.get_file_extension(file_obj.filename)
            filename = f"{document_id}.{extension}"
            
            # Create category folder if needed
            category_folder = os.path.join(self.upload_folder, category.lower())
            Path(category_folder).mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path = os.path.join(category_folder, filename)
            file_obj.save(file_path)
            
            return True, document_id, file_path
        
        except Exception as e:
            return False, "", f"Error uploading file: {str(e)}"
    
    def delete_file(self, document_id: str, category: str = "other") -> bool:
        """Delete an uploaded file"""
        
        try:
            category_folder = os.path.join(self.upload_folder, category.lower())
            
            # Find and delete file (we need to check all extensions since we don't know it)
            for filename in os.listdir(category_folder):
                if filename.startswith(document_id):
                    file_path = os.path.join(category_folder, filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        return True
            
            return False
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return False
    
    def get_document_info(self, document_id: str, category: str = "other") -> Optional[dict]:
        """Get information about an uploaded document"""
        
        try:
            category_folder = os.path.join(self.upload_folder, category.lower())
            
            for filename in os.listdir(category_folder):
                if filename.startswith(document_id):
                    file_path = os.path.join(category_folder, filename)
                    file_stat = os.stat(file_path)
                    
                    return {
                        "document_id": document_id,
                        "filename": filename,
                        "file_path": file_path,
                        "file_size": file_stat.st_size,
                        "upload_date": datetime.fromtimestamp(file_stat.st_ctime),
                        "category": category,
                    }
            
            return None
        except Exception as e:
            print(f"Error getting document info: {str(e)}")
            return None
    
    def list_documents(self, category: str = None) -> list:
        """List all uploaded documents, optionally filtered by category"""
        
        documents = []
        
        try:
            if category:
                categories = [category.lower()]
            else:
                categories = os.listdir(self.upload_folder)
            
            for cat in categories:
                cat_path = os.path.join(self.upload_folder, cat)
                if not os.path.isdir(cat_path):
                    continue
                
                for filename in os.listdir(cat_path):
                    file_path = os.path.join(cat_path, filename)
                    if os.path.isfile(file_path):
                        file_stat = os.stat(file_path)
                        document_id = filename.rsplit('.', 1)[0]
                        
                        documents.append({
                            "document_id": document_id,
                            "filename": filename,
                            "category": cat,
                            "file_size": file_stat.st_size,
                            "upload_date": datetime.fromtimestamp(file_stat.st_ctime),
                        })
            
            return documents
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []


def test_upload_handler():
    """Test upload handler"""
    handler = UploadHandler("test_uploads")
    
    print("Upload Handler Test")
    print("=" * 50)
    print(f"Upload folder: {handler.upload_folder}")
    print(f"Allowed extensions: {handler.ALLOWED_EXTENSIONS}")
    print(f"Max file size: {handler.MAX_FILE_SIZE / 1024 / 1024} MB")
    print("=" * 50)
    
    # Test file validation
    print("\nFile validation tests:")
    print(f"  'document.pdf' allowed: {handler.allowed_file('document.pdf')}")
    print(f"  'form.docx' allowed: {handler.allowed_file('form.docx')}")
    print(f"  'script.exe' allowed: {handler.allowed_file('script.exe')}")
    
    # Test document ID generation
    print("\nDocument ID generation:")
    for i in range(3):
        doc_id = handler.generate_document_id()
        print(f"  Generated ID {i+1}: {doc_id}")
    
    print("\n✓ Upload handler initialized successfully")


if __name__ == "__main__":
    test_upload_handler()
