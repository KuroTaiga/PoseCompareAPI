import os
import shutil
import uuid
import json
from datetime import datetime
from pathlib import Path

class FileManager:
    """
    Manages file organization for user sessions, uploads, and processed results
    """
    
    def __init__(self, base_dir="storage"):
        """
        Initialize the file manager with base directory
        
        Args:
            base_dir (str): Base directory for all storage
        """
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.uploads_dir = self.base_dir / "uploads"
        self.results_dir = self.base_dir / "results"
        
        # Create base directories if they don't exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_user_session(self):
        """
        Create a new user session
        
        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Create session metadata file
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "uploads": [],
            "results": []
        }
        
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create subdirectories for this session
        user_upload_dir = self.uploads_dir / session_id
        user_results_dir = self.results_dir / session_id
        
        user_upload_dir.mkdir(exist_ok=True)
        user_results_dir.mkdir(exist_ok=True)
        
        print(f"Created new user session: {session_id}")
        return session_id
    
    def save_upload(self, session_id, file, original_filename=None):
        """
        Save an uploaded file for a specific session
        
        Args:
            session_id (str): Session ID
            file: File object to save
            original_filename (str, optional): Original filename
            
        Returns:
            dict: File information including path and ID
            or tuple(None, str): None and error message if failed
        """
        # Validate session
        session_dir = self.sessions_dir / session_id
        if not session_dir.exists():
            error_msg = f"Session {session_id} does not exist"
            print(f"Error: {error_msg}")
            return None, f"Session expired. Please refresh the page."
        
        # Create a unique ID for this upload
        upload_id = str(uuid.uuid4())
        
        # Get file extension
        if original_filename:
            _, ext = os.path.splitext(original_filename)
        else:
            _, ext = os.path.splitext(file.filename)
        
        # Create filename
        filename = f"{upload_id}{ext}"
        
        # Save file to session's upload directory
        upload_dir = self.uploads_dir / session_id
        upload_path = upload_dir / filename
        
        # Clear previous uploads if requested
        self._clear_previous_uploads(session_id)
        
        try:
            # Save the file
            file.save(str(upload_path))
            
            # Update session metadata
            self._update_session_metadata(session_id, "uploads", {
                "id": upload_id,
                "original_filename": original_filename or file.filename,
                "filename": filename,
                "path": str(upload_path),
                "uploaded_at": datetime.now().isoformat()
            })
            
            print(f"Saved upload {upload_id} for session {session_id}")
            
            return {
                "session_id": session_id,
                "upload_id": upload_id,
                "filename": filename,
                "path": str(upload_path),
                "original_filename": original_filename or file.filename
            }, None
        except Exception as e:
            error_msg = f"Failed to save upload: {str(e)}"
            print(f"Error: {error_msg}")
            return None, "Failed to save your upload. Please try again."
    
    def save_result(self, session_id, upload_id, result_type, original_path, metadata=None):
        """
        Save a result file for a specific upload
        
        Args:
            session_id (str): Session ID
            upload_id (str): Upload ID
            result_type (str): Type of result (e.g., 'pose', 'heatmap')
            original_path (str): Path to the result file
            metadata (dict, optional): Additional metadata
            
        Returns:
            dict: Result information
            or tuple(None, str): None and error message if failed
        """
        # Validate session
        session_dir = self.sessions_dir / session_id
        if not session_dir.exists():
            error_msg = f"Session {session_id} does not exist"
            print(f"Error: {error_msg}")
            return None, "Session expired. Please refresh the page."
        
        # Create result ID
        result_id = str(uuid.uuid4())
        
        # Get file extension
        _, ext = os.path.splitext(original_path)
        
        # Create filename
        filename = f"{result_type}_{upload_id}_{result_id}{ext}"
        
        # Copy result to session's results directory
        results_dir = self.results_dir / session_id
        result_path = results_dir / filename
        
        try:
            shutil.copy2(original_path, result_path)
            
            # Create result info
            result_info = {
                "id": result_id,
                "upload_id": upload_id,
                "type": result_type,
                "filename": filename,
                "path": str(result_path),
                "created_at": datetime.now().isoformat()
            }
            
            # Add additional metadata
            if metadata:
                result_info["metadata"] = metadata
            
            # Update session metadata
            self._update_session_metadata(session_id, "results", result_info)
            
            print(f"Saved result {result_id} for upload {upload_id} in session {session_id}")
            
            return result_info, None
        except Exception as e:
            error_msg = f"Error copying result file: {str(e)}"
            print(f"Error: {error_msg}")
            return None, "Failed to save processing result. Please try again."
    
    def get_session_details(self, session_id):
        """
        Get details about a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            dict: Session details including uploads and results
            or tuple(None, str): None and error message if failed
        """
        session_file = self.sessions_dir / session_id / "metadata.json"
        if not session_file.exists():
            error_msg = f"Session {session_id} metadata does not exist"
            print(f"Error: {error_msg}")
            return None, "Session not found. A new session will be created."
        
        try:
            with open(session_file, "r") as f:
                return json.load(f), None
        except Exception as e:
            error_msg = f"Error reading session metadata: {str(e)}"
            print(f"Error: {error_msg}")
            return None, "Session data is corrupted. A new session will be created."
    
    def get_session_results(self, session_id, upload_id=None, result_type=None):
        """
        Get results for a session, optionally filtered by upload ID or result type
        
        Args:
            session_id (str): Session ID
            upload_id (str, optional): Filter by upload ID
            result_type (str, optional): Filter by result type
            
        Returns:
            list: Filtered results
            or tuple([], str): Empty list and error message if failed
        """
        session_data, error = self.get_session_details(session_id)
        if not session_data:
            return [], error or "No results found"
        
        results = session_data.get("results", [])
        
        # Apply filters
        if upload_id:
            results = [r for r in results if r.get("upload_id") == upload_id]
        
        if result_type:
            results = [r for r in results if r.get("type") == result_type]
        
        return results, None
    
    def get_result_path(self, session_id, result_id):
        """
        Get the path to a specific result file
        
        Args:
            session_id (str): Session ID
            result_id (str): Result ID
            
        Returns:
            str: Path to the result file or None if not found
            or tuple(None, str): None and error message if failed
        """
        session_data, error = self.get_session_details(session_id)
        if not session_data:
            return None, error or "Result not found"
        
        for result in session_data.get("results", []):
            if result.get("id") == result_id:
                return result.get("path"), None
        
        return None, "Result not found"
    
    def get_upload_path(self, session_id, upload_id):
        """
        Get the path to a specific upload file
        
        Args:
            session_id (str): Session ID
            upload_id (str): Upload ID
            
        Returns:
            str: Path to the upload file or None if not found
            or tuple(None, str): None and error message if failed
        """
        session_data, error = self.get_session_details(session_id)
        if not session_data:
            return None, error or "Upload not found"
        
        for upload in session_data.get("uploads", []):
            if upload.get("id") == upload_id:
                return upload.get("path"), None
        
        return None, "Upload not found"
    
    def _update_session_metadata(self, session_id, key, value):
        """
        Update session metadata
        
        Args:
            session_id (str): Session ID
            key (str): Metadata key to update
            value: Value to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        metadata_file = self.sessions_dir / session_id / "metadata.json"
        
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            if isinstance(metadata.get(key), list):
                metadata[key].append(value)
            else:
                metadata[key] = value
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
                
            return True
        except Exception as e:
            error_msg = f"Error updating session metadata: {str(e)}"
            print(f"Error: {error_msg}")
            return False
    
    def _clear_previous_uploads(self, session_id):
        """
        Clear previous uploads for a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        session_data, error = self.get_session_details(session_id)
        if not session_data:
            print(f"Error clearing previous uploads: {error}")
            return False
        
        # Get previous uploads
        previous_uploads = session_data.get("uploads", [])
        
        # Get previous results
        previous_results = session_data.get("results", [])
        
        # Delete previous upload files
        for upload in previous_uploads:
            upload_path = upload.get("path")
            if upload_path and os.path.exists(upload_path):
                try:
                    os.remove(upload_path)
                    print(f"Deleted previous upload: {upload_path}")
                except Exception as e:
                    print(f"Error deleting file {upload_path}: {str(e)}")
        
        # Delete previous result files
        for result in previous_results:
            result_path = result.get("path")
            if result_path and os.path.exists(result_path):
                try:
                    os.remove(result_path)
                    print(f"Deleted previous result: {result_path}")
                except Exception as e:
                    print(f"Error deleting file {result_path}: {str(e)}")
        
        # Reset session metadata for uploads and results
        metadata_file = self.sessions_dir / session_id / "metadata.json"
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            metadata["uploads"] = []
            metadata["results"] = []
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error resetting session metadata: {str(e)}")
            return False

    def get_latest_upload(self, session_id):
        """
        Get the most recent upload for a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            dict: Upload information or None if no uploads
            or tuple(None, str): None and error message if failed
        """
        session_data, error = self.get_session_details(session_id)
        if not session_data:
            return None, error or "No uploads found"
        
        uploads = session_data.get("uploads", [])
        if not uploads:
            return None, "No uploads found"
        
        # Return the most recent upload (last in the list)
        return uploads[-1], None