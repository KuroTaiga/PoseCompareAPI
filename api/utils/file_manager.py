import os
import shutil
import uuid
import json
from datetime import datetime
from pathlib import Path

class FileManager:
    """
    Manages file organization for user sessions, uploads, and processed results
    
    Storage structure:
    /storage/
        /sessions/
            /<session_id>/
                metadata.json
                /<upload_id>/
                    original.mp4  # Original uploaded video
                    mediapipe_butterworth.mp4  # Example processed output
                    sapiens_2b_original.mp4  # Example processed output
                    ... other processing results
    """
    
    def __init__(self, base_dir="storage"):
        """
        Initialize the file manager with base directory
        
        Args:
            base_dir (str): Base directory for all storage
        """
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        
        # Create base directories if they don't exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Create upload directory within the session
        upload_dir = session_dir / upload_id
        upload_dir.mkdir(exist_ok=True)
        
        # Save file with a standardized name (original.ext)
        filename = f"original{ext}"
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
                "dir": str(upload_dir),
                "uploaded_at": datetime.now().isoformat()
            })
            
            print(f"Saved upload {upload_id} for session {session_id}")
            
            return {
                "session_id": session_id,
                "upload_id": upload_id,
                "filename": filename,
                "path": str(upload_path),
                "dir": str(upload_dir),
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
            result_type (str): Type of result (e.g., 'mediapipe_butterworth')
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
        
        # Validate upload directory
        upload_dir = session_dir / upload_id
        if not upload_dir.exists():
            error_msg = f"Upload {upload_id} does not exist"
            print(f"Error: {error_msg}")
            return None, "Upload not found. Please try again."
        
        # Create result ID
        result_id = str(uuid.uuid4())
        
        # Get file extension
        _, ext = os.path.splitext(original_path)
        
        # Create result filename (e.g., mediapipe_butterworth.mp4)
        filename = f"{result_type}{ext}"
        result_path = upload_dir / filename
        
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
    
    def get_upload_directory(self, session_id, upload_id):
        """
        Get the directory path for a specific upload
        
        Args:
            session_id (str): Session ID
            upload_id (str): Upload ID
            
        Returns:
            str: Path to the upload directory or None if not found
            or tuple(None, str): None and error message if failed
        """
        session_data, error = self.get_session_details(session_id)
        if not session_data:
            return None, error or "Upload not found"
        
        for upload in session_data.get("uploads", []):
            if upload.get("id") == upload_id:
                return upload.get("dir"), None
        
        return None, "Upload not found"
    
    def get_all_files_for_upload(self, session_id, upload_id):
        """
        Get all files (original and processed) for a specific upload
        
        Args:
            session_id (str): Session ID
            upload_id (str): Upload ID
            
        Returns:
            dict: Dictionary of files with type as key and path as value
            or tuple(None, str): None and error message if failed
        """
        upload_dir, error = self.get_upload_directory(session_id, upload_id)
        if not upload_dir:
            return None, error
        
        # Get the directory as Path object
        dir_path = Path(upload_dir)
        if not dir_path.exists():
            return None, "Upload directory not found"
        
        # Get all files in the directory
        files = {}
        for file_path in dir_path.glob("*"):
            if file_path.is_file():
                file_type = file_path.stem  # Use filename without extension as type
                files[file_type] = str(file_path)
        
        return files, None
    
    def delete_session(self, session_id):
        """
        Delete a session and all its files
        
        Args:
            session_id (str): Session ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        session_dir = self.sessions_dir / session_id
        if not session_dir.exists():
            print(f"Session {session_id} does not exist, nothing to delete")
            return True
        
        try:
            shutil.rmtree(session_dir)
            print(f"Deleted session {session_id} and all its files")
            return True
        except Exception as e:
            print(f"Error deleting session {session_id}: {str(e)}")
            return False
    
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
        
        # Delete previous upload directories
        for upload in previous_uploads:
            upload_dir = upload.get("dir")
            if upload_dir and os.path.exists(upload_dir):
                try:
                    shutil.rmtree(upload_dir)
                    print(f"Deleted previous upload directory: {upload_dir}")
                except Exception as e:
                    print(f"Error deleting directory {upload_dir}: {str(e)}")
        
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

    def cleanup_expired_sessions(self, max_age_hours=5):
        """
        Clean up expired sessions
        
        Args:
            max_age_hours (int): Maximum age of sessions in hours
            
        Returns:
            int: Number of sessions deleted
        """
        now = datetime.now()
        deleted_count = 0
        
        # Iterate through all session directories
        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            metadata_file = session_dir / "metadata.json"
            if not metadata_file.exists():
                # No metadata, just delete the directory
                shutil.rmtree(session_dir)
                deleted_count += 1
                continue
            
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Get creation time
                created_at = datetime.fromisoformat(metadata.get("created_at", "2000-01-01T00:00:00"))
                
                # Calculate age in hours
                age_hours = (now - created_at).total_seconds() / 3600
                
                # Delete if older than max age
                if age_hours > max_age_hours:
                    shutil.rmtree(session_dir)
                    deleted_count += 1
                    print(f"Deleted expired session {session_dir.name}, age: {age_hours:.1f} hours")
            except Exception as e:
                print(f"Error processing session {session_dir.name}: {str(e)}")
        
        return deleted_count