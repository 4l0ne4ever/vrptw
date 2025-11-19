"""
Database service for high-level database operations.
"""

import json
from typing import List, Optional, Dict
from app.config.database import SessionLocal
from app.database import crud
from app.core.logger import setup_app_logger
from app.core.exceptions import DatabaseError

logger = setup_app_logger()


class DatabaseService:
    """Service for database operations."""
    
    def save_dataset(self, name: str, description: str, dataset_type: str,
                    data_dict: Dict, metadata: Optional[Dict] = None) -> int:
        """
        Save dataset to database.
        
        Args:
            name: Dataset name
            description: Dataset description
            dataset_type: Type of dataset ("hanoi_mockup" or "solomon")
            data_dict: Dataset data dictionary
            metadata: Optional metadata dictionary
            
        Returns:
            Dataset ID
        """
        db = SessionLocal()
        try:
            # Prepare data JSON
            data_json = json.dumps(data_dict, ensure_ascii=False)
            
            # Prepare metadata JSON
            metadata_json = None
            if metadata:
                metadata_json = json.dumps(metadata, ensure_ascii=False)
            
            # Create dataset
            dataset = crud.create_dataset(
                db=db,
                name=name,
                description=description,
                type=dataset_type,
                data_json=data_json,
                metadata_json=metadata_json
            )
            
            logger.info(f"Dataset saved: {name} (ID: {dataset.id})")
            return dataset.id
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            db.rollback()
            raise DatabaseError(f"Error saving dataset: {str(e)}")
        finally:
            db.close()
    
    def load_dataset(self, dataset_id: int) -> Optional[Dict]:
        """
        Load dataset from database.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset data dictionary or None if not found
        """
        db = SessionLocal()
        try:
            dataset = crud.get_dataset(db, dataset_id)
            
            if dataset is None:
                return None
            
            # Parse JSON data
            data_dict = json.loads(dataset.data_json)
            
            # Add metadata if available
            if dataset.metadata_json:
                metadata = json.loads(dataset.metadata_json)
                data_dict['metadata'] = metadata
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise DatabaseError(f"Error loading dataset: {str(e)}")
        finally:
            db.close()
    
    def get_all_datasets(self, dataset_type: Optional[str] = None, 
                        skip: int = 0, limit: int = 100) -> List[Dict]:
        """
        Get all datasets with optional filtering.
        
        Args:
            dataset_type: Optional filter by type
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            List of dataset dictionaries with metadata
        """
        db = SessionLocal()
        try:
            datasets = crud.get_datasets(db, skip=skip, limit=limit, type=dataset_type)
            
            result = []
            for dataset in datasets:
                # Parse metadata
                metadata = {}
                if dataset.metadata_json:
                    try:
                        metadata = json.loads(dataset.metadata_json)
                    except:
                        pass
                
                # Get basic info from data JSON
                data_dict = json.loads(dataset.data_json)
                num_customers = len(data_dict.get('customers', []))
                
                result.append({
                    'id': dataset.id,
                    'name': dataset.name,
                    'description': dataset.description,
                    'type': dataset.type,
                    'num_customers': num_customers,
                    'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
                    'updated_at': dataset.updated_at.isoformat() if dataset.updated_at else None,
                    'metadata': metadata
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting datasets: {e}")
            raise DatabaseError(f"Error getting datasets: {str(e)}")
        finally:
            db.close()
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """
        Delete dataset from database.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            True if deleted, False if not found
        """
        db = SessionLocal()
        try:
            result = crud.delete_dataset(db, dataset_id)
            if result:
                logger.info(f"Dataset deleted: ID {dataset_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            db.rollback()
            raise DatabaseError(f"Error deleting dataset: {str(e)}")
        finally:
            db.close()
    
    def save_distance_matrix(self, cache_key: str, coordinates: list, 
                            distance_matrix: 'np.ndarray', dataset_type: str,
                            use_real_routes: bool) -> bool:
        """
        Save distance matrix to database cache using binary serialization (version 2).
        
        Args:
            cache_key: Unique cache key (MD5 hash)
            coordinates: List of coordinate tuples
            distance_matrix: Numpy distance matrix
            dataset_type: Type of dataset ('hanoi' or 'solomon')
            use_real_routes: Whether OSRM was used
            
        Returns:
            True if saved successfully
        """
        import json
        import pickle
        import numpy as np
        
        db = SessionLocal()
        try:
            # Serialize coordinates as JSON (for metadata/verification)
            coordinates_json = json.dumps(coordinates, ensure_ascii=False)
            
            # Serialize distance matrix as binary (pickle) for version 2
            # This is 70-80% faster than JSON serialization
            distance_matrix_binary = pickle.dumps(distance_matrix, protocol=pickle.HIGHEST_PROTOCOL)
            
            num_points = len(coordinates)
            
            crud.create_distance_matrix_cache(
                db=db,
                cache_key=cache_key,
                coordinates_json=coordinates_json,
                distance_matrix_json=None,  # Legacy field, not used for version 2
                distance_matrix_binary=distance_matrix_binary,
                serialization_version=2,  # Use binary format
                dataset_type=dataset_type,
                use_real_routes=use_real_routes,
                num_points=num_points
            )
            
            logger.info(f"Distance matrix saved to database cache (binary): {cache_key} ({num_points} points)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving distance matrix to database: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def load_distance_matrix(self, cache_key: str) -> Optional['np.ndarray']:
        """
        Load distance matrix from database cache with version-aware deserialization.
        Supports both JSON (version 1) and binary (version 2) formats for backward compatibility.
        
        Args:
            cache_key: Unique cache key (MD5 hash)
            
        Returns:
            Distance matrix as numpy array, or None if not found
        """
        import json
        import pickle
        import numpy as np
        
        db = SessionLocal()
        try:
            cache_entry = crud.get_distance_matrix_cache(db, cache_key)
            
            if cache_entry is None:
                return None
            
            # Version-aware deserialization
            version = getattr(cache_entry, 'serialization_version', 1)
            
            if version == 2:
                # Binary format (pickle) - faster
                if cache_entry.distance_matrix_binary is None:
                    logger.warning(f"Cache entry {cache_key} has version 2 but no binary data")
                    return None
                distance_matrix = pickle.loads(cache_entry.distance_matrix_binary)
                logger.info(f"Distance matrix loaded from database cache (binary): {cache_key} ({cache_entry.num_points} points)")
            else:
                # Legacy JSON format (version 1) - for backward compatibility
                if cache_entry.distance_matrix_json is None:
                    logger.warning(f"Cache entry {cache_key} has version 1 but no JSON data")
                    return None
                distance_matrix_list = json.loads(cache_entry.distance_matrix_json)
                distance_matrix = np.array(distance_matrix_list)
                logger.info(f"Distance matrix loaded from database cache (JSON legacy): {cache_key} ({cache_entry.num_points} points)")
            
            return distance_matrix
            
        except Exception as e:
            logger.error(f"Error loading distance matrix from database: {e}")
            return None
        finally:
            db.close()

