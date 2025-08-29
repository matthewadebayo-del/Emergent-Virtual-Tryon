"""
Firebase Database Operations
Wrapper for Firestore operations to replace MongoDB syntax
"""

import asyncio
from typing import Dict, Any, List, Optional
from firebase_admin import firestore
import logging
import uuid

logger = logging.getLogger(__name__)

class FirebaseDB:
    """Firebase Firestore database wrapper with MongoDB-like interface"""
    
    def __init__(self, db_client=None):
        self.db = db_client
        self.mock_mode = db_client is None
        if self.mock_mode:
            logger.warning("Running in mock database mode - data will not persist")
            self._mock_data = {
                'users': {},
                'products': {},
                'tryon_results': {}
            }
    
    def collection(self, collection_name: str):
        """Get a collection reference"""
        if self.mock_mode:
            return MockCollection(collection_name, self._mock_data)
        if self.db:
            return FirestoreCollection(self.db.collection(collection_name))
        return MockCollection(collection_name, self._mock_data)

    @property
    def users(self):
        return self.collection('users')
    
    @property
    def products(self):
        return self.collection('products')
    
    @property
    def tryon_results(self):
        return self.collection('tryon_results')


class FirestoreCollection:
    """Firestore collection with MongoDB-like interface"""
    
    def __init__(self, collection_ref):
        self.collection_ref = collection_ref
    
    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find one document matching the query"""
        try:
            docs = self.collection_ref.limit(1)
            for field, value in query.items():
                docs = docs.where(field, '==', value)
            
            results = docs.stream()
            for doc in results:
                data = doc.to_dict()
                data['_id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Error in find_one: {e}")
            return None
    
    async def insert_one(self, document: Dict[str, Any]) -> str:
        """Insert one document"""
        try:
            doc_data = document.copy()
            doc_id = doc_data.pop('_id', None) or doc_data.get('id', str(uuid.uuid4()))
            
            doc_ref = self.collection_ref.document(doc_id)
            doc_ref.set(doc_data)
            return doc_ref.id
        except Exception as e:
            logger.error(f"Error in insert_one: {e}")
            raise
    
    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Update one document matching the query"""
        try:
            doc = await self.find_one(query)
            if not doc:
                return False
            
            update_data = {}
            if '$set' in update:
                update_data.update(update['$set'])
            
            doc_ref = self.collection_ref.document(doc['_id'])
            doc_ref.update(update_data)
            return True
        except Exception as e:
            logger.error(f"Error in update_one: {e}")
            return False
    
    def find(self, query: Optional[Dict[str, Any]] = None):
        """Find documents matching the query"""
        return FirestoreQuery(self.collection_ref, query or {})


class FirestoreQuery:
    """Firestore query with MongoDB-like interface"""
    
    def __init__(self, collection_ref, query: Dict[str, Any]):
        self.collection_ref = collection_ref
        self.query = query
    
    async def to_list(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Convert query results to list"""
        try:
            docs = self.collection_ref.limit(limit)
            
            for field, value in self.query.items():
                docs = docs.where(field, '==', value)
            
            results = []
            for doc in docs.stream():
                data = doc.to_dict()
                data['_id'] = doc.id
                results.append(data)
            
            return results
        except Exception as e:
            logger.error(f"Error in to_list: {e}")
            return []


class MockCollection:
    """Mock collection for development without Firebase"""
    
    def __init__(self, collection_name: str, mock_data: Dict):
        self.collection_name = collection_name
        self.mock_data = mock_data
        if collection_name not in self.mock_data:
            self.mock_data[collection_name] = {}
    
    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Mock find_one implementation"""
        collection = self.mock_data[self.collection_name]
        for doc_id, doc in collection.items():
            if all(doc.get(k) == v for k, v in query.items()):
                result = doc.copy()
                result['_id'] = doc_id
                return result
        return None
    
    async def insert_one(self, document: Dict[str, Any]) -> str:
        """Mock insert_one implementation"""
        doc_id = document.get('id', str(uuid.uuid4()))
        doc_data = document.copy()
        doc_data.pop('_id', None)
        self.mock_data[self.collection_name][doc_id] = doc_data
        return doc_id
    
    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Mock update_one implementation"""
        doc = await self.find_one(query)
        if not doc:
            return False
        
        doc_id = doc['_id']
        if '$set' in update:
            self.mock_data[self.collection_name][doc_id].update(update['$set'])
        
        return True
    
    def find(self, query: Optional[Dict[str, Any]] = None):
        """Mock find implementation"""
        return MockQuery(self.collection_name, self.mock_data, query or {})


class MockQuery:
    """Mock query for development"""
    
    def __init__(self, collection_name: str, mock_data: Dict, query: Dict[str, Any]):
        self.collection_name = collection_name
        self.mock_data = mock_data
        self.query = query
    
    async def to_list(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Mock to_list implementation"""
        collection = self.mock_data[self.collection_name]
        results = []
        
        for doc_id, doc in collection.items():
            if all(doc.get(k) == v for k, v in self.query.items()):
                result = doc.copy()
                result['_id'] = doc_id
                results.append(result)
                if len(results) >= limit:
                    break
        
        return results
