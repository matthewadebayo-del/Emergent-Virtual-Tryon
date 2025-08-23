from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
from models import User, Product, TryOnResult

class Database:
    def __init__(self):
        self.client = None
        self.db = None
    
    async def connect(self):
        """Connect to MongoDB"""
        mongo_url = os.environ.get('MONGO_URL')
        db_name = os.environ.get('DB_NAME', 'virtual_tryon')
        
        self.client = AsyncIOMotorClient(mongo_url)
        self.db = self.client[db_name]
        
        # Create indexes
        await self.create_indexes()
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
    
    async def create_indexes(self):
        """Create database indexes for performance"""
        # User indexes
        await self.db.users.create_index("email", unique=True)
        await self.db.users.create_index("id", unique=True)
        
        # Product indexes
        await self.db.products.create_index("id", unique=True)
        await self.db.products.create_index("category")
        await self.db.products.create_index("brand")
        
        # Try-on result indexes
        await self.db.tryon_results.create_index("user_id")
        await self.db.tryon_results.create_index("product_id")
        await self.db.tryon_results.create_index("created_at")
    
    # User operations
    async def create_user(self, user: User) -> User:
        """Create a new user"""
        user_dict = user.dict()
        await self.db.users.insert_one(user_dict)
        return user
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        user_data = await self.db.users.find_one({"email": email})
        if user_data:
            return User(**user_data)
        return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        user_data = await self.db.users.find_one({"id": user_id})
        if user_data:
            return User(**user_data)
        return None
    
    async def update_user_measurements(self, user_id: str, measurements: Dict[str, Any]) -> bool:
        """Update user measurements"""
        result = await self.db.users.update_one(
            {"id": user_id},
            {
                "$set": {
                    "measurements": measurements,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
    
    async def update_user_profile_photo(self, user_id: str, photo_url: str) -> bool:
        """Update user profile photo"""
        result = await self.db.users.update_one(
            {"id": user_id},
            {
                "$set": {
                    "profile_photo": photo_url,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
    
    # Product operations
    async def create_product(self, product: Product) -> Product:
        """Create a new product"""
        product_dict = product.dict()
        await self.db.products.insert_one(product_dict)
        return product
    
    async def get_products(self, skip: int = 0, limit: int = 20, category: Optional[str] = None) -> List[Product]:
        """Get products with pagination and filtering"""
        filter_query = {}
        if category:
            filter_query["category"] = category
        
        cursor = self.db.products.find(filter_query).skip(skip).limit(limit)
        products = []
        async for product_data in cursor:
            products.append(Product(**product_data))
        return products
    
    async def get_product_by_id(self, product_id: str) -> Optional[Product]:
        """Get product by ID"""
        product_data = await self.db.products.find_one({"id": product_id})
        if product_data:
            return Product(**product_data)
        return None
    
    async def count_products(self, category: Optional[str] = None) -> int:
        """Count total products"""
        filter_query = {}
        if category:
            filter_query["category"] = category
        return await self.db.products.count_documents(filter_query)
    
    # Try-on result operations
    async def create_tryon_result(self, result: TryOnResult) -> TryOnResult:
        """Create a new try-on result"""
        result_dict = result.dict()
        await self.db.tryon_results.insert_one(result_dict)
        return result
    
    async def get_user_tryon_history(self, user_id: str, skip: int = 0, limit: int = 10) -> List[TryOnResult]:
        """Get user's try-on history"""
        cursor = self.db.tryon_results.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        results = []
        async for result_data in cursor:
            results.append(TryOnResult(**result_data))
        return results
    
    async def get_tryon_result_by_id(self, result_id: str) -> Optional[TryOnResult]:
        """Get try-on result by ID"""
        result_data = await self.db.tryon_results.find_one({"id": result_id})
        if result_data:
            return TryOnResult(**result_data)
        return None

# Global database instance
database = Database()