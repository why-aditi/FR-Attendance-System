from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB setup using Motor
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client["employee_db"]
employees_collection = db["employees"]