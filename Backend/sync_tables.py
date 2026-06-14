"""
Sync database tables — run this after any model changes.
Creates all new tables and columns defined in models.py.
"""
import sys
import os

# Add Backend folder to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db.database import engine
from app.db import models

print("Syncing database tables...")
models.Base.metadata.create_all(bind=engine)
print("Done! All tables are up to date.")
print("\nTables created/verified:")
for table_name in models.Base.metadata.tables.keys():
    print(f"  ✓ {table_name}")
