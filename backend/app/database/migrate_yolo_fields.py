from sqlalchemy import text
from app.database.connection import engine


def migrate_images_table():
    """
    Safely adds detected_objects_count and bounding_boxes columns
    to the images table if they do not already exist.
    """
    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE images ADD COLUMN IF NOT EXISTS detected_objects_count INTEGER DEFAULT 0;")
        )
        connection.execute(
            text("ALTER TABLE images ADD COLUMN IF NOT EXISTS bounding_boxes JSONB;")
        )
    print("Database migration successful: detected_objects_count and bounding_boxes columns added or verified.")


if __name__ == "__main__":
    migrate_images_table()
