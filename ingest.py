#!/usr/bin/env python3
"""
Batch catalog ingestion from CSV file with new data structure.

CSV Format (Required Columns):
- pinecone_id: Unique product identifier (used as Pinecone vector ID)
- image_url: URL to product image
- assigned_category: Category for detection and Pinecone namespace
- name_english: English product name
- name_arabic: Arabic product name
- price_amount: Product price (will be converted to int)
- price_unit: Currency code (SAR, USD, etc.)
- is_active: Product active status (boolean)
- store_id: Store identifier (will be converted to int)
- countries: Available countries (e.g., "[SA,AE]" → ["SA", "AE"])
- store: Store name
- product_url: URL to product page

Usage:
    python ingest.py --csv data/products.csv --batch-size 10
    python ingest.py --csv data/products.csv --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

# Metadata columns that will be stored in Pinecone
# Note: assigned_category from CSV will be stored as "category" in metadata
METADATA_COLS = [
    "image_url",
    "product_url",
    "name_english",
    "name_arabic",
    "category",  # This will be populated from assigned_category
    "price_amount",
    "price_unit",
    "is_active",
    "store_id",
    "countries",
    "store"
]

# Supported furniture categories
CATEGORIES = [
    "chair", "2-seater-sofa", "l-shape-sofa", "sofa",
    "bed", "bedspread", "pillow", "mattresses",
    "service-table", "center-table", "side-table", "console",
    "dressing-table", "comforter", "tv-table", "dining-table",
    "storage-box", "carpet", "flower-pot-and-plant", "statue-and-antique",
    "laundry-basket", "candle", "vase", "flower",
    "wall-clock", "shelve", "decorative-hanger", "lighting",
    "lampshade", "floor-stand", "wall-lighting", "outdoor-lighting",
    "chandelier", "pendant-lighting", "coffee-maker", "cooking-appliance",
    "food-processor", "cooking-pot", "serving-utensil-and-tray",
    "cup", "plate", "chaise-lounge", "art-canvas", "office-table", "office-chair"
]


def prepare_metadata(row: pd.Series) -> dict[str, Any]:
    """
    Prepare metadata dict from row with correct data types.
    Handles NaN values and type conversions per Pinecone requirements.
    
    Note: Maps 'assigned_category' from CSV to 'category' in metadata.
    """
    metadata = {}

    for col in METADATA_COLS:
        col_lower = col.lower()
        
        # Special handling: map 'category' metadata key to 'assigned_category' CSV column
        csv_col = 'assigned_category' if col_lower == 'category' else col_lower
        
        if csv_col in row.index:
            val = row[csv_col]

            # Handle NaN values with appropriate defaults
            if pd.isna(val):
                if col_lower in ['store_id', 'price_amount']:
                    metadata[col] = 0
                elif col_lower == 'is_active':
                    metadata[col] = False
                elif col_lower == 'countries':
                    metadata[col] = []
                else:
                    metadata[col] = ""
                continue

            # Type conversions based on field
            if col_lower == 'store_id':
                # Convert to integer
                try:
                    metadata[col] = int(float(val))
                except (ValueError, TypeError):
                    metadata[col] = 0

            elif col_lower == 'price_amount':
                # Convert to integer (599, not 599.00)
                try:
                    metadata[col] = int(float(val))
                except (ValueError, TypeError):
                    metadata[col] = 0

            elif col_lower == 'is_active':
                # Convert to boolean
                if isinstance(val, str):
                    metadata[col] = val.lower() in ['true', '1', 'yes']
                elif isinstance(val, bool):
                    metadata[col] = val
                else:
                    metadata[col] = bool(val)

            elif col_lower == 'countries':
                # Convert to list of strings: ["SA", "QA"]
                if isinstance(val, list):
                    metadata[col] = val
                elif isinstance(val, str):
                    # Parse "[SA]" or "[SA,QA]" → ["SA"] or ["SA", "QA"]
                    countries_str = val.strip('[]')
                    if countries_str:
                        metadata[col] = [c.strip() for c in countries_str.split(',')]
                    else:
                        metadata[col] = []
                else:
                    metadata[col] = []

            else:
                # All other fields as strings
                metadata[col] = str(val)

    return metadata


class CatalogIngester:
    """Batch catalog ingestion from CSV with new data structure."""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Statistics
        self.total = 0
        self.success = 0
        self.failed = 0
        self.failed_items: list[dict[str, Any]] = []
    
    def read_csv(self, csv_path: str) -> pd.DataFrame:
        """Read CSV file and validate required columns."""
        path = Path(csv_path)
        
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Validate required columns
        required_cols = ['pinecone_id', 'image_url', 'assigned_category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate metadata columns (warn if missing)
        missing_metadata = [col.lower() for col in METADATA_COLS if col.lower() not in df.columns]
        if missing_metadata:
            logger.warning(f"Missing optional metadata columns: {missing_metadata}")
        
        logger.info(f"Loaded {len(df)} products from CSV")
        return df
    
    async def download_image(self, image_url: str) -> bytes:
        """Download image from URL with validation."""
        try:
            logger.debug(f"Downloading image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Check file size
            size_mb = len(response.content) / (1024 * 1024)
            if size_mb > 15:
                raise ValueError(f"Image too large: {size_mb:.1f}MB (max 15MB)")
            
            logger.debug(f"Downloaded {size_mb:.2f}MB")
            return response.content
            
        except Exception as e:
            raise ValueError(f"Failed to download image from {image_url}: {str(e)}")
    
    async def upsert_catalog_item(
        self,
        pinecone_id: str,
        assigned_category: str,
        image_bytes: bytes,
        metadata: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> dict:
        """Upload product to catalog with new data structure."""
        # Prepare form data
        files = {
            "image": ("image.jpg", image_bytes, "image/jpeg"),
        }
        
        # Send only required fields to API
        data = {
            "pinecone_id": pinecone_id,
            "assigned_category": assigned_category,
            # Metadata is sent as JSON string
            "metadata_json": json.dumps(metadata),
        }
        
        try:
            response = await client.post(
                f"{self.api_base_url}/api/v1/catalog/upsert",
                files=files,
                data=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError:
            # Re-raise HTTPStatusError to preserve status code for retry logic
            raise
        except Exception as e:
            raise ValueError(f"Failed to upsert: {str(e)}")
    
    async def process_item(
        self,
        row: pd.Series,
        client: httpx.AsyncClient,
    ) -> bool:
        """Process a single item with retries."""
        # Convert pinecone_id from float to int, then to string
        try:
            pinecone_id = str(int(float(row['pinecone_id'])))
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid pinecone_id: {row.get('pinecone_id')} - {e}")
            return False
        
        image_url = str(row['image_url'])
        assigned_category = str(row['assigned_category'])
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Step 1: Prepare metadata with correct types
                metadata = prepare_metadata(row)
                
                # Step 2: Download image (no retry on download failures)
                logger.debug(f"[{pinecone_id}] Downloading image")
                try:
                    image_bytes = await self.download_image(image_url)
                except Exception as download_error:
                    # Immediately fail on download errors (404, network issues, etc.) - no retry
                    logger.error(
                        f"[{pinecone_id}] ❌ Download failed (no retry): {download_error}"
                    )
                    self.failed_items.append({
                        "pinecone_id": pinecone_id,
                        "image_url": image_url,
                        "error": f"Download failed: {str(download_error)}",
                    })
                    return False
                
                # Step 3: Upload to catalog
                logger.debug(f"[{pinecone_id}] Uploading to catalog (category: {assigned_category})")
                result = await self.upsert_catalog_item(
                    pinecone_id=pinecone_id,
                    assigned_category=assigned_category,
                    image_bytes=image_bytes,
                    metadata=metadata,
                    client=client,
                )
                
                logger.success(
                    f"[{pinecone_id}] ✅ Successfully ingested (category: {assigned_category})"
                )
                return True
                
            except httpx.HTTPStatusError as e:
                # Check if it's a validation error (400) - no need to retry
                if e.response.status_code == 400:
                    error_detail = e.response.text
                    logger.error(
                        f"[{pinecone_id}] ❌ Validation failed (no retry): {error_detail}"
                    )
                    self.failed_items.append({
                        "pinecone_id": pinecone_id,
                        "image_url": image_url,
                        "error": f"Validation error: {error_detail}",
                    })
                    return False
                
                # For other HTTP errors (5xx, network issues), retry
                if attempt < self.max_retries:
                    logger.warning(f"[{pinecone_id}] Attempt {attempt} failed: {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"[{pinecone_id}] ❌ Failed after {self.max_retries} attempts: {e}")
                    self.failed_items.append({
                        "pinecone_id": pinecone_id,
                        "image_url": image_url,
                        "error": str(e),
                    })
                    return False
            
            except Exception as e:
                # For non-HTTP errors, retry
                if attempt < self.max_retries:
                    logger.warning(f"[{pinecone_id}] Attempt {attempt} failed: {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"[{pinecone_id}] ❌ Failed after {self.max_retries} attempts: {e}")
                    self.failed_items.append({
                        "pinecone_id": pinecone_id,
                        "image_url": image_url,
                        "error": str(e),
                    })
                    return False
        
        return False
    
    async def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 10,
    ) -> None:
        """Process all items in batches."""
        self.total = len(df)
        
        async with httpx.AsyncClient() as client:
            # Process in batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                
                logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} items)")
                
                # Process batch concurrently
                tasks = [self.process_item(row, client) for _, row in batch.iterrows()]
                results = await asyncio.gather(*tasks)
                
                # Update statistics
                self.success += sum(results)
                self.failed += len(results) - sum(results)
                
                logger.info(
                    f"Batch complete. Success: {sum(results)}/{len(results)}, "
                    f"Total: {self.success}/{self.total}"
                )
    
    def print_summary(self) -> None:
        """Print ingestion summary."""
        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Total products:     {self.total}")
        print(f"✅ Successfully ingested: {self.success}")
        print(f"❌ Failed:          {self.failed}")
        if self.total > 0:
            print(f"Success rate:       {self.success / self.total * 100:.1f}%")
        print("=" * 60)
        
        if self.failed_items:
            print("\nFailed Items:")
            print("-" * 60)
            for item in self.failed_items[:10]:  # Show first 10
                print(f"  ID: {item['pinecone_id']}")
                print(f"  URL: {item['image_url']}")
                print(f"  Error: {item['error']}")
                print("-" * 60)
            
            if len(self.failed_items) > 10:
                print(f"  ... and {len(self.failed_items) - 10} more")
            
            # Save failed items to file
            failed_df = pd.DataFrame(self.failed_items)
            failed_path = "failed_ingestions.csv"
            failed_df.to_csv(failed_path, index=False)
            print(f"\n❌ Failed items saved to: {failed_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch catalog ingestion from CSV with new data structure."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file with required columns: pinecone_id, image_url, assigned_category, and metadata columns",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Backend API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of items to process concurrently (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="API timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per item (default: 3)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add("ingestion.log", level="DEBUG", rotation="10 MB")
    
    # Create ingester
    ingester = CatalogIngester(
        api_base_url=args.api_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    
    try:
        # Read CSV
        df = ingester.read_csv(args.csv)
        
        # Process all items
        logger.info(f"Starting batch ingestion ({len(df)} products)")
        await ingester.process_batch(df, batch_size=args.batch_size)
        
        # Print summary
        ingester.print_summary()
        
        # Exit with error code if any failures
        sys.exit(0 if ingester.failed == 0 else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
