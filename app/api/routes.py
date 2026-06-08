import os
import shutil
import logging
import asyncio
import zipfile
import uuid
import datetime
from io import BytesIO
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import pillow_heif
import requests
import psycopg2
from io import BytesIO

# Register HEIC opener for PIL
pillow_heif.register_heif_opener()

# Database imports
from app.db.database import init_db, get_db, serialize_encoding, deserialize_encoding, hash_password
# Face matcher core
from app.services.face_recognition import extract_reference_encoding, FaceRecognitionError
from app.services.storage import process_and_save_uploaded_image, upload_to_telegram
from app.services.telegram_helpers import get_telegram_file_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FaceRecognitionAPI")

from fastapi import APIRouter

router = APIRouter()

# Directories (moved to parent storage folder to prevent Uvicorn reload loops)
from app.core.config import settings


# --- DATABASE AND MIGRATIONS ---
@router.on_event("startup")
async def startup_event():
    logger.info("Initializing services and tables...")
    init_db()
    # Start auto-expiry scheduler task
    asyncio.create_task(expiry_cleanup_scheduler())

@router.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down backend services...")

# --- BACKGROUND AI PROCESSING ---

def scan_and_save_faces(photo_id: int, file_id: str):
    import face_recognition
    try:
        from app.services.face_recognition import load_image_rgb, resize_image_if_large
        from io import BytesIO
        import requests
        import numpy as np
        from PIL import Image
        
        # Download image from Telegram
        download_url = get_telegram_file_url(file_id)
        response = requests.get(download_url)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        rgb_image = np.array(image)
        rgb_image = resize_image_if_large(rgb_image, max_width=1024)
        
        # CNN model is heavily optimized for side-poses and extreme angles
        face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model="cnn")
        encodings = []
        if face_locations:
            # High jitters for maximum robustness (lenient/powerful matching)
            encodings = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=10)
            
        with get_db() as conn:
            conn.execute("DELETE FROM face_encodings WHERE photo_id = ?", (photo_id,))
            for enc in encodings:
                enc_blob = serialize_encoding(enc)
                conn.execute(
                    "INSERT INTO face_encodings (photo_id, encoding) VALUES (?, ?)",
                    (photo_id, enc_blob)
                )
            conn.execute(
                "UPDATE event_photos SET faces_scanned = 1, faces_count = ? WHERE id = ?",
                (len(encodings), photo_id)
            )
        logger.info(f"AI Ingestion: Scanned photo ID {photo_id}. Found {len(encodings)} faces.")
    except Exception as e:
        logger.error(f"AI Ingestion error on photo ID {photo_id}: {e}", exc_info=True)
        with get_db() as conn:
            conn.execute("UPDATE event_photos SET faces_scanned = -1 WHERE id = ?", (photo_id,))

async def ingestion_worker(event_id: str):
    """
    Sequentially processes all unscanned images for the specified event in a thread pool.
    """
    logger.info(f"Starting ingestion worker for event: {event_id}")
    while True:
        photo = None
        with get_db() as conn:
            row = conn.execute(
                "SELECT id, file_id FROM event_photos WHERE event_id = ? AND faces_scanned = 0 LIMIT 1",
                (event_id,)
            ).fetchone()
            if row:
                photo = dict(row)
                
        if not photo:
            break
            
        photo_id = photo["id"]
        file_id = photo["file_id"]
        
        # Run face detector without blocking FastAPI async loop
        await asyncio.to_thread(scan_and_save_faces, photo_id, file_id)
        
    logger.info(f"Finished ingestion worker for event: {event_id}")

# --- AUTO EXPIRY SYSTEM ---
async def expiry_cleanup_scheduler():
    """Daily scheduler checking for expired events and removing their resources."""
    while True:
        try:
            logger.info("Auto-Expiry: Running checks for expired events...")
            today = datetime.date.today().isoformat()
            
            expired_events = []
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT id, name FROM events WHERE auto_expiry = 1 AND TO_DATE(date, 'YYYY-MM-DD') <= ?::date - INTERVAL '30 days'",
                    (today,)
                ).fetchall()
                for row in rows:
                    expired_events.append(dict(row))
                    
            for event in expired_events:
                event_id = event["id"]
                logger.info(f"Auto-Expiry: Event '{event['name']}' ({event_id}) has expired. Cleaning up...")
                
                # Delete photos folder
                    
                # Delete database records (Cascade ON delete will clear event_photos, face_encodings, analytics)
                with get_db() as conn:
                    conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
                    
                logger.info(f"Auto-Expiry: Cleanup complete for event: {event_id}")
        except Exception as e:
            logger.error(f"Auto-Expiry task encountered error: {e}", exc_info=True)
            
        # Run every 24 hours
        await asyncio.sleep(24 * 3600)


# --- DIAGNOSTIC TELEMETRY LOGGERS ---
def log_audit_action(username: str, action: str, duration_ms: int, payload_size: int, status_str: str):
    """Inserts a diagnostic telemetry entry into SQLite log tables."""
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (username, action, duration_ms, payload_size, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, action, duration_ms, payload_size, status_str)
            )
    except Exception as e:
        logger.error(f"Failed to record diagnostic telemetry: {e}")


# --- STORAGE QUOTA HELPERS ---
def get_total_storage_used() -> int:
    """Queries total size of all uploaded event photos in bytes."""
    with get_db() as conn:
        row = conn.execute("SELECT SUM(file_size) as total_size FROM event_photos").fetchone()
        return row["total_size"] or 0

def get_admin_storage_used(username: str) -> int:
    """Queries size of uploaded photos belonging to a specific photographer owner."""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT SUM(ep.file_size) as total_size 
            FROM event_photos ep
            JOIN events e ON ep.event_id = e.id
            WHERE e.owner_username = ?
            """,
            (username,)
        ).fetchone()
        return row["total_size"] or 0

@router.get("/storage-quota")
async def get_storage_quota(username: str = Query(...)):
    """Returns storage used by photographer vs their specific plan limit and billing status."""
    used = get_admin_storage_used(username)
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT allocated_storage_bytes, custom_storage_bytes, 
                   subscription_expires_at, processing_priority 
            FROM users WHERE username = ?
            """,
            (username,)
        ).fetchone()
        
    limit = 53687091200
    sub_expiry = None
    custom_bytes = None
    priority = "normal"
    
    if row:
        priority = row["processing_priority"] or "normal"
        sub_expiry = row["subscription_expires_at"]
        custom_bytes = row["custom_storage_bytes"]
        limit = custom_bytes if custom_bytes is not None else (row["allocated_storage_bytes"] or 53687091200)

    # Check if subscription has expired
    subscription_expired = False
    if sub_expiry:
        try:
            expiry_date = datetime.datetime.strptime(sub_expiry, "%Y-%m-%d").date()
            if datetime.date.today() > expiry_date:
                subscription_expired = True
        except Exception as e:
            logger.error(f"Expiry parse error in storage-quota: {e}")
            
    return {
        "status": "success",
        "used_bytes": used,
        "max_bytes": limit,
        "used_gb": round(used / (1024 * 1024 * 1024), 3),
        "max_gb": round(limit / (1024 * 1024 * 1024), 1),
        "subscription_expires_at": sub_expiry,
        "subscription_expired": subscription_expired,
        "custom_storage_bytes": custom_bytes,
        "processing_priority": priority
    }


# --- BILLING KILL SWITCH VALIDATORS ---
def check_event_active(event_id: str):
    """
    Validates if the owning photographer is Active, subscription is not expired,
    and event is Active. Raises 403 Forbidden on failure.
    """
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT u.status, u.username, u.subscription_expires_at, e.status as event_status
            FROM events e
            LEFT JOIN users u ON e.owner_username = u.username
            WHERE e.id = ?
            """,
            (event_id,)
        ).fetchone()
        
    if not row:
        return
        
    # 1. Billing Active check
    if row["status"] == "suspended":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gallery Suspended. This wedding album has been temporarily suspended by the provider. Please contact the photographer."
        )
        
    # 2. Subscription Expiry check
    sub_expiry = row["subscription_expires_at"]
    if sub_expiry:
        try:
            expiry_date = datetime.datetime.strptime(sub_expiry, "%Y-%m-%d").date()
            if datetime.date.today() > expiry_date:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Gallery Suspended. The photographer's subscription has expired. Please contact the photographer."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Expiry validation error: {e}")
            
    # 3. Event Status check
    if row["event_status"] == "paused":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gallery Paused. This wedding gallery has been temporarily paused by the photographer."
        )
    elif row["event_status"] == "unpublished":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gallery Private. This wedding gallery has been unpublished by the photographer."
        )

def check_photo_active(photo_id: int):
    """
    Validates if the owning photographer of the photo is active and subscription is not expired,
    and event is not paused/unpublished. Raises 403.
    """
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT u.status, u.subscription_expires_at, e.status as event_status
            FROM event_photos ep
            JOIN events e ON ep.event_id = e.id
            LEFT JOIN users u ON e.owner_username = u.username
            WHERE ep.id = ?
            """,
            (photo_id,)
        ).fetchone()
        
    if not row:
        return
        
    # 1. Billing Active check
    if row["status"] == "suspended":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gallery Suspended. This wedding album has been temporarily suspended."
        )
        
    # 2. Subscription Expiry check
    sub_expiry = row["subscription_expires_at"]
    if sub_expiry:
        try:
            expiry_date = datetime.datetime.strptime(sub_expiry, "%Y-%m-%d").date()
            if datetime.date.today() > expiry_date:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Gallery Suspended. The photographer's subscription has expired."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Expiry validation error: {e}")
            
    # 3. Event Status check
    if row["event_status"] == "paused":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gallery Paused. This wedding gallery has been temporarily paused."
        )
    elif row["event_status"] == "unpublished":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gallery Private. This wedding gallery has been unpublished."
        )


# --- AUTHENTICATION & TEAM MANAGEMENT (RBAC) ---
@router.post("/login")
async def login(username: str = Query(...), password: str = Query(...)):
    hashed = hash_password(password)
    with get_db() as conn:
        row = conn.execute(
            "SELECT role, name, status FROM users WHERE username = ? AND password_hash = ?",
            (username, hashed)
        ).fetchone()
        
    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password credentials."
        )
        
    if row["status"] == "suspended":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your studio account has been suspended due to pending billing. Please contact the administrator."
        )
        
    return {
        "status": "success",
        "username": username,
        "role": row["role"],
        "name": row["name"]
    }

@router.get("/users")
async def list_users():
    with get_db() as conn:
        rows = conn.execute("SELECT id, username, role, name, status, plan_name, created_at FROM users").fetchall()
        return [dict(r) for r in rows]

@router.post("/users")
async def create_user(
    username: str = Query(...),
    password: str = Query(...),
    role: str = Query(...),
    name: str = Query(...)
):
    if role not in {"owner", "junior", "superadmin"}:
        raise HTTPException(status_code=400, detail="Invalid role type. Must be 'owner', 'junior', or 'superadmin'.")
        
    hashed = hash_password(password)
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, name) VALUES (?, ?, ?, ?)",
                (username, hashed, role, name)
            )
        return {"status": "success", "message": f"User {username} created successfully."}
    except psycopg2.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists.")

@router.delete("/users/{user_id}")
async def delete_user(user_id: int):
    with get_db() as conn:
        owner_rows = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE role = 'owner'").fetchone()
        target_row = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
        
        if not target_row:
            raise HTTPException(status_code=404, detail="User not found.")
            
        if target_row["role"] == "owner" and owner_rows["cnt"] <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last admin owner account.")
            
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
    return {"status": "success", "message": "User deleted successfully."}


# --- SUPERADMIN INFRASTRUCTURE & BILLING TIER CONTROL ---
@router.get("/superadmin/admins")
async def superadmin_get_admins():
    with get_db() as conn:
        admins = conn.execute(
            """
            SELECT id, username, name, status, plan_name, allocated_storage_bytes, 
                   custom_storage_bytes, subscription_expires_at, processing_priority, created_at 
            FROM users WHERE role = 'owner'
            """
        ).fetchall()
        res = []
        for admin in admins:
            user = admin["username"]
            e_count = conn.execute("SELECT COUNT(*) as cnt FROM events WHERE owner_username = ?", (user,)).fetchone()["cnt"]
            used_bytes = get_admin_storage_used(user)
            limit = admin["custom_storage_bytes"] if admin["custom_storage_bytes"] is not None else admin["allocated_storage_bytes"]
            
            # Check subscription expiry status
            sub_expiry = admin["subscription_expires_at"]
            subscription_expired = False
            if sub_expiry:
                try:
                    expiry_date = datetime.datetime.strptime(sub_expiry, "%Y-%m-%d").date()
                    if datetime.date.today() > expiry_date:
                        subscription_expired = True
                except:
                    pass

            res.append({
                "id": admin["id"],
                "username": user,
                "name": admin["name"],
                "status": admin["status"],
                "plan_name": admin["plan_name"],
                "allocated_gb": round(limit / (1024**3), 1),
                "custom_storage_bytes": admin["custom_storage_bytes"],
                "subscription_expires_at": sub_expiry,
                "subscription_expired": subscription_expired,
                "processing_priority": admin["processing_priority"] or "normal",
                "consumed_mb": round(used_bytes / (1024*1024), 2),
                "events_count": e_count,
                "created_at": admin["created_at"]
            })
        return res

@router.post("/superadmin/admins/{username}/status")
async def superadmin_toggle_status(username: str, status_str: str = Query(..., alias="status")):
    if status_str not in {"active", "suspended"}:
        raise HTTPException(status_code=400, detail="Status must be 'active' or 'suspended'.")
    with get_db() as conn:
        conn.execute("UPDATE users SET status = ? WHERE username = ?", (status_str, username))
    return {"status": "success", "message": f"User {username} billing status changed to {status_str}."}

@router.post("/superadmin/admins/{username}/plan")
async def superadmin_change_plan(username: str, plan_name: str = Query(...)):
    with get_db() as conn:
        plan = conn.execute("SELECT storage_limit_gb FROM plans WHERE name = ?", (plan_name,)).fetchone()
        if not plan:
            raise HTTPException(status_code=404, detail="Selected plan package tier does not exist.")
        limit_bytes = plan["storage_limit_gb"] * 1024 * 1024 * 1024
        conn.execute(
            "UPDATE users SET plan_name = ?, allocated_storage_bytes = ? WHERE username = ?",
            (plan_name, limit_bytes, username)
        )
    return {"status": "success", "message": f"Photographer plan upgraded to {plan_name}."}

@router.get("/superadmin/plans")
async def superadmin_get_plans():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM plans").fetchall()
        return [dict(r) for r in rows]

@router.post("/superadmin/plans")
async def superadmin_create_plan(
    name: str = Query(...),
    storage_limit_gb: int = Query(...),
    event_limit: int = Query(...),
    price_inr: int = Query(...)
):
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO plans (name, storage_limit_gb, event_limit, price_inr)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET 
                    storage_limit_gb = excluded.storage_limit_gb,
                    event_limit = excluded.event_limit,
                    price_inr = excluded.price_inr
                """,
                (name, storage_limit_gb, event_limit, price_inr)
            )
        return {"status": "success", "message": f"Pricing package {name} configured."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/superadmin/logs")
async def superadmin_get_logs():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 200").fetchall()
        return [dict(r) for r in rows]

@router.get("/superadmin/stats")
async def superadmin_get_stats():
    with get_db() as conn:
        ph_row = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE role = 'owner'").fetchone()
        ev_row = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
        alloc_row = conn.execute("SELECT SUM(allocated_storage_bytes) as total FROM users WHERE role = 'owner'").fetchone()
        
    total_bytes = get_total_storage_used()
    allocated_bytes = alloc_row["total"] if alloc_row and alloc_row["total"] else 0
    
    return {
        "total_photographers": ph_row["cnt"] if ph_row else 0,
        "total_events": ev_row["cnt"] if ev_row else 0,
        "total_allocated_gb": round(allocated_bytes / (1024**3), 1),
        "total_consumed_gb": round(total_bytes / (1024**3), 3),
        "total_consumed_mb": round(total_bytes / (1024*1024), 2)
    }


@router.get("/superadmin/admins/{username}/xray")
async def superadmin_get_admin_xray(username: str):
    with get_db() as conn:
        # Fetch admin general info
        admin = conn.execute(
            """
            SELECT id, username, name, status, plan_name, allocated_storage_bytes, 
                   custom_storage_bytes, subscription_expires_at, processing_priority, created_at 
            FROM users WHERE username = ? AND role = 'owner'
            """,
            (username,)
        ).fetchone()
        
        if not admin:
            raise HTTPException(status_code=404, detail="Photographer not found.")
            
        # Fetch all events with status and storage details
        events_raw = conn.execute(
            "SELECT id, name, client_name, date, venue, auto_expiry, pin, status, created_at FROM events WHERE owner_username = ?",
            (username,)
        ).fetchall()
        
        events = []
        for ev in events_raw:
            # Count event photos
            p_count = conn.execute("SELECT COUNT(*) as cnt FROM event_photos WHERE event_id = ?", (ev["id"],)).fetchone()["cnt"]
            # Storage used by event
            used_bytes = conn.execute("SELECT SUM(file_size) as total FROM event_photos WHERE event_id = ?", (ev["id"],)).fetchone()["total"] or 0
            # Guest traffic analytics (scans)
            scans_count = conn.execute("SELECT COUNT(*) as cnt FROM analytics WHERE event_id = ? AND metric_type = 'qr_scan'", (ev["id"],)).fetchone()["cnt"]
            # Lead capture count
            leads_count = conn.execute("SELECT COUNT(*) as cnt FROM leads WHERE event_id = ?", (ev["id"],)).fetchone()["cnt"]
            
            events.append({
                "id": ev["id"],
                "name": ev["name"],
                "client_name": ev["client_name"],
                "date": ev["date"],
                "venue": ev["venue"],
                "auto_expiry": ev["auto_expiry"],
                "pin": ev["pin"],
                "status": ev["status"] or "active",
                "photos_count": p_count,
                "storage_mb": round(used_bytes / (1024*1024), 2),
                "scans_count": scans_count,
                "leads_count": leads_count,
                "created_at": ev["created_at"]
            })
        
        # Get total API calls (face matches) across photographer's events
        total_api_calls = 0
        for ev in events:
            action_pattern = f"face_match (event: {ev['id']})%"
            calls = conn.execute(
                "SELECT COUNT(*) as cnt FROM audit_logs WHERE action LIKE ?",
                (action_pattern,)
            ).fetchone()["cnt"]
            total_api_calls += calls
            
        # Unique guests count across all events
        total_leads = conn.execute(
            """
            SELECT COUNT(DISTINCT phone) as cnt 
            FROM leads l
            JOIN events e ON l.event_id = e.id
            WHERE e.owner_username = ?
            """,
            (username,)
        ).fetchone()["cnt"]
        
        # Total unique guest scans
        total_scans = conn.execute(
            """
            SELECT COUNT(*) as cnt 
            FROM analytics a
            JOIN events e ON a.event_id = e.id
            WHERE e.owner_username = ? AND a.metric_type = 'qr_scan'
            """,
            (username,)
        ).fetchone()["cnt"]

    used_bytes = get_admin_storage_used(username)
    limit = admin["custom_storage_bytes"] if admin["custom_storage_bytes"] is not None else admin["allocated_storage_bytes"]
    
    return {
        "username": admin["username"],
        "name": admin["name"],
        "status": admin["status"],
        "plan_name": admin["plan_name"],
        "allocated_gb": round(limit / (1024**3), 1),
        "custom_storage_bytes": admin["custom_storage_bytes"],
        "subscription_expires_at": admin["subscription_expires_at"],
        "processing_priority": admin["processing_priority"] or "normal",
        "consumed_mb": round(used_bytes / (1024*1024), 2),
        "events": events,
        "total_api_calls": total_api_calls,
        "unique_leads_count": total_leads,
        "total_scans_count": total_scans,
        "created_at": admin["created_at"]
    }

@router.post("/superadmin/events/{event_id}/status")
async def superadmin_set_event_status(event_id: str, status_str: str = Query(..., alias="status")):
    if status_str not in {"active", "paused", "unpublished"}:
        raise HTTPException(status_code=400, detail="Status must be 'active', 'paused', or 'unpublished'.")
    with get_db() as conn:
        conn.execute("UPDATE events SET status = ? WHERE id = ?", (status_str, event_id))
    return {"status": "success", "message": f"Event status changed to {status_str}."}

@router.delete("/superadmin/events/{event_id}")
async def superadmin_delete_event(event_id: str):
    with get_db() as conn:
        # Clean up event files
        
            
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    return {"status": "success", "message": "Event deleted successfully by superadmin."}

@router.post("/superadmin/admins/{username}/financials")
async def superadmin_update_financials(
    username: str,
    subscription_expires_at: Optional[str] = Query(None, description="Format YYYY-MM-DD"),
    custom_storage_gb: Optional[float] = Query(None, description="Custom storage limit override in GB")
):
    if subscription_expires_at:
        try:
            datetime.datetime.strptime(subscription_expires_at, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Expiry date must be in YYYY-MM-DD format.")
            
    custom_bytes = int(custom_storage_gb * 1024 * 1024 * 1024) if custom_storage_gb is not None else None
    
    with get_db() as conn:
        conn.execute(
            """
            UPDATE users 
            SET subscription_expires_at = ?, custom_storage_bytes = ? 
            WHERE username = ? AND role = 'owner'
            """,
            (subscription_expires_at, custom_bytes, username)
        )
    return {"status": "success", "message": "Photographer subscription ledger & override updated successfully."}

@router.post("/superadmin/admins/{username}/priority")
async def superadmin_update_priority(username: str, priority: str = Query(..., description="'low', 'normal', or 'high'")):
    if priority not in {"low", "normal", "high"}:
        raise HTTPException(status_code=400, detail="Priority must be 'low', 'normal', or 'high'.")
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET processing_priority = ? WHERE username = ? AND role = 'owner'",
            (priority, username)
        )
    return {"status": "success", "message": f"Processing priority for @{username} updated to {priority}."}

@router.post("/superadmin/broadcast")
async def superadmin_create_broadcast(message: str = Query(...)):
    with get_db() as conn:
        conn.execute("UPDATE broadcasts SET active = 0")
        conn.execute("INSERT INTO broadcasts (message, active) VALUES (?, 1)", (message,))
    return {"status": "success", "message": "System alert broadcast published successfully."}

@router.post("/superadmin/broadcast/deactivate")
async def superadmin_deactivate_broadcast():
    with get_db() as conn:
        conn.execute("UPDATE broadcasts SET active = 0")
    return {"status": "success", "message": "System broadcast deactivated."}

@router.get("/broadcast/active")
async def get_active_broadcast():
    with get_db() as conn:
        row = conn.execute("SELECT message FROM broadcasts WHERE active = 1 ORDER BY created_at DESC LIMIT 1").fetchone()
    if row:
        return {"status": "success", "active": True, "message": row["message"]}
    return {"status": "success", "active": False, "message": None}


# --- LEAD GENERATION CRM ROUTERS ---
@router.post("/events/{event_id}/leads")
async def add_lead(event_id: str, name: str = Query(...), phone: str = Query(...)):
    check_event_active(event_id)
    with get_db() as conn:
        event = conn.execute("SELECT id FROM events WHERE id = ?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found.")
            
        existing = conn.execute(
            "SELECT id FROM leads WHERE event_id = ? AND phone = ?",
            (event_id, phone)
        ).fetchone()
        
        if not existing:
            conn.execute(
                "INSERT INTO leads (event_id, name, phone) VALUES (?, ?, ?)",
                (event_id, name, phone)
            )
            
    return {"status": "success", "message": "Lead registration recorded."}

@router.get("/events/{event_id}/leads")
async def get_event_leads(event_id: str):
    check_event_active(event_id)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT name, phone, created_at FROM leads WHERE event_id = ? ORDER BY created_at DESC",
            (event_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- CLIENT PROOFING & SELECTIONS ---
@router.post("/photos/{photo_id}/select")
async def toggle_photo_selection(photo_id: int, selected: int = Query(1)):
    check_photo_active(photo_id)
    with get_db() as conn:
        photo = conn.execute("SELECT id FROM event_photos WHERE id = ?", (photo_id,)).fetchone()
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found.")
            
        conn.execute(
            """
            INSERT INTO photo_selections (photo_id, selected) 
            VALUES (?, ?)
            ON CONFLICT(photo_id) DO UPDATE SET selected = excluded.selected
            """,
            (photo_id, selected)
        )
        
    return {"status": "success", "message": "Photo selection state updated."}


# --- SAAS EVENT MANAGEMENT ROUTERS ---
@router.post("/events", status_code=status.HTTP_201_CREATED)
async def create_event(
    name: str = Query(...),
    client_name: str = Query(...),
    date: str = Query(...),
    venue: str = Query(...),
    pin: Optional[str] = Query(None),
    auto_expiry: int = Query(0),
    owner_username: str = Query("admin")
):
    # Verify owner photographer account is active
    with get_db() as conn:
        admin = conn.execute("SELECT status, plan_name FROM users WHERE username = ?", (owner_username,)).fetchone()
        if not admin:
            raise HTTPException(status_code=404, detail="Photographer owner account not found.")
        if admin["status"] == "suspended":
            raise HTTPException(status_code=403, detail="Photographer account suspended. Cannot create new events.")
            
        # Verify event limit on Basic plans
        e_count = conn.execute("SELECT COUNT(*) as cnt FROM events WHERE owner_username = ?", (owner_username,)).fetchone()["cnt"]
        if admin["plan_name"] == "Basic" and e_count >= 10:
            raise HTTPException(status_code=403, detail="Event limit hit (Max 10 Events on Basic Plan). Please upgrade.")

    event_id = name.strip() # Set event ID directly to event name
    
    # Check if event name already exists to prevent duplicate IDs
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM events WHERE id = ?", (event_id,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="An event with this name already exists. Please choose a unique name.")
            
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO events (id, name, client_name, date, venue, pin, auto_expiry, owner_username)
            VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (event_id, name, client_name, date, venue, auto_expiry, owner_username)
        )
    return {"status": "success", "message": "Event created successfully.", "event_id": event_id}

@router.get("/events")
async def list_events(owner_username: Optional[str] = Query(None)):
    query = "SELECT * FROM events"
    params = []
    if owner_username:
        query += " WHERE owner_username = ?"
        params.append(owner_username)
    query += " ORDER BY created_at DESC"
    
    with get_db() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]

@router.get("/events/{event_id}")
async def get_event_details(event_id: str):
    check_event_active(event_id)
    with get_db() as conn:
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found.")
    return dict(row)

@router.delete("/events/{event_id}")
async def delete_event(event_id: str):
        
    with get_db() as conn:
            # Cloudinary cleanup not implemented here, but no local file to delete
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    return {"status": "success", "message": f"Event {event_id} deleted."}

@router.post("/events/{event_id}/logo")
async def upload_logo(event_id: str, file: UploadFile = File(...)):
    check_event_active(event_id)
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported logo image format.")
        
    content = await file.read()
    public_url = upload_to_telegram(content, f"{event_id}_logo.jpg", event_id)
        
    with get_db() as conn:
        conn.execute("UPDATE events SET logo_filename = ? WHERE id = ?", (public_url, event_id))
    return {"status": "success", "logo_url": public_url}

@router.post("/events/{event_id}/toggle-settings")
async def update_settings(
    event_id: str,
    watermark_enabled: Optional[int] = Query(None),
    paywall_enabled: Optional[int] = Query(None),
    pin: Optional[str] = Query(None),
    auto_expiry: Optional[int] = Query(None)
):
    check_event_active(event_id)
    with get_db() as conn:
        updates = []
        params = []
        if watermark_enabled is not None:
            updates.append("watermark_enabled = ?")
            params.append(watermark_enabled)
        if paywall_enabled is not None:
            updates.append("paywall_enabled = ?")
            params.append(paywall_enabled)
        if pin is not None:
            updates.append("pin = ?")
            params.append(pin if pin.strip() != "" else None)
        if auto_expiry is not None:
            updates.append("auto_expiry = ?")
            params.append(auto_expiry)
            
        if not updates:
            raise HTTPException(status_code=400, detail="No settings fields provided for update.")
            
        params.append(event_id)
        conn.execute(f"UPDATE events SET {', '.join(updates)} WHERE id = ?", tuple(params))
    return {"status": "success", "message": "Settings updated successfully."}

@router.post("/events/{event_id}/verify-pin")
async def verify_pin(event_id: str, pin: str = Query(...)):
    check_event_active(event_id)
    with get_db() as conn:
        row = conn.execute("SELECT pin FROM events WHERE id = ?", (event_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found.")
        
    if row["pin"] == pin:
        return {"status": "success", "valid": True}
    else:
        return {"status": "success", "valid": False}

# --- ANALYTICS AND TELEMETRY ---
@router.post("/events/{event_id}/log-scan")
async def log_scan(event_id: str, request: Request):
    check_event_active(event_id)
    visitor_ip = request.client.host if request.client else "Unknown"
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id FROM analytics 
            WHERE event_id = ? AND visitor_ip = ? AND metric_type = 'qr_scan'
            AND timestamp >= datetime('now', '-1 hour')
            LIMIT 1
            """,
            (event_id, visitor_ip)
        ).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO analytics (event_id, metric_type, visitor_ip) VALUES (?, 'qr_scan', ?)",
                (event_id, visitor_ip)
            )
    return {"status": "success"}

@router.get("/events/{event_id}/analytics")
async def get_event_analytics(event_id: str):
    check_event_active(event_id)
    with get_db() as conn:
        storage_row = conn.execute(
            "SELECT SUM(file_size) as total_size FROM event_photos WHERE event_id = ?",
            (event_id,)
        ).fetchone()
        total_bytes = storage_row["total_size"] or 0
        total_mb = round(total_bytes / (1024 * 1024), 2)
        
        scan_row = conn.execute(
            "SELECT COUNT(*) as scans FROM analytics WHERE event_id = ? AND metric_type = 'qr_scan'",
            (event_id,)
        ).fetchone()
        
        success_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM analytics WHERE event_id = ? AND metric_type = 'search_success'",
            (event_id,)
        ).fetchone()
        total_matches_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM analytics WHERE event_id = ? AND metric_type IN ('search_success', 'search_fail')",
            (event_id,)
        ).fetchone()
        
        success_cnt = success_row["cnt"] or 0
        total_matches_cnt = total_matches_row["cnt"] or 0
        success_rate = round((success_cnt / total_matches_cnt * 100), 1) if total_matches_cnt > 0 else 100.0

    return {
        "status": "success",
        "storage_mb": total_mb,
        "total_scans": scan_row["scans"] or 0,
        "match_success_rate": success_rate
    }

# --- QR CODE GENERATION ---
@router.get("/events/{event_id}/qr")
async def get_event_qr(event_id: str, client_url: str = Query(None)):
    check_event_active(event_id)
    import qrcode
    
    with get_db() as conn:
        row = conn.execute("SELECT name, pin FROM events WHERE id = ?", (event_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found.")
        
    pin = row["pin"]
    target_url = client_url or f"http://localhost:50172/#/?event_id={event_id}"
    if pin:
        target_url += f"&pin={pin}"
    
    def generate_qr():
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(target_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#111827", back_color="white")
        
        buf = BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        return buf
        
    try:
        qr_io = await asyncio.to_thread(generate_qr)
        return StreamingResponse(qr_io, media_type="image/png")
    except Exception as e:
        logger.error(f"Failed to generate QR: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate QR code.")

# --- INGESTION & PHOTO MANAGEMENT ---
@router.post("/upload-event-photos", status_code=status.HTTP_201_CREATED)
async def upload_event_photos(
    event_id: str = Query(...),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    check_event_active(event_id)
    
    # Get photographer owner limit & settings
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT u.username, u.allocated_storage_bytes, u.custom_storage_bytes,
                   u.processing_priority, u.subscription_expires_at
            FROM events e
            JOIN users u ON e.owner_username = u.username
            WHERE e.id = ?
            """,
            (event_id,)
        ).fetchone()
        
    owner_username = row["username"] if row else "admin"
    priority = row["processing_priority"] if row else "normal"
    max_storage = row["custom_storage_bytes"] if row and row["custom_storage_bytes"] is not None else (row["allocated_storage_bytes"] if row else 53687091200)
    
    # Enforce Subscription Expiry Lockout
    sub_expiry = row["subscription_expires_at"] if row else None
    if sub_expiry:
        try:
            expiry_date = datetime.datetime.strptime(sub_expiry, "%Y-%m-%d").date()
            if datetime.date.today() > expiry_date:
                log_audit_action(owner_username, "upload_photos", 0, 0, "failed (subscription expired)")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Subscription Expired. Your studio dashboard is locked. Please pay your outstanding invoices to resume uploading."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Expiry validation error in upload: {e}")
            
    # Enforce Storage Quotas
    current_storage = get_admin_storage_used(owner_username)
    if current_storage >= max_storage:
        log_audit_action(owner_username, "upload_photos", 0, 0, "failed (quota full)")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Storage quota full. Please contact administrator to upgrade your plan."
        )

    
    
    start_time = datetime.datetime.now()
    file_contents = []
    photo_ids_map = []
    
    with get_db() as conn:
        for file in files:
            content_type = file.content_type or ""
            ext = os.path.splitext(file.filename)[1].lower()
            if content_type.startswith("image/") or ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".tiff", ".gif"}:
                content = await file.read()
                
                cursor = conn.execute(
                    """
                    INSERT INTO event_photos (event_id, file_id, file_size, faces_scanned)
                    VALUES (?, ?, ?, 0)
                    RETURNING id
                    """,
                    (event_id, "pending_upload", 0)
                )
                photo_id = cursor.fetchone()["id"]
                
                file_contents.append((file.filename, content, content_type, photo_id))
                photo_ids_map.append({"filename": file.filename, "photo_id": photo_id})
            await file.close()

    async def background_upload(event_id_str: str, files_data: list, owner_usr: str, prio: str, max_stor: int):
        saved_files = []
        total_uploaded_bytes = 0
        
        async def process_single(filename, content, ctype, photo_id):
            nonlocal total_uploaded_bytes
            if prio == "low":
                await asyncio.sleep(0.5)
            try:
                if get_admin_storage_used(owner_usr) >= max_stor:
                    return None
                
                public_url, file_size = await asyncio.to_thread(process_and_save_uploaded_image, content, filename, event_id_str)
                total_uploaded_bytes += file_size
                
                with get_db() as conn:
                    conn.execute(
                        """
                        UPDATE event_photos 
                        SET file_id = ?, file_size = ?
                        WHERE id = ?
                        """,
                        (public_url, file_size, photo_id)
                    )
                    return {"id": photo_id, "file_id": public_url}
            except Exception as e:
                logger.error(f"Failed saving uploaded file {filename}: {e}")
                return None

        tasks = [process_single(fn, c, ct, pid) for fn, c, ct, pid in files_data]
        results = await asyncio.gather(*tasks)
        for res in results:
            if res:
                saved_files.append(res)
                
        if saved_files:
            duration = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
            log_audit_action(owner_usr, f"upload_photos (event: {event_id_str})", duration, total_uploaded_bytes, "success")
            await ingestion_worker(event_id_str)

    background_tasks.add_task(background_upload, event_id, file_contents, owner_username, priority, max_storage)
        
    return {
        "status": "success",
        "message": f"Successfully queued {len(file_contents)} images for background uploading and scanning. You may safely close this page.",
        "uploaded_count": len(file_contents),
        "photo_ids": photo_ids_map
    }

@router.post("/events/{event_id}/upgrade-photo")
async def upgrade_photo_res(
    event_id: str,
    photo_id: int = Form(...),
    file: UploadFile = File(...)
):
    check_event_active(event_id)
    content = await file.read()
    
    try:
        public_url, file_size = await asyncio.to_thread(
            process_and_save_uploaded_image, content, file.filename, event_id
        )
    except Exception as e:
        logger.error(f"Failed to upgrade photo {photo_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload high-res image.")
        
    with get_db() as conn:
        conn.execute(
            """
            UPDATE event_photos 
            SET file_id = ?, file_size = ?
            WHERE id = ? AND event_id = ?
            """,
            (public_url, file_size, photo_id, event_id)
        )
        
    return {"status": "success", "photo_id": photo_id, "file_id": public_url}

@router.get("/events/{event_id}/photos")
async def list_event_photos(event_id: str, selected_only: bool = Query(False)):
    check_event_active(event_id)
    query = """
        SELECT ep.id, ep.file_id, ep.file_size, ep.faces_scanned, ep.faces_count, 
               CASE WHEN ps.selected = 1 THEN 1 ELSE 0 END as selected
        FROM event_photos ep
        LEFT JOIN photo_selections ps ON ep.id = ps.photo_id
        WHERE ep.event_id = ?
    """
    params = [event_id]
    if selected_only:
        query += " AND ps.selected = 1"
    query += " ORDER BY ep.created_at DESC"
    
    with get_db() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]

@router.delete("/photos/{photo_id}")
async def delete_photo(photo_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT event_id, file_id FROM event_photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found.")
        
    event_id = row["event_id"]
    file_id = row["file_id"]
    
    # No local file to delete, Cloudinary deletion logic can be added later if needed
            
    with get_db() as conn:
        conn.execute("DELETE FROM event_photos WHERE id = ?", (photo_id,))
    return {"status": "success", "message": "Photo deleted successfully."}

@router.get("/events/{event_id}/ingestion-status")
async def get_ingestion_status(event_id: str):
    check_event_active(event_id)
    with get_db() as conn:
        total_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM event_photos WHERE event_id = ?",
            (event_id,)
        ).fetchone()
        scanned_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM event_photos WHERE event_id = ? AND faces_scanned != 0",
            (event_id,)
        ).fetchone()
        
    total = total_row["cnt"] if total_row else 0
    scanned = scanned_row["cnt"] if scanned_row else 0
    
    return {"status": "success", "total": total, "scanned": scanned}

# --- DYNAMIC RENDERING & PROTECTION GATEWAYS ---
@router.get("/preview-photo/{photo_id}")
async def get_preview_photo(photo_id: int):
    check_photo_active(photo_id)
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT ep.file_id, ep.event_id, e.watermark_enabled, e.name as event_name,
                   e.logo_filename, u.name as brand_name
            FROM event_photos ep
            JOIN events e ON ep.event_id = e.id
            JOIN users u ON e.owner_username = u.username
            WHERE ep.id = ?
            """,
            (photo_id,)
        ).fetchone()
        
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found.")
        
    file_id = row["file_id"]
    event_id = row["event_id"]
    watermark_enabled = row["watermark_enabled"]
    event_name = row["event_name"]
    logo_filename = row["logo_filename"]
    brand_name = row["brand_name"]
    
    image_path = file_id
    if not file_id:
        raise HTTPException(status_code=404, detail="File does not exist.")
        
    try:
        download_url = get_telegram_file_url(file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not fetch image from Telegram.")

    image_path = download_url
    if not watermark_enabled:
        return RedirectResponse(url=download_url, status_code=301, headers={"Cache-Control": "public, max-age=31536000"})
        
        def get_watermarked_bytes():
            from PIL import Image, ImageDraw, ImageFont
            import requests
            from io import BytesIO
            import os
            from app.core.config import settings
            
            response = requests.get(download_url)
            image = Image.open(BytesIO(response.content))
            if image.mode != "RGBA":
                image = image.convert("RGBA")
                
            txt = Image.new("RGBA", image.size, (255, 255, 255, 0))
            d = ImageDraw.Draw(txt)
            
            w, h = image.size
            watermark_str = f"{brand_name.upper()} • {event_name.upper()}"
            
            try:
                font = ImageFont.truetype("arial.ttf", int(w / 35))
            except IOError:
                font = ImageFont.load_default()
                
            # Draw diagonal repeating text
            for x in range(30, w, int(w/2.5)):
                for y in range(30, h, int(h/4)):
                    d.text((x, y), watermark_str, fill=(255, 255, 255, 40), font=font)
                    
            # Center logo overlay
            if logo_filename:
                logo_path = os.path.join(settings.LOGOS_DIR, logo_filename)
                if os.path.exists(logo_path):
                    try:
                        logo = Image.open(logo_path)
                        if logo.mode != "RGBA":
                            logo = logo.convert("RGBA")
                        
                        logo_w = int(w * 0.25)
                        logo_h = int(logo_w * logo.height / logo.width)
                        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
                        
                        alpha = logo.split()[3]
                        alpha = alpha.point(lambda p: p * 0.5) # 50% opacity
                        logo.putalpha(alpha)
                        
                        logo_x = int((w - logo_w) / 2)
                        logo_y = int((h - logo_h) / 2)
                        txt.paste(logo, (logo_x, logo_y), logo)
                    except Exception as e:
                        pass
                
            watermarked = Image.alpha_composite(image, txt)
            rgb_im = watermarked.convert("RGB")
        
        out_buf = BytesIO()
        rgb_im.save(out_buf, "JPEG", quality=80)
        out_buf.seek(0)
        
        image.close()
        txt.close()
        watermarked.close()
        rgb_im.close()
        return out_buf

    try:
        buffer = await asyncio.to_thread(get_watermarked_bytes)
        return StreamingResponse(buffer, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Watermark rendering failed: {e}")
        return RedirectResponse(url=download_url)
    #image_path, media_type="image/jpeg")

@router.get("/download-photo/{photo_id}")
async def download_photo(photo_id: int):
    check_photo_active(photo_id)
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT ep.file_id, ep.event_id, e.paywall_enabled 
            FROM event_photos ep
            JOIN events e ON ep.event_id = e.id
            WHERE ep.id = ?
            """,
            (photo_id,)
        ).fetchone()
        
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found.")
        
    if row["paywall_enabled"]:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="This high-resolution original image is locked behind a paywall."
        )
        
    file_id = row["file_id"]
    if not file_id:
        raise HTTPException(status_code=404, detail="File not found.")
        
    try:
        download_url = get_telegram_file_url(file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not fetch image from Telegram.")
        
    return RedirectResponse(url=download_url, status_code=301, headers={"Cache-Control": "public, max-age=31536000"})

# --- DUAL-SIDE FACE SEARCH MECHANICS ---
@router.post("/find-matches", response_class=JSONResponse)
async def find_matches(
    event_id: str = Query(...),
    file: UploadFile = File(...),
    tolerance: float = Query(0.65)
):
    check_event_active(event_id)
    start_time = datetime.datetime.now()
    content = await file.read()
    
    temp_selfie_filename = f"reference_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    
    try:
        # Upload selfie directly to Telegram
        public_url = upload_to_telegram(content, f"{temp_selfie_filename}.jpg", "temp_selfies")
            
        ref_encoding = await asyncio.to_thread(extract_reference_encoding, public_url, model="cnn")
        
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT ep.id, ep.file_id, fe.encoding 
                FROM face_encodings fe
                JOIN event_photos ep ON fe.photo_id = ep.id
                WHERE ep.event_id = ? AND ep.faces_scanned = 1
                """,
                (event_id,)
            ).fetchall()
            
        matched_photos = []
        import face_recognition
        
        photo_encodings = {}
        for row in rows:
            p_id = row["id"]
            fname = row["file_id"]
            enc = deserialize_encoding(row["encoding"])
            if p_id not in photo_encodings:
                photo_encodings[p_id] = {"file_id": fname, "encodings": []}
            photo_encodings[p_id]["encodings"].append(enc)
            
        def run_vector_search():
            for p_id, val in photo_encodings.items():
                encs = val["encodings"]
                matches = face_recognition.compare_faces(encs, ref_encoding, tolerance=tolerance)
                if any(matches):
                    matched_photos.append({
                        "id": p_id,
                        "filename": val["file_id"]
                    })
                    
        await asyncio.to_thread(run_vector_search)
        
        with get_db() as conn:
            metric = "search_success" if matched_photos else "search_fail"
            conn.execute(
                "INSERT INTO analytics (event_id, metric_type) VALUES (?, ?)",
                (event_id, metric)
            )
            
        duration = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        log_audit_action("guest", f"face_match (event: {event_id})", duration, len(content), "success")
        
        return {
            "status": "success",
            "matches": matched_photos,
            "match_count": len(matched_photos)
        }
        
    except FaceRecognitionError as fre:
        with get_db() as conn:
            conn.execute("INSERT INTO analytics (event_id, metric_type) VALUES (?, 'search_fail')", (event_id,))
        duration = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        log_audit_action("guest", f"face_match (event: {event_id})", duration, len(content), "failed (no face)")
        raise HTTPException(status_code=400, detail=str(fre))
    except Exception as e:
        logger.error(f"Matching error: {e}", exc_info=True)
        duration = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
        log_audit_action("guest", f"face_match (event: {event_id})", duration, len(content), "failed (error)")
        raise HTTPException(status_code=500, detail="Error occurred while processing face match.")
    finally:
        await file.close()

# --- STREAMING ZIP ARCHIVE ---
@router.get("/download-zip")
async def download_zip(
    event_id: str = Query(...),
    photo_ids: List[int] = Query(default=None, description="List of photo database IDs to bundle into ZIP. If not provided, bundles all event photos.")
):
    check_event_active(event_id)
    with get_db() as conn:
        event = conn.execute("SELECT name, paywall_enabled FROM events WHERE id = ?", (event_id,)).fetchone()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found.")
        
    if event["paywall_enabled"]:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Bulk downloads are locked behind a paywall for this event."
        )
        
    with get_db() as conn:
        if photo_ids and len(photo_ids) > 0:
            placeholders = ",".join("?" for _ in photo_ids)
            rows = conn.execute(
                f"SELECT file_id FROM event_photos WHERE id IN ({placeholders}) AND event_id = ?",
                (*photo_ids, event_id)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT file_id FROM event_photos WHERE event_id = ?",
                (event_id,)
            ).fetchall()
        
    file_ids = [r["file_id"] for r in rows]
    if not filenames:
        raise HTTPException(status_code=400, detail="No matching files found for zip download.")
        
    def build_zip() -> BytesIO:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, f_id in enumerate(file_ids):
                try:
                    url = get_telegram_file_url(f_id)
                    resp = requests.get(url)
                    if resp.status_code == 200:
                        zip_file.writestr(f"photo_{idx}.jpg", resp.content)
                except Exception:
                    pass
                
        zip_buffer.seek(0)
        return zip_buffer
        
    try:
        zip_io = await asyncio.to_thread(build_zip)
        zip_name = f"{event['name'].replace(' ', '_')}_matched_photos.zip"
        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_name}",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        logger.error(f"ZIP Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to bundle files into ZIP.")
