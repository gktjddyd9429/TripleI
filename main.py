from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import os
import uuid
from urllib import request


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    model: str | None = Field(default=None, description="사용할 OpenAI 모델명")


class ChatResponse(BaseModel):
    reply: str
    success: bool
    model: str
    provider: str = "openai"
    error: str | None = None


class ChatbotLogItem(BaseModel):
    id: int
    flag: int
    message_text: str


class ChatbotService:
    def __init__(self):
        self._load_env()
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        self.db_path = BASE_DIR / "localhub.db"

    def _load_env(self):
        base_dir = Path(__file__).resolve().parent
        env_paths = [base_dir / "chatbot.env", base_dir / ".env"]

        for env_path in env_paths:
            if not env_path.exists():
                continue

            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    def _get_db_context(self, message: str) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT name, address, category_name, description FROM location ORDER BY id LIMIT 12"
            ).fetchall()
            conn.close()
        except Exception:
            return ""

        if not rows:
            return ""

        context_items = []
        for row in rows:
            category = row["category_name"] or "기타"
            context_items.append(
                f"- {row['name']} | {category} | {row['address']} | {row['description'] or ''}".strip()
            )

        prompt_context = (
            "아래는 현재 DB에 저장된 구미·경북권 여행 관련 데이터입니다. "
            "사용자 질문과 관련된 항목을 먼저 참고해서 답하세요.\n"
            f"{chr(10).join(context_items)}"
        )
        return prompt_context

    def _save_chat_log(self, flag: int, message_text: str) -> None:
        conn = get_conn()
        conn.execute(
            "INSERT INTO chatbot_log (flag, message_text) VALUES (?, ?)",
            (flag, message_text),
        )
        conn.commit()
        conn.close()

    def ask(self, message: str, model: str | None = None) -> dict:
        selected_model = (model or self.model).strip() or self.model
        self._save_chat_log(1, message)

        if not self.api_key or self.api_key == "your_openai_api_key_here":
            self._save_chat_log(0, "OpenAI API 키가 아직 없어요. chatbot.env에 OPENAI_API_KEY를 설정하면 바로 답변할 수 있어요.")
            return {
                "reply": "OpenAI API 키가 아직 없어요. chatbot.env에 OPENAI_API_KEY를 설정하면 바로 답변할 수 있어요.",
                "success": False,
                "model": selected_model,
                "provider": "openai",
                "error": "missing_api_key",
            }

        db_context = self._get_db_context(message)
        system_prompt = (
            "당신은 구미·경북권 여행 정보 챗봇입니다. "
            "사용자가 여행지, 맛집, 축제, 숙박, 쇼핑에 대해 묻으면 DB에 있는 정보와 연결해서 짧고 실용적으로 답하세요. "
            "가능하면 실제 장소명과 주소를 포함하세요."
        )
        if db_context:
            system_prompt += f"\n\n참고 데이터:\n{db_context}"

        payload = {
            "model": selected_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
        }

        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self._save_chat_log(0, f"챗봇 호출 중 오류가 발생했어요: {exc}")
            return {
                "reply": f"챗봇 호출 중 오류가 발생했어요: {exc}",
                "success": False,
                "model": selected_model,
                "provider": "openai",
                "error": "request_failed",
            }

        try:
            content = data["choices"][0]["message"]["content"]
            reply = content or "응답이 비어 있었어요."
            self._save_chat_log(0, reply)
            return {
                "reply": reply,
                "success": True,
                "model": selected_model,
                "provider": "openai",
                "error": None,
            }
        except Exception:
            self._save_chat_log(0, "챗봇 응답 형식이 올바르지 않았어요.")
            return {
                "reply": "챗봇 응답 형식이 올바르지 않았어요.",
                "success": False,
                "model": selected_model,
                "provider": "openai",
                "error": "invalid_response",
            }


app = FastAPI(
    title="LocalHub DB API",
    description="구미·경북권 여행 정보와 챗봇 연동을 위한 FastAPI 서버",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "json"
DB_PATH = BASE_DIR / "localhub.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_location_table_columns(conn):
    cursor = conn.execute("PRAGMA table_info(location)")
    return [row[1] for row in cursor.fetchall()]


def get_post_final_table_columns(conn):
    cursor = conn.execute("PRAGMA table_info(post_final)")
    return [row[1] for row in cursor.fetchall()]


def get_comment_final_table_columns(conn):
    cursor = conn.execute("PRAGMA table_info(comment_final)")
    return [row[1] for row in cursor.fetchall()]


def ensure_location_table():
    conn = get_conn()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='location'")
    exists = cursor.fetchone() is not None

    if exists:
        existing_columns = get_location_table_columns(conn)
        if "category_name" not in existing_columns:
            conn.execute("ALTER TABLE location ADD COLUMN category_name TEXT")
        if "image_url" not in existing_columns:
            conn.execute("ALTER TABLE location ADD COLUMN image_url TEXT")
        if "created_at" not in existing_columns:
            conn.execute("ALTER TABLE location ADD COLUMN created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")
        if "updated_at" not in existing_columns:
            conn.execute("ALTER TABLE location ADD COLUMN updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")
        conn.commit()
        conn.close()
        return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS location (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            description TEXT,
            category_name TEXT,
            image_url TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def ensure_post_final_table():
    conn = get_conn()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='post_final'")
    exists = cursor.fetchone() is not None

    if exists:
        existing_columns = get_post_final_table_columns(conn)
        if "nickname" not in existing_columns:
            conn.execute("ALTER TABLE post_final ADD COLUMN nickname TEXT")
        if "category_name" not in existing_columns:
            conn.execute("ALTER TABLE post_final ADD COLUMN category_name TEXT")
        if "image_url" not in existing_columns:
            conn.execute("ALTER TABLE post_final ADD COLUMN image_url TEXT")
        if "created_at" not in existing_columns:
            conn.execute("ALTER TABLE post_final ADD COLUMN created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")
        if "updated_at" not in existing_columns:
            conn.execute("ALTER TABLE post_final ADD COLUMN updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")
        conn.commit()
        conn.close()
        return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS post_final (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            password TEXT NOT NULL,
            nickname TEXT NOT NULL,
            category_name TEXT,
            image_url TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    old_post_exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='post'").fetchone() is not None
    if old_post_exists:
        try:
            rows = conn.execute(
                "SELECT id, title, content, password, category_name, image_url, created_at, updated_at FROM post ORDER BY id"
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT INTO post_final (id, title, content, password, nickname, category_name, image_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["id"],
                        row["title"],
                        row["content"],
                        row["password"],
                        "", 
                        row["category_name"],
                        row["image_url"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()

    conn.commit()
    conn.close()


def ensure_comment_final_table():
    conn = get_conn()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='comment_final'")
    exists = cursor.fetchone() is not None

    if exists:
        conn.close()
        return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comment_final (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            nickname TEXT NOT NULL,
            password TEXT NOT NULL,
            FOREIGN KEY(post_id) REFERENCES post_final(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    conn.close()


def ensure_chatbot_log_table():
    conn = get_conn()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chatbot_log'")
    exists = cursor.fetchone() is not None

    if exists:
        conn.close()
        return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chatbot_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flag INTEGER NOT NULL,
            message_text TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def row_to_dict(row):
    return {key: row[key] for key in row.keys()}


EXPECTED_JSON_ITEMS_COLUMNS = [
    "id",
    "source_file",
    "region",
    "content_type",
    "content_type_id",
    "content_id",
    "data",
]


def get_json_items_table_columns(conn):
    cursor = conn.execute("PRAGMA table_info(json_items)")
    return [row[1] for row in cursor.fetchall()]


def ensure_json_items_table():
    conn = get_conn()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='json_items'")
    exists = cursor.fetchone() is not None
    if exists:
        existing_columns = get_json_items_table_columns(conn)
        if existing_columns != EXPECTED_JSON_ITEMS_COLUMNS:
            conn.execute("DROP TABLE IF EXISTS json_items")
            conn.commit()
            conn.close()
            init_json_items_table()
            return
    conn.close()
    init_json_items_table()


def init_json_items_table():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS json_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            region TEXT,
            content_type TEXT,
            content_type_id INTEGER,
            content_id TEXT NOT NULL,
            data TEXT NOT NULL,
            UNIQUE(source_file, content_id)
        )
        """
    )
    conn.commit()
    conn.close()


def seed_json_items_table():
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM json_items").fetchone()[0]
    if count > 0:
        conn.close()
        return

    if not JSON_DIR.exists() or not JSON_DIR.is_dir():
        conn.close()
        return

    for json_path in sorted(JSON_DIR.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        source_file = json_path.name
        region = data.get("region") or ""
        content_type = data.get("contentType") or ""
        content_type_id = data.get("contentTypeId")

        for item in data.get("items", []):
            payload = dict(item)
            payload["sourceFile"] = source_file
            payload["region"] = region
            payload["contentType"] = content_type
            payload["contentTypeId"] = content_type_id
            content_id = str(payload.get("contentid") or uuid.uuid4().hex)
            payload["contentid"] = content_id

            conn.execute(
                "INSERT OR IGNORE INTO json_items (source_file, region, content_type, content_type_id, content_id, data) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source_file,
                    region,
                    content_type,
                    content_type_id,
                    content_id,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    conn.commit()
    conn.close()


def json_row_to_payload(row):
    data_text = None
    if "data" in row.keys():
        data_text = row["data"]
    elif "payload" in row.keys():
        data_text = row["payload"]
    else:
        raise RuntimeError("json_items row has neither data nor payload column")

    payload = json.loads(data_text)
    payload["sourceFile"] = row["source_file"]
    payload["region"] = row["region"]
    payload["contentType"] = row["content_type"]
    payload["contentTypeId"] = row["content_type_id"]
    payload["contentid"] = row["content_id"]
    return payload


def _select_json_fields(payload: dict):
    return {
        "addr1": payload.get("addr1"),
        "addr2": payload.get("addr2"),
        "firstimage": payload.get("firstimage"),
        "firstimage2": payload.get("firstimage2"),
        "mapx": payload.get("mapx"),
        "mapy": payload.get("mapy"),
        "tel": payload.get("tel"),
        "title": payload.get("title"),
        "modifiedtime": payload.get("modifiedtime"),
    }


ensure_json_items_table()
ensure_location_table()
ensure_post_final_table()
ensure_comment_final_table()
ensure_chatbot_log_table()
seed_json_items_table()


@app.post("/chatbot/chat", response_model=ChatResponse, summary="챗봇으로 여행 정보 질의", description="DB에 저장된 여행지/맛집/축제 정보를 참고해 OpenAI로 답변합니다.")
def chatbot_chat(request: ChatRequest):
    service = ChatbotService()
    result = service.ask(request.message, request.model)
    return ChatResponse(**result)


@app.get("/chatbot/health")
def chatbot_health():
    service = ChatbotService()
    configured = bool(service.api_key and service.api_key != "your_openai_api_key_here")
    return {
        "status": "ok",
        "configured": configured,
        "model": service.model,
        "provider": "openai",
    }


@app.get("/chatbot/config")
def chatbot_config():
    service = ChatbotService()
    return {
        "configured": bool(service.api_key and service.api_key != "your_openai_api_key_here"),
        "model": service.model,
        "provider": "openai",
        "example_request": {
            "message": "구미에서 가족끼리 갈만한 여행지나 맛집 추천해줘",
            "model": "gpt-4o-mini"
        },
        "example_response": {
            "reply": "구미에서 가족끼리 가기 좋은 장소로 ...",
            "success": True,
            "model": "gpt-4o-mini",
            "provider": "openai",
            "error": None
        }
    }


@app.get("/chatbot/logs", response_model=list[ChatbotLogItem])
def list_chatbot_logs():
    conn = get_conn()
    rows = conn.execute("SELECT id, flag, message_text FROM chatbot_log ORDER BY id").fetchall()
    conn.close()
    return [ChatbotLogItem(id=row["id"], flag=row["flag"], message_text=row["message_text"]) for row in rows]


@app.get("/chatbot/logs/{log_id}", response_model=ChatbotLogItem)
def get_chatbot_log(log_id: int):
    conn = get_conn()
    row = conn.execute("SELECT id, flag, message_text FROM chatbot_log WHERE id = ?", (log_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="chatbot log not found")
    return ChatbotLogItem(id=row["id"], flag=row["flag"], message_text=row["message_text"])


@app.delete("/chatbot/logs/{log_id}")
def delete_chatbot_log(log_id: int):
    conn = get_conn()
    row = conn.execute("SELECT id FROM chatbot_log WHERE id = ?", (log_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="chatbot log not found")
    conn.execute("DELETE FROM chatbot_log WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()
    return {"deleted": True, "logId": log_id}


@app.delete("/chatbot/logs")
def clear_chatbot_logs():
    conn = get_conn()
    conn.execute("DELETE FROM chatbot_log")
    conn.commit()
    conn.close()
    return {"deleted": True, "count": 0}


@app.get("/")
def root():
    return {
        "message": "LocalHub API connected to localhub.db",
        "db_path": str(DB_PATH),
        "tables": ["categories", "location", "post_final", "comment_final", "json_items"],
    }


def _now_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


@app.get("/categories")
def list_categories():
    conn = get_conn()
    rows = conn.execute("SELECT id, name FROM categories ORDER BY id").fetchall()
    conn.close()
    return {"total": len(rows), "categories": [row_to_dict(row) for row in rows]}


@app.get("/categories/{category_id}")
def get_category(category_id: int):
    conn = get_conn()
    row = conn.execute("SELECT id, name FROM categories WHERE id = ?", (category_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="category not found")

    return row_to_dict(row)


@app.post("/categories", status_code=201)
def create_category(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    conn = get_conn()
    try:
        cursor = conn.execute("INSERT INTO categories (name) VALUES (?)", (name,))
        conn.commit()
        category_id = cursor.lastrowid
        row = conn.execute("SELECT id, name FROM categories WHERE id = ?", (category_id,)).fetchone()
        conn.close()
        return row_to_dict(row)
    except sqlite3.Error as exc:
        conn.close()
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/categories/{category_id}")
def update_category(category_id: int, payload: dict):
    conn = get_conn()
    row = conn.execute("SELECT id, name FROM categories WHERE id = ?", (category_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="category not found")

    name = (payload.get("name") or "").strip()
    if not name:
        conn.close()
        raise HTTPException(status_code=400, detail="name is required")

    conn.execute("UPDATE categories SET name = ? WHERE id = ?", (name, category_id))
    conn.commit()
    updated_row = conn.execute("SELECT id, name FROM categories WHERE id = ?", (category_id,)).fetchone()
    conn.close()
    return row_to_dict(updated_row)


@app.delete("/categories/{category_id}")
def delete_category(category_id: int):
    conn = get_conn()
    row = conn.execute("SELECT id, name FROM categories WHERE id = ?", (category_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="category not found")

    conn.execute("DELETE FROM categories WHERE id = ?", (category_id,))
    conn.commit()
    conn.close()
    return {"deleted": True, "categoryId": category_id}


@app.get("/categories/search/{query}")
def search_categories(query: str):
    q = f"%{query}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name FROM categories WHERE LOWER(name) LIKE LOWER(?) ORDER BY id",
        (q,),
    ).fetchall()
    conn.close()
    return {"query": query, "total": len(rows), "categories": [row_to_dict(row) for row in rows]}


@app.get("/locations")
def list_locations():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, address, description, category_name, image_url, created_at, updated_at FROM location ORDER BY id"
    ).fetchall()
    conn.close()
    return {"total": len(rows), "locations": [row_to_dict(row) for row in rows]}


@app.get("/locations/{location_id}")
def get_location(location_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, address, description, category_name, image_url, created_at, updated_at FROM location WHERE id = ?",
        (location_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="location not found")

    return row_to_dict(row)


@app.post("/locations", status_code=201)
def create_location(payload: dict):
    name = (payload.get("name") or "").strip()
    address = (payload.get("address") or "").strip()
    if not name or not address:
        raise HTTPException(status_code=400, detail="name and address are required")

    description = payload.get("description")
    if description is not None:
        description = str(description).strip() or None

    category_name = payload.get("category_name")
    if category_name is not None:
        category_name = str(category_name).strip() or None

    image_url = payload.get("image_url")
    if image_url is not None:
        image_url = str(image_url).strip() or None

    now = _now_timestamp()

    conn = get_conn()
    try:
        cursor = conn.execute(
            "INSERT INTO location (name, address, description, category_name, image_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, address, description, category_name, image_url, now, now),
        )
        conn.commit()
        location_id = cursor.lastrowid
        row = conn.execute(
            "SELECT id, name, address, description, category_name, image_url, created_at, updated_at FROM location WHERE id = ?",
            (location_id,),
        ).fetchone()
        conn.close()
        return row_to_dict(row)
    except sqlite3.Error as exc:
        conn.close()
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/locations/{location_id}")
def update_location(location_id: int, payload: dict):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, address, description, category_name, image_url, created_at, updated_at FROM location WHERE id = ?",
        (location_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="location not found")

    name = payload.get("name")
    address = payload.get("address")
    description = payload.get("description")
    category_name = payload.get("category_name")
    image_url = payload.get("image_url")
    now = _now_timestamp()

    if name is None and address is None and description is None and category_name is None and image_url is None:
        conn.close()
        raise HTTPException(status_code=400, detail="at least one field is required")

    if name is not None:
        name = str(name).strip()
        if not name:
            conn.close()
            raise HTTPException(status_code=400, detail="name cannot be empty")
    if address is not None:
        address = str(address).strip()
        if not address:
            conn.close()
            raise HTTPException(status_code=400, detail="address cannot be empty")
    if description is not None:
        description = str(description).strip() or None
    if category_name is not None:
        category_name = str(category_name).strip() or None
    if image_url is not None:
        image_url = str(image_url).strip() or None

    if name is not None:
        conn.execute("UPDATE location SET name = ? WHERE id = ?", (name, location_id))
    if address is not None:
        conn.execute("UPDATE location SET address = ? WHERE id = ?", (address, location_id))
    if description is not None:
        conn.execute("UPDATE location SET description = ? WHERE id = ?", (description, location_id))
    if category_name is not None:
        conn.execute("UPDATE location SET category_name = ? WHERE id = ?", (category_name, location_id))
    if image_url is not None:
        conn.execute("UPDATE location SET image_url = ? WHERE id = ?", (image_url, location_id))
    conn.execute("UPDATE location SET updated_at = ? WHERE id = ?", (now, location_id))
    conn.commit()
    updated_row = conn.execute(
        "SELECT id, name, address, description, category_name, image_url, created_at, updated_at FROM location WHERE id = ?",
        (location_id,),
    ).fetchone()
    conn.close()
    return row_to_dict(updated_row)


@app.delete("/locations/{location_id}")
def delete_location(location_id: int):
    conn = get_conn()
    row = conn.execute("SELECT id FROM location WHERE id = ?", (location_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="location not found")

    conn.execute("DELETE FROM location WHERE id = ?", (location_id,))
    conn.commit()
    conn.close()
    return {"deleted": True, "locationId": location_id}


@app.get("/locations/search/{query}")
def search_locations(query: str):
    q = f"%{query}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, address, description, category_name, image_url, created_at, updated_at FROM location"
        " WHERE LOWER(name) LIKE LOWER(?) OR LOWER(address) LIKE LOWER(?) ORDER BY id",
        (q, q),
    ).fetchall()
    conn.close()
    return {"query": query, "total": len(rows), "locations": [row_to_dict(row) for row in rows]}


@app.get("/locations/search/name/{query}")
def search_locations_by_name(query: str):
    q = f"%{query}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, address, description, category_name, image_url, created_at, updated_at FROM location WHERE LOWER(name) LIKE LOWER(?) ORDER BY id",
        (q,),
    ).fetchall()
    conn.close()
    return {"query": query, "total": len(rows), "locations": [row_to_dict(row) for row in rows]}


@app.get("/locations/search/address/{query}")
def search_locations_by_address(query: str):
    q = f"%{query}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, address, description, category_name, image_url, created_at, updated_at FROM location WHERE LOWER(address) LIKE LOWER(?) ORDER BY id",
        (q,),
    ).fetchall()
    conn.close()
    return {"query": query, "total": len(rows), "locations": [row_to_dict(row) for row in rows]}


@app.get("/posts")
def list_posts():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final ORDER BY id"
    ).fetchall()
    conn.close()
    return {"total": len(rows), "posts": [row_to_dict(row) for row in rows]}


@app.get("/posts/{post_id}")
def get_post(post_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final WHERE id = ?",
        (post_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="post not found")

    return row_to_dict(row)


@app.post("/posts", status_code=201)
def create_post(payload: dict):
    title = (payload.get("title") or "").strip()
    content = (payload.get("content") or "").strip()
    password = (payload.get("password") or "").strip()
    nickname = (payload.get("nickname") or "").strip()

    if not title or not content or not password or not nickname:
        raise HTTPException(status_code=400, detail="title, content, password, and nickname are required")

    category_name = payload.get("category_name")
    if category_name is not None:
        category_name = str(category_name).strip() or None

    image_url = payload.get("image_url")
    if image_url is not None:
        image_url = str(image_url).strip() or None

    now = _now_timestamp()
    conn = get_conn()
    try:
        cursor = conn.execute(
            "INSERT INTO post_final (title, content, password, nickname, category_name, image_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (title, content, password, nickname, category_name, image_url, now, now),
        )
        conn.commit()
        post_id = cursor.lastrowid
        row = conn.execute(
            "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final WHERE id = ?",
            (post_id,),
        ).fetchone()
        conn.close()
        return row_to_dict(row)
    except sqlite3.Error as exc:
        conn.close()
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/posts/{post_id}")
def update_post(post_id: int, payload: dict):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final WHERE id = ?",
        (post_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="post not found")

    title = payload.get("title")
    content = payload.get("content")
    password = payload.get("password")
    nickname = payload.get("nickname")
    category_name = payload.get("category_name")
    image_url = payload.get("image_url")
    now = _now_timestamp()

    if title is None and content is None and password is None and nickname is None and category_name is None and image_url is None:
        conn.close()
        raise HTTPException(status_code=400, detail="at least one field is required")

    if title is not None:
        title = str(title).strip()
        if not title:
            conn.close()
            raise HTTPException(status_code=400, detail="title cannot be empty")
        conn.execute("UPDATE post_final SET title = ? WHERE id = ?", (title, post_id))
    if content is not None:
        content = str(content).strip()
        if not content:
            conn.close()
            raise HTTPException(status_code=400, detail="content cannot be empty")
        conn.execute("UPDATE post_final SET content = ? WHERE id = ?", (content, post_id))
    if password is not None:
        password = str(password).strip()
        if not password:
            conn.close()
            raise HTTPException(status_code=400, detail="password cannot be empty")
        conn.execute("UPDATE post_final SET password = ? WHERE id = ?", (password, post_id))
    if nickname is not None:
        nickname = str(nickname).strip()
        if not nickname:
            conn.close()
            raise HTTPException(status_code=400, detail="nickname cannot be empty")
        conn.execute("UPDATE post_final SET nickname = ? WHERE id = ?", (nickname, post_id))
    if category_name is not None:
        category_name = str(category_name).strip() or None
        conn.execute("UPDATE post_final SET category_name = ? WHERE id = ?", (category_name, post_id))
    if image_url is not None:
        image_url = str(image_url).strip() or None
        conn.execute("UPDATE post_final SET image_url = ? WHERE id = ?", (image_url, post_id))

    conn.execute("UPDATE post_final SET updated_at = ? WHERE id = ?", (now, post_id))
    conn.commit()
    updated_row = conn.execute(
        "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final WHERE id = ?",
        (post_id,),
    ).fetchone()
    conn.close()
    return row_to_dict(updated_row)


@app.delete("/posts/{post_id}")
def delete_post(post_id: int):
    conn = get_conn()
    row = conn.execute("SELECT id FROM post_final WHERE id = ?", (post_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="post not found")

    conn.execute("DELETE FROM post_final WHERE id = ?", (post_id,))
    conn.commit()
    conn.close()
    return {"deleted": True, "postId": post_id}


@app.get("/posts/search/{query}")
def search_posts(query: str):
    q = f"%{query}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final"
        " WHERE LOWER(title) LIKE LOWER(?) OR LOWER(content) LIKE LOWER(?) ORDER BY id",
        (q, q),
    ).fetchall()
    conn.close()

    return {
        "query": query,
        "total": len(rows),
        "posts": [row_to_dict(row) for row in rows],
    }


@app.get("/posts/search/title/{query}")
def search_posts_by_title(query: str):
    q = f"%{query}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final WHERE LOWER(title) LIKE LOWER(?) ORDER BY id",
        (q,),
    ).fetchall()
    conn.close()
    return {"query": query, "total": len(rows), "posts": [row_to_dict(row) for row in rows]}


@app.get("/posts/search/content/{query}")
def search_posts_by_content(query: str):
    q = f"%{query}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final WHERE LOWER(content) LIKE LOWER(?) ORDER BY id",
        (q,),
    ).fetchall()
    conn.close()
    return {"query": query, "total": len(rows), "posts": [row_to_dict(row) for row in rows]}


@app.get("/categories/{category_id}/posts")
def list_posts_by_category(category_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, content, password, nickname, category_name, image_url, created_at, updated_at FROM post_final WHERE category_name = ? ORDER BY id",
        (str(category_id),),
    ).fetchall()
    conn.close()
    return {"categoryId": category_id, "total": len(rows), "posts": [row_to_dict(row) for row in rows]}


@app.get("/comments")
def list_comments():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, post_id, content, nickname, password FROM comment_final ORDER BY id"
    ).fetchall()
    conn.close()
    return {"total": len(rows), "comments": [row_to_dict(row) for row in rows]}


@app.get("/comments/{comment_id}")
def get_comment(comment_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, post_id, content, nickname, password FROM comment_final WHERE id = ?",
        (comment_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="comment not found")

    return row_to_dict(row)


@app.post("/comments", status_code=201)
def create_comment(payload: dict):
    post_id = payload.get("post_id")
    content = (payload.get("content") or "").strip()
    nickname = (payload.get("nickname") or "").strip()
    password = (payload.get("password") or "").strip()

    if post_id is None or not content or not nickname or not password:
        raise HTTPException(status_code=400, detail="post_id, content, nickname, and password are required")

    conn = get_conn()
    post_row = conn.execute("SELECT id FROM post_final WHERE id = ?", (int(post_id),)).fetchone()
    if not post_row:
        conn.close()
        raise HTTPException(status_code=404, detail="post not found")

    try:
        cursor = conn.execute(
            "INSERT INTO comment_final (post_id, content, nickname, password) VALUES (?, ?, ?, ?)",
            (int(post_id), content, nickname, password),
        )
        conn.commit()
        comment_id = cursor.lastrowid
        row = conn.execute(
            "SELECT id, post_id, content, nickname, password FROM comment_final WHERE id = ?",
            (comment_id,),
        ).fetchone()
        conn.close()
        return row_to_dict(row)
    except sqlite3.Error as exc:
        conn.close()
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/comments/{comment_id}")
def update_comment(comment_id: int, payload: dict):
    conn = get_conn()
    row = conn.execute("SELECT id FROM comment_final WHERE id = ?", (comment_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="comment not found")

    content = payload.get("content")
    nickname = payload.get("nickname")
    password = payload.get("password")

    if content is None and nickname is None and password is None:
        conn.close()
        raise HTTPException(status_code=400, detail="at least one field is required")

    if content is not None:
        content = str(content).strip()
        if not content:
            conn.close()
            raise HTTPException(status_code=400, detail="content cannot be empty")
        conn.execute("UPDATE comment_final SET content = ? WHERE id = ?", (content, comment_id))
    if nickname is not None:
        nickname = str(nickname).strip()
        if not nickname:
            conn.close()
            raise HTTPException(status_code=400, detail="nickname cannot be empty")
        conn.execute("UPDATE comment_final SET nickname = ? WHERE id = ?", (nickname, comment_id))
    if password is not None:
        password = str(password).strip()
        if not password:
            conn.close()
            raise HTTPException(status_code=400, detail="password cannot be empty")
        conn.execute("UPDATE comment_final SET password = ? WHERE id = ?", (password, comment_id))

    conn.commit()
    updated_row = conn.execute(
        "SELECT id, post_id, content, nickname, password FROM comment_final WHERE id = ?",
        (comment_id,),
    ).fetchone()
    conn.close()
    return row_to_dict(updated_row)


@app.delete("/comments/{comment_id}")
def delete_comment(comment_id: int):
    conn = get_conn()
    row = conn.execute("SELECT id FROM comment_final WHERE id = ?", (comment_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="comment not found")

    conn.execute("DELETE FROM comment_final WHERE id = ?", (comment_id,))
    conn.commit()
    conn.close()
    return {"deleted": True, "commentId": comment_id}


@app.get("/posts/{post_id}/comments")
def list_comments_by_post(post_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, post_id, content, nickname, password FROM comment_final WHERE post_id = ? ORDER BY id",
        (post_id,),
    ).fetchall()
    conn.close()
    return {"postId": post_id, "total": len(rows), "comments": [row_to_dict(row) for row in rows]}


@app.get("/json-items")
def list_json_items(limit: int = 50, offset: int = 0):
    """
    모든 json_items 조회 (페이지네이션)
    - limit: 한 페이지당 항목 수 (기본: 50, 최대: 200)
    - offset: 시작 위치 (기본: 0)
    """
    limit = min(limit, 2000)
    offset = max(offset, 0)
    
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM json_items").fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM json_items ORDER BY id LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [json_row_to_payload(row) for row in rows]
    }


@app.get("/json-sources")
def list_json_sources():
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT source_file FROM json_items ORDER BY source_file").fetchall()
    conn.close()
    return {"sourceFiles": [row["source_file"] for row in rows]}


@app.get("/json-items/source/{source_file}")
def list_json_items_by_source(source_file: str):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM json_items WHERE source_file = ? ORDER BY id", (source_file,)).fetchall()
    conn.close()
    return {"sourceFile": source_file, "total": len(rows), "items": [json_row_to_payload(row) for row in rows]}


@app.get("/json-items/item/{source_file}/{content_id}")
def get_json_item(source_file: str, content_id: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM json_items WHERE source_file = ? AND content_id = ?",
        (source_file, content_id),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="item not found")

    return json_row_to_payload(row)


@app.get("/json-items/search/{query}")
def search_json_items(query: str):
    q = query.lower()
    conn = get_conn()
    rows = conn.execute("SELECT * FROM json_items ORDER BY id").fetchall()
    conn.close()

    items = []
    for row in rows:
        payload = json_row_to_payload(row)
        if q in str(payload.get("title", "")).lower() or q in str(payload.get("addr1", "")).lower():
            items.append(payload)

    return {
        "query": query,
        "total": len(items),
        "items": items,
    }


@app.get("/json-items/content-type/{content_type}")
def list_json_items_by_content_type(content_type: str):
    pattern = f"%{content_type}%"
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM json_items WHERE LOWER(content_type) LIKE LOWER(?) ORDER BY id",
        (pattern,),
    ).fetchall()
    conn.close()

    items = []
    for row in rows:
        payload = json_row_to_payload(row)
        items.append(_select_json_fields(payload))

    return {
        "contentType": content_type,
        "total": len(items),
        "items": items,
    }


@app.get("/json-items/addr1-search/{query}")
def search_json_items_by_addr1(query: str):
    q = query.lower()
    conn = get_conn()
    rows = conn.execute("SELECT * FROM json_items ORDER BY id").fetchall()
    conn.close()

    items = []
    for row in rows:
        payload = json_row_to_payload(row)
        if q in str(payload.get("addr1", "")).lower():
            items.append(_select_json_fields(payload))

    return {
        "query": query,
        "total": len(items),
        "items": items,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
