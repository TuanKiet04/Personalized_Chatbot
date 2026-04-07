# app/main.py
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import engine, Base, get_db
from app.models import User, UserInteraction

# Tự động tạo bảng trong DB nếu chưa có
Base.metadata.create_all(bind=engine)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.post("/login")
async def process_login(
    request: Request, 
    username: str = Form(...), 
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    # Tìm user theo username
    user = db.query(User).filter(User.username == username).first()
    
    # Nếu chưa có user, tạo mới (Vibecode: chế tạm email từ username để ko lỗi Not Null)
    if not user:
        new_user = User(
            username=username, 
            email=f"{username}@example.com", # Tạo tạm email
            password_hash=password            # Lưu tạm pass thô, sau này dùng pass_hash sau
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        user = new_user

    # Kiểm tra mật khẩu (so sánh với password_hash trong DB của bạn)
    if user.password_hash == password:
        response = RedirectResponse(url="/news", status_code=302)
        response.set_cookie(key="session_user", value=username)
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error_msg": "Sai mật khẩu!"
        })

@app.get("/login", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/log-interaction/{article_id}")
async def log_interaction(
    article_id: int, 
    request: Request, 
    db: Session = Depends(get_db)
):
    # Lấy username từ cookie để biết ai đang click
    username = request.cookies.get("session_user")
    if not username:
        return {"status": "ignored"}
    
    user = db.query(User).filter(User.username == username).first()
    if user:
        # Lưu hành vi click vào DB
        new_interaction = UserInteraction(
            user_id=user.id,
            article_id=article_id,
            action="click"
        )
        db.add(new_interaction)
        db.commit()
        return {"status": "success", "article_id": article_id}
    
    return {"status": "user_not_found"}

@app.get("/register", response_class=HTMLResponse)
async def get_register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def process_register(
    request: Request, 
    username: str = Form(...), 
    email: str = Form(...),
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    # 1. Kiểm tra username hoặc email đã tồn tại chưa
    existing_user = db.query(User).filter((User.username == username) | (User.email == email)).first()
    if existing_user:
        return templates.TemplateResponse("register.html", {
            "request": request, 
            "error_msg": "Username hoặc Email đã được sử dụng!"
        })

    # 2. Tạo user mới (Ở phase này ta vẫn lưu pass thô theo ý bạn, sau này sẽ hash sau)
    new_user = User(
        username=username,
        email=email,
        password_hash=password
    )
    
    try:
        db.add(new_user)
        db.commit()
        # Đăng ký xong cho đăng nhập luôn
        response = RedirectResponse(url="/news", status_code=302)
        response.set_cookie(key="session_user", value=username)
        return response
    except Exception as e:
        db.rollback()
        return templates.TemplateResponse("register.html", {
            "request": request, 
            "error_msg": "Có lỗi xảy ra khi tạo tài khoản!"
        })

# Cập nhật lại route "/" để làm trang chủ (Landing Page)
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/news", response_class=HTMLResponse)
async def get_news_page(request: Request, db: Session = Depends(get_db), page: int = 1):
    current_user = request.cookies.get("session_user")
    if not current_user:
        return RedirectResponse(url="/login")
    
    size = 15 # Số bài mỗi trang
    offset = (page - 1) * size
    
    query = text(f"""
        SELECT id, title, url, topic, 
               to_char(published_at, 'DD/MM/YYYY HH24:MI') as published_at, 
               LEFT(content, 150) as snippet
        FROM public.raw_data 
        ORDER BY published_at DESC 
        LIMIT {size} OFFSET {offset}
    """)
    
    try:
        result = db.execute(query).mappings().all()
        articles = [dict(row) for row in result]
    except Exception as e:
        print(f"Lỗi: {e}")
        articles = [] 

    return templates.TemplateResponse("news.html", {
        "request": request, 
        "username": current_user,
        "articles": articles,
        "next_page": page + 1,
        "prev_page": page - 1 if page > 1 else None
    })

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302) # Sửa thành "/"
    response.delete_cookie("session_user")
    return response

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    # Ở Phase 2, chúng ta sẽ đưa RAG và Ollama vào đây
    # Hiện tại trả về phản hồi giả lập để test giao diện
    user_msg = request.message.lower()
    
    if "xin chào" in user_msg:
        reply = "Chào bạn! Tôi là trợ lý tin tức cá nhân hóa của bạn."
    elif "tin tức" in user_msg:
        reply = "Hôm nay có khá nhiều tin mới về Công nghệ và Kinh tế, bạn muốn xem chủ đề nào?"
    else:
        reply = f"Tôi đã nhận được câu hỏi: '{request.message}'. Tôi đang học hỏi để trả lời tốt hơn!"
        
    return {"response": reply}