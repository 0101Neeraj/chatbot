main.py


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import api_router
from .models import init_db_and_seed


def create_app() -> FastAPI:
	app = FastAPI(
		title="AI Campus Assistant API",
		version="1.0.0",
		description="REST API chatbot to help students discover and enroll in AI/CS courses",
		docs_url="/docs",
		redoc_url="/redoc",
	)

	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	@app.on_event("startup")
	def on_startup() -> None:
		init_db_and_seed()

	app.include_router(api_router)
	return app


app = create_app()


models.py

from __future__ import annotations

import datetime as dt
from typing import Generator, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

DATABASE_URL = "sqlite:///ai_campus_chatbot_api/database.db"

engine = create_engine(
	DATABASE_URL,
	connect_args={"check_same_thread": False},
	future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


class Course(Base):
	__tablename__ = "courses"

	id = Column(Integer, primary_key=True, index=True)
	code = Column(String(50), unique=True, nullable=False, index=True)
	name = Column(String(255), nullable=False)
	category = Column(String(100), nullable=False)
	instructor = Column(String(100), nullable=False)
	keywords = Column(Text, nullable=True)  # comma-separated

	enrollments = relationship("Enrollment", back_populates="course", cascade="all, delete-orphan")


class SessionModel(Base):
	__tablename__ = "sessions"

	id = Column(Integer, primary_key=True, index=True)
	session_id = Column(String(100), unique=True, index=True, nullable=False)
	user_email = Column(String(255), index=True, nullable=False)
	started_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

	messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
	enrollments = relationship("Enrollment", back_populates="session", cascade="all, delete-orphan")


class Enrollment(Base):
	__tablename__ = "enrollments"

	id = Column(Integer, primary_key=True, index=True)
	session_id = Column(String(100), ForeignKey("sessions.session_id"), nullable=False, index=True)
	course_code = Column(String(50), ForeignKey("courses.code"), nullable=False)
	student_name = Column(String(255), nullable=False)
	student_email = Column(String(255), nullable=False)
	enrolled_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

	session = relationship("SessionModel", back_populates="enrollments", primaryjoin="Enrollment.session_id==SessionModel.session_id")
	course = relationship("Course", back_populates="enrollments", primaryjoin="Enrollment.course_code==Course.code")


class ChatMessage(Base):
	__tablename__ = "chat_messages"

	id = Column(Integer, primary_key=True, index=True)
	session_id = Column(String(100), ForeignKey("sessions.session_id"), nullable=False, index=True)
	user_message = Column(Text, nullable=False)
	bot_response = Column(Text, nullable=False)
	intent = Column(String(50), nullable=False)
	created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

	session = relationship("SessionModel", back_populates="messages", primaryjoin="ChatMessage.session_id==SessionModel.session_id")


# DB helpers

def get_db() -> Generator[Session, None, None]:
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()


def init_db_and_seed() -> None:
	"""Create tables and seed default courses if not present."""
	Base.metadata.create_all(bind=engine)
	with SessionLocal() as db:
		# Seed default courses if empty
		count = db.scalar(select(Integer).select_from(Course))
		# The above select doesn't count; use proper count
		num_courses = db.query(Course).count()
		if num_courses == 0:
			default_courses = [
				{"code": "CS102", "name": "Python for AI", "category": "AI Fundamentals", "instructor": "Dr. Smith", "keywords": "python, beginner, programming"},
				{"code": "ML201", "name": "ML Fundamentals", "category": "Machine Learning", "instructor": "Prof. Lee", "keywords": "machine learning, ml, supervised"},
				{"code": "DL301", "name": "Deep Learning", "category": "Deep Learning", "instructor": "Dr. Kim", "keywords": "deep learning, neural network, tensorflow"},
				{"code": "NLP201", "name": "NLP Basics", "category": "NLP", "instructor": "Dr. Patel", "keywords": "nlp, text, language"},
			]
			for c in default_courses:
				db.add(Course(**c))
			db.commit()


routes.py

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from .models import get_db
from .schemas import (
	ChatRequest,
	ChatResponse,
	CourseOut,
	EnrollRequest,
	EnrollmentOut,
	SessionCreate,
	SessionOut,
)
from .services import (
	ChatBot,
	create_enrollment,
	create_session,
	get_session_by_id,
	get_sessions_by_email,
	list_courses,
	list_enrollments,
	log_chat,
	search_courses,
)

api_router = APIRouter()


@api_router.get("/", summary="Welcome endpoint")
async def root() -> dict:
	return {
		"message": "Welcome to AI Campus Assistant API. See /docs for API documentation.",
	}


@api_router.post("/api/session", response_model=SessionOut)
def api_create_session(payload: SessionCreate, db=Depends(get_db)):
	session = create_session(db, payload.user_email)
	return session


@api_router.get("/api/courses", response_model=List[CourseOut])
def api_list_courses(db=Depends(get_db)):
	return list_courses(db)


@api_router.get("/api/courses/search", response_model=List[CourseOut])
def api_search_courses(keyword: str, db=Depends(get_db)):
	if not keyword:
		raise HTTPException(status_code=400, detail="keyword is required")
	return search_courses(db, keyword)


@api_router.get("/api/enrollments", response_model=List[EnrollmentOut])
def api_list_enrollments(db=Depends(get_db)):
	return list_enrollments(db)


@api_router.post("/api/enroll", response_model=EnrollmentOut)
def api_enroll(payload: EnrollRequest, db=Depends(get_db)):
	# validate course exists
	courses = search_courses(db, payload.course_code)
	if not any(c.code.lower() == payload.course_code.lower() for c in courses):
		raise HTTPException(status_code=404, detail="Course not found")
	# validate session exists
	if not get_session_by_id(db, payload.session_id):
		raise HTTPException(status_code=404, detail="Session not found")
	return create_enrollment(
		db,
		payload.session_id,
		payload.course_code,
		payload.student_name,
		payload.student_email,
	)


@api_router.post("/api/chat", response_model=ChatResponse)
def api_chat(payload: ChatRequest, db=Depends(get_db)):
	# Ensure session
	if payload.session_id:
		session = get_session_by_id(db, payload.session_id)
		if not session:
			raise HTTPException(status_code=404, detail="Session not found")
	else:
		session = create_session(db, payload.user_email)

	bot = ChatBot(db)
	intent, response_text, matched = bot.process_message(payload.message)

	# log
	log_chat(
		db,
		session.session_id,
		payload.message,
		response_text,
		intent,
	)

	return ChatResponse(
		intent=intent,
		response=response_text,
		matched_courses=[CourseOut.model_validate(c) for c in matched],
		session_id=session.session_id,
	)


@api_router.get("/api/chat-history")
def api_chat_history(email: str, db=Depends(get_db)):
	sessions = get_sessions_by_email(db, email)
	return {
		"email": email,
		"sessions": [
			{"session_id": s.session_id, "started_at": s.started_at}
			for s in sessions
		],
	}


schemas.py

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# Course
class CourseOut(BaseModel):
	code: str
	name: str
	category: str
	instructor: str
	keywords: Optional[str] = None

	class Config:
		from_attributes = True


# Session
class SessionCreate(BaseModel):
	user_email: EmailStr


class SessionOut(BaseModel):
	session_id: str
	user_email: EmailStr
	started_at: dt.datetime

	class Config:
		from_attributes = True


# Chat
class ChatRequest(BaseModel):
	user_email: EmailStr
	message: str
	session_id: Optional[str] = None


class ChatResponse(BaseModel):
	intent: str
	response: str
	matched_courses: List[CourseOut] = Field(default_factory=list)
	session_id: str


# Enrollment
class EnrollRequest(BaseModel):
	session_id: str
	course_code: str
	student_name: str
	student_email: EmailStr


class EnrollmentOut(BaseModel):
	id: int
	session_id: str
	course_code: str
	student_name: str
	student_email: EmailStr
	enrolled_at: dt.datetime

	class Config:
		from_attributes = True


# History
class ChatMessageOut(BaseModel):
	id: int
	session_id: str
	user_message: str
	bot_response: str
	intent: str
	created_at: dt.datetime

	class Config:
		from_attributes = True


services.py

from __future__ import annotations

import datetime as dt
import uuid
from typing import List, Optional, Tuple

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .models import ChatMessage, Course, Enrollment, SessionModel


# Service utilities

def create_session(db: Session, user_email: str) -> SessionModel:
	session = SessionModel(session_id=str(uuid.uuid4()), user_email=user_email)
	db.add(session)
	db.commit()
	db.refresh(session)
	return session


def get_session_by_id(db: Session, session_id: str) -> Optional[SessionModel]:
	return db.query(SessionModel).filter(SessionModel.session_id == session_id).first()


def get_sessions_by_email(db: Session, user_email: str) -> List[SessionModel]:
	return db.query(SessionModel).filter(SessionModel.user_email == user_email).order_by(SessionModel.started_at.desc()).all()


def list_courses(db: Session) -> List[Course]:
	return db.query(Course).order_by(Course.code.asc()).all()


def search_courses(db: Session, keyword: str) -> List[Course]:
	kw = f"%{keyword.lower()}%"
	return (
		db.query(Course)
		.filter(
			or_(
				Course.name.ilike(kw),
				Course.category.ilike(kw),
				Course.instructor.ilike(kw),
				Course.code.ilike(kw),
				Course.keywords.ilike(kw),
			)
		)
		.order_by(Course.code.asc())
		.all()
	)


def create_enrollment(
	db: Session,
	session_id: str,
	course_code: str,
	student_name: str,
	student_email: str,
) -> Enrollment:
	enroll = Enrollment(
		session_id=session_id,
		course_code=course_code,
		student_name=student_name,
		student_email=student_email,
	)
	db.add(enroll)
	db.commit()
	db.refresh(enroll)
	return enroll


def list_enrollments(db: Session) -> List[Enrollment]:
	return db.query(Enrollment).order_by(Enrollment.enrolled_at.desc()).all()


def log_chat(
	db: Session,
	session_id: str,
	user_message: str,
	bot_response: str,
	intent: str,
) -> ChatMessage:
	msg = ChatMessage(
		session_id=session_id,
		user_message=user_message,
		bot_response=bot_response,
		intent=intent,
	)
	db.add(msg)
	db.commit()
	db.refresh(msg)
	return msg


# ChatBot class
class ChatBot:
	greeting_keywords = {"hi", "hello", "hey"}
	list_keywords = {"list", "all courses", "show courses"}
	search_keywords = {"search", "find", "looking", "look for"}
	enroll_keywords = {"enroll", "register", "join"}
	exit_keywords = {"bye", "exit", "quit"}

	def __init__(self, db: Session):
		self.db = db

	def detect_intent(self, message: str) -> str:
		m = message.lower().strip()
		if any(k in m for k in self.greeting_keywords):
			return "greeting"
		if any(k in m for k in self.exit_keywords):
			return "exit"
		if any(k in m for k in self.enroll_keywords):
			return "enroll"
		if any(k in m for k in self.list_keywords):
			return "list"
		if any(k in m for k in self.search_keywords):
			return "search"
		# Heuristic: if message contains known course/category keywords, treat as search
		if self._message_contains_course_keyword(m):
			return "search"
		return "unknown"

	def _message_contains_course_keyword(self, m: str) -> bool:
		courses = list_courses(self.db)
		for c in courses:
			all_kw = f"{c.name} {c.category} {c.instructor} {c.code} {c.keywords or ''}".lower()
			# match any token
			for token in m.split():
				if token and token in all_kw:
					return True
		return False

	def find_courses(self, message: str) -> List[Course]:
		m = message.lower()
		matches = []
		for c in list_courses(self.db):
			kw_blob = f"{c.name} {c.category} {c.instructor} {c.code} {c.keywords or ''}".lower()
			if any(token in kw_blob for token in m.split()):
				matches.append(c)
		return matches

	def process_message(self, message: str) -> Tuple[str, str, List[Course]]:
		"""Return (intent, response, matched_courses)."""
		intent = self.detect_intent(message)
		if intent == "greeting":
			return intent, (
				"Hello! I can help you discover AI/CS courses. "
				"Try: 'list courses', 'search deep learning', or 'enroll in ML201'."
			), []
		if intent == "exit":
			return intent, "Goodbye! Feel free to come back anytime.", []
		if intent == "list":
			courses = list_courses(self.db)
			if not courses:
				return intent, "No courses available right now.", []
			lines = [f"- {c.name} ({c.code}) — {c.category} by {c.instructor}" for c in courses]
			return intent, "Here are the available courses:\n" + "\n".join(lines), courses
		if intent == "search":
			courses = self.find_courses(message)
			if not courses:
				return intent, "I couldn't find courses for that query.", []
			lines = [f"- {c.name} ({c.code}) — {c.category}" for c in courses]
			return intent, "I found these matching courses:\n" + "\n".join(lines), courses
		if intent == "enroll":
			return intent, (
				"Sure! Tell me the course code and your name/email, or use the /api/enroll endpoint."
			), []
		# unknown
		return intent, (
			"I'm not sure I understood. You can say 'list courses', 'search NLP', or 'enroll in CS102'."
		), []
