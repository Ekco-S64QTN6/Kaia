import contextlib
import logging
import os
import re
import string
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from sqlalchemy import (
    Column, DateTime, ForeignKey, Index, Integer, MetaData, Table, Text,
    UniqueConstraint, create_engine, func, inspect as sqlalchemy_inspect, select
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

load_dotenv()

DB_USER = os.getenv('KAIA_DB_USER', 'kaiauser')
DB_PASS = os.getenv('KAIA_DB_PASS', '')
DB_HOST = os.getenv('KAIA_DB_HOST', 'localhost')
DB_NAME = os.getenv('KAIA_DB_NAME', 'kaiadb')

DB_URL_OBJECT = URL.create(
    "postgresql",
    username=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    database=DB_NAME
)
DB_PATH = str(DB_URL_OBJECT)

engine = None
Session = None
metadata = MetaData()

# Table Definitions
users_table = Table(
    'users', metadata,
    Column('user_id', Text, primary_key=True),
    Column('created_at', DateTime(timezone=True), default=func.now())
)

user_preferences_table = Table(
    'user_preferences', metadata,
    Column('preference_id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.user_id'), nullable=False),
    Column('preference_key', Text, nullable=False),
    Column('preference_value', Text),
    Column('last_updated', DateTime(timezone=True), default=func.now(), onupdate=func.now()),
    UniqueConstraint('user_id', 'preference_key', name='uq_user_preference_key')
)

facts_table = Table(
    'facts', metadata,
    Column('fact_id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.user_id'), nullable=False),
    Column('fact_text', Text, nullable=False),
    Column('created_at', DateTime(timezone=True), default=func.now()),
    Index('idx_user_fact', 'user_id')
)

interaction_history_table = Table(
    'interaction_history', metadata,
    Column('interaction_id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.user_id'), nullable=False),
    Column('timestamp', DateTime(timezone=True), default=func.now()),
    Column('user_query', Text, nullable=False),
    Column('kaia_response', Text, nullable=False),
    Column('response_type', Text)
)

Index("idx_interaction_timestamp", interaction_history_table.c.timestamp)

def normalize_query(query: str) -> str:
    """Normalizes a user query by lowercasing and removing punctuation."""
    translator = str.maketrans('', '', string.punctuation)
    return query.lower().translate(translator).strip()

def match_query_category(clean_query: str) -> str:
    """Categorizes a query using regex for more flexible matching."""
    patterns = {
        "about_me": r"\b(about me|know about me|my information|my data|memories|tell me about myself|what do you know)\b",
        "preferences": r"\b(preferences|settings|options|theme|mode|favorite|my preference)\b",
        "facts": r"\b(facts|remembered|stored|what have you remembered)\b",
        "history": r"\b(history|conversations|past|previous|interactions|what have we talked about)\b",
    }

    for category, pattern in patterns.items():
        if re.search(pattern, clean_query, re.IGNORECASE):
            return category

    return "unknown"

def initialize_db() -> bool:
    """Initializes the PostgreSQL database and creates tables if they don't exist."""
    global engine, Session
    logging.info("Initializing PostgreSQL database")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            engine = create_engine(DB_PATH, pool_size=10, max_overflow=20, connect_args={"sslmode": "prefer"})
            Session = sessionmaker(bind=engine)
            metadata.create_all(engine)
            logging.info(f"PostgreSQL database initialized successfully for {DB_PATH}")
            return True
        except OperationalError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"Connection failed (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Could not connect to database after {max_retries} attempts: {e}")
                return False
        except SQLAlchemyError as e:
            logging.error(f"Database initialization error: {e}")
            return False
    return False

@contextlib.contextmanager
def get_session():
    """Provides a SQLAlchemy session for database operations."""
    if Session is None or engine is None:
        raise RuntimeError("Database not initialized. Call initialize_db() first.")
    session = Session()
    try:
        yield session
    finally:
        session.close()

def ensure_user(user_id: str):
    """Ensures a user exists in the database, adding them if not present."""
    with get_session() as session:
        try:
            stmt = postgresql.insert(users_table).values(user_id=user_id)
            stmt = stmt.on_conflict_do_nothing(index_elements=['user_id'])
            session.execute(stmt)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Error ensuring user '{user_id}': {e}")

def handle_memory_storage(user_id: str, content: str) -> Tuple[bool, str]:
    """Stores user preferences or facts in the database."""
    with get_session() as session:
        try:
            pref_match = re.match(r"(?:i prefer|my preference is|my preferred (?:editor|theme|mode) is)\s*(.+)", content, re.I)
            if pref_match:
                preference_phrase = pref_match.group(1).strip()
                key = ""
                value = ""
                key_value_match = re.match(r"(.+?)(?:\s+is\s+|=)\s*(.+)", preference_phrase, re.I)
                if key_value_match:
                    key = key_value_match.group(1).strip()
                    value = key_value_match.group(2).strip()
                else:
                    key = preference_phrase
                    value = "enabled"

                if not key:
                    return False, "Please specify what preference you want me to remember (e.g., 'dark mode' or 'my theme is dark')."

                stmt = postgresql.insert(user_preferences_table).values(
                    user_id=user_id,
                    preference_key=key,
                    preference_value=value
                )
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['user_id', 'preference_key'],
                    set_=dict(preference_value=stmt.excluded.preference_value, last_updated=func.now())
                )
                session.execute(on_conflict_stmt)
                session.commit()
                logging.info(f"Preference '{key}: {value}' stored for user '{user_id}'.")
                return True, f"Okay, I'll remember that your preference for '{key}' is '{value}'."

            fact_text = content.strip()
            if not fact_text:
                return False, "Please provide content to remember."

            session.execute(facts_table.insert().values(user_id=user_id, fact_text=fact_text))
            session.commit()
            logging.info(f"Fact '{fact_text}' stored for user '{user_id}'.")
            return True, f"Got it. I'll remember that: {fact_text}."

        except re.error as e:
            session.rollback()
            logging.error(f"Preference parsing regex error: {e}", exc_info=True)
            return False, f"Preference parsing error: {e}"
        except IntegrityError:
            session.rollback()
            return False, "There was a database error storing that. It might be a duplicate."
        except Exception as e:
            session.rollback()
            logging.error(f"Error storing memory: {e}", exc_info=True)
            return False, f"An unexpected error occurred while trying to remember that: {e}"

def handle_data_retrieval(user_id: str, query: str) -> Dict[str, Any]:
    """Retrieves user data based on the query category."""
    with get_session() as session:
        try:
            clean_query = normalize_query(query)
            category = match_query_category(clean_query)

            if category == "about_me":
                return get_user_profile(session, user_id)
            elif category == "preferences":
                return get_user_preferences(session, user_id)
            elif category == "facts":
                return get_user_facts(session, user_id)
            elif category == "history":
                return get_interaction_history(session, user_id)
            else:
                return {
                    'message': "I can retrieve your preferences, facts, or interaction history. Please be more specific.",
                    'data': [],
                    'response_type': "unhandled_retrieval_query"
                }

        except SQLAlchemyError as e:
            logging.error(f"Database error during retrieval: {e}", exc_info=True)
            return {
                'message': "A database error occurred while retrieving your information.",
                'data': [],
                'response_type': "retrieval_error"
            }
        except Exception as e:
            logging.error(f"Unexpected retrieval error: {e}", exc_info=True)
            return {
                'message': "An unexpected error occurred while processing your request.",
                'data': [],
                'response_type': "retrieval_error"
            }

def get_user_profile(session, user_id: str) -> Dict[str, Any]:
    """Retrieves a comprehensive profile of the user."""
    all_data = []

    prefs = session.execute(
        select(
            user_preferences_table.c.preference_key,
            user_preferences_table.c.preference_value
        ).filter_by(user_id=user_id)
    ).fetchall()

    if prefs:
        all_data.append("Your preferences:")
        all_data.extend(f"• {key}: {value}" for key, value in prefs)
    else:
        all_data.append("You haven't told me any preferences yet.")

    facts = session.execute(
        select(facts_table.c.fact_text)
        .filter_by(user_id=user_id)
    ).fetchall()

    if facts:
        all_data.append("\nFacts I remember:")
        all_data.extend(f"• {fact[0]}" for fact in facts)
    else:
        all_data.append("\nI haven't stored any facts for you yet.")

    return {
        'message': "Here's what I know about you:",
        'data': all_data,
        'response_type': "user_profile_retrieved"
    }

def get_user_preferences(session, user_id: str) -> Dict[str, Any]:
    """Retrieves user-defined preferences."""
    prefs = session.execute(
        select(
            user_preferences_table.c.preference_key,
            user_preferences_table.c.preference_value
        ).filter_by(user_id=user_id)
    ).fetchall()

    if prefs:
        return {
            'message': "Your preferences:",
            'data': [f"{key}: {value}" for key, value in prefs],
            'response_type': "preferences_retrieved"
        }
    return {
        'message': "You haven't told me any preferences yet.",
        'data': [],
        'response_type': "no_preferences"
    }

def get_user_facts(session, user_id: str) -> Dict[str, Any]:
    """Retrieves facts stored for the user."""
    facts = session.execute(
        select(facts_table.c.fact_text)
        .filter_by(user_id=user_id)
    ).fetchall()

    if facts:
        return {
            'message': "Facts I remember:",
            'data': [fact[0] for fact in facts],
            'response_type': "facts_retrieved"
        }
    return {
        'message': "I haven't stored any facts for you yet.",
        'data': [],
        'response_type': "no_facts"
    }

def get_interaction_history(session, user_id: str) -> Dict[str, Any]:
    """Retrieves recent interaction history for the user."""
    history = session.execute(
        select(
            interaction_history_table.c.timestamp,
            interaction_history_table.c.user_query,
            interaction_history_table.c.kaia_response
        )
        .filter_by(user_id=user_id)
        .order_by(interaction_history_table.c.timestamp.desc())
        .limit(10)
    ).fetchall()

    if history:
        formatted = []
        for timestamp, query, response in history:
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            truncated_response = (response[:70] + '...') if len(response) > 70 else response
            formatted.append(f"[{time_str}] You: {query} | Kaia: {truncated_response}")
        return {
            'message': "Recent interactions:",
            'data': formatted,
            'response_type': "history_retrieved"
        }
    return {
        'message': "No interaction history found.",
        'data': [],
        'response_type': "no_history"
    }

def log_interaction(user_id: str, user_query: str, kaia_response: str, response_type: str):
    """Logs a user interaction and Kaia's response."""
    with get_session() as session:
        try:
            session.execute(interaction_history_table.insert().values(
                user_id=user_id,
                user_query=user_query,
                kaia_response=kaia_response,
                response_type=response_type
            ))
            session.commit()
            logging.info(f"Interaction logged for user '{user_id}'.")
        except Exception as e:
            session.rollback()
            logging.error(f"Error logging interaction for user '{user_id}': {e}", exc_info=True)

def get_database_status() -> Dict:
    """Checks the connection status and lists tables in the database."""
    if not engine:
        return {'connected': False, 'error': 'Engine not initialized', 'tables': []}

    try:
        inspector = sqlalchemy_inspect(engine)
        return {
            'connected': True,
            'tables': inspector.get_table_names()
        }
    except Exception as e:
        logging.error(f"Error getting database status: {e}", exc_info=True)
        return {
            'connected': False,
            'error': str(e),
            'tables': []
        }

def get_current_user() -> str:
    """Retrieves the current system user's login name."""
    try:
        return os.getlogin()
    except (OSError, AttributeError):
        # Fallback for environments where getlogin() might fail (e.g., cron jobs)
        return os.environ.get('USER', 'default_user')
