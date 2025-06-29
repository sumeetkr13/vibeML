import logging
import json
import os
from datetime import datetime
from pathlib import Path
import streamlit as st
from functools import wraps

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'client_ip'):
            log_entry['client_ip'] = record.client_ip
        if hasattr(record, 'event_type'):
            log_entry['event_type'] = record.event_type
        if hasattr(record, 'data'):
            log_entry['data'] = record.data
            
        return json.dumps(log_entry, ensure_ascii=False)

# Set up loggers
def setup_loggers():
    """Set up different loggers for different types of events"""
    
    # Main app logger
    app_logger = logging.getLogger('ml_app')
    app_logger.setLevel(logging.INFO)
    
    # User activity logger
    user_logger = logging.getLogger('user_activity')
    user_logger.setLevel(logging.INFO)
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    
    # Error logger
    error_logger = logging.getLogger('errors')
    error_logger.setLevel(logging.ERROR)
    
    # Create handlers
    handlers = {
        'app': logging.FileHandler(LOGS_DIR / 'app.log'),
        'user': logging.FileHandler(LOGS_DIR / 'user_activity.log'),
        'performance': logging.FileHandler(LOGS_DIR / 'performance.log'),
        'errors': logging.FileHandler(LOGS_DIR / 'errors.log')
    }
    
    # Set up JSON formatter for all handlers
    formatter = JSONFormatter()
    for handler in handlers.values():
        handler.setFormatter(formatter)
    
    # Add handlers to loggers
    app_logger.addHandler(handlers['app'])
    user_logger.addHandler(handlers['user'])
    perf_logger.addHandler(handlers['performance'])
    error_logger.addHandler(handlers['errors'])
    
    # Prevent duplicate logs
    for logger in [app_logger, user_logger, perf_logger, error_logger]:
        logger.propagate = False
    
    return app_logger, user_logger, perf_logger, error_logger

# Initialize loggers
app_logger, user_logger, perf_logger, error_logger = setup_loggers()

def get_client_ip():
    """Get client IP address from Streamlit request headers"""
    try:
        # Try to get IP from Streamlit context (newer versions)
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            headers = st.context.headers
            # Check common proxy headers first
            for header in ['X-Forwarded-For', 'X-Real-IP', 'CF-Connecting-IP', 'X-Client-IP']:
                if header in headers:
                    ip = headers[header].split(',')[0].strip()
                    return ip
            
            # Fallback to direct IP
            if 'Remote-Addr' in headers:
                return headers['Remote-Addr']
        
        # Try accessing through session info (alternative method)
        if hasattr(st, 'session_state') and hasattr(st.session_state, '_state'):
            try:
                # This is a more direct approach for some Streamlit versions
                import streamlit.web.server.websocket_headers as wsh
                headers = wsh.get_websocket_headers()
                if headers:
                    for header in ['X-Forwarded-For', 'X-Real-IP', 'CF-Connecting-IP']:
                        if header.lower() in headers:
                            return headers[header.lower()].split(',')[0].strip()
            except:
                pass
        
        # Try environment variables (set by some reverse proxies)
        import os
        for env_var in ['HTTP_X_FORWARDED_FOR', 'HTTP_X_REAL_IP', 'REMOTE_ADDR']:
            if env_var in os.environ:
                return os.environ[env_var].split(',')[0].strip()
        
        # Local development fallback
        return '127.0.0.1'
        
    except Exception:
        # If all else fails, return localhost
        return '127.0.0.1'

def get_session_info():
    """Get current session information including IP address"""
    session_id = getattr(st.session_state, 'session_id', 'unknown')
    user_id = getattr(st.session_state, 'username', 'anonymous')
    client_ip = get_client_ip()
    return session_id, user_id, client_ip

def log_user_activity(event_type, message, data=None):
    """Log user activity events"""
    session_id, user_id, client_ip = get_session_info()
    
    user_logger.info(
        message,
        extra={
            'event_type': event_type,
            'user_id': user_id,
            'session_id': session_id,
            'client_ip': client_ip,
            'data': data or {}
        }
    )

def log_app_event(event_type, message, data=None):
    """Log general app events"""
    session_id, user_id, client_ip = get_session_info()
    
    app_logger.info(
        message,
        extra={
            'event_type': event_type,
            'user_id': user_id,
            'session_id': session_id,
            'client_ip': client_ip,
            'data': data or {}
        }
    )

def log_performance(event_type, message, duration=None, data=None):
    """Log performance metrics"""
    session_id, user_id, client_ip = get_session_info()
    
    perf_data = data or {}
    if duration is not None:
        perf_data['duration_seconds'] = duration
    
    perf_logger.info(
        message,
        extra={
            'event_type': event_type,
            'user_id': user_id,
            'session_id': session_id,
            'client_ip': client_ip,
            'data': perf_data
        }
    )

def log_error(event_type, message, error=None, data=None):
    """Log error events"""
    session_id, user_id, client_ip = get_session_info()
    
    error_data = data or {}
    if error is not None:
        error_data['error_type'] = type(error).__name__
        error_data['error_message'] = str(error)
    
    error_logger.error(
        message,
        extra={
            'event_type': event_type,
            'user_id': user_id,
            'session_id': session_id,
            'client_ip': client_ip,
            'data': error_data
        }
    )

def track_time(event_type, description=None):
    """Decorator to track execution time of functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                log_performance(
                    event_type=event_type,
                    message=description or f"Function {func.__name__} completed",
                    duration=duration,
                    data={'function_name': func.__name__}
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                log_error(
                    event_type=f"{event_type}_error",
                    message=f"Function {func.__name__} failed",
                    error=e,
                    data={'function_name': func.__name__, 'duration_seconds': duration}
                )
                raise
        return wrapper
    return decorator

# Specific logging functions for common events
def log_login(username, success=True):
    """Log user login attempts"""
    event_type = "login_success" if success else "login_failure"
    message = f"User login {'successful' if success else 'failed'}"
    log_user_activity(event_type, message, {'username': username})

def log_logout(username):
    """Log user logout"""
    log_user_activity("logout", "User logged out", {'username': username})

def log_data_upload(filename, file_size, shape=None, encoding=None):
    """Log data upload events"""
    data = {
        'filename': filename,
        'file_size_mb': round(file_size / (1024 * 1024), 2),
    }
    if shape:
        data['rows'] = shape[0]
        data['columns'] = shape[1]
    if encoding:
        data['encoding'] = encoding
    
    log_user_activity("data_upload", "Data file uploaded", data)

def log_demo_data_load():
    """Log demo dataset usage"""
    log_user_activity("demo_data_load", "Demo dataset loaded", {'dataset': 'personality_classification'})

def log_step_transition(from_step, to_step):
    """Log navigation between steps"""
    log_user_activity(
        "step_transition", 
        f"User navigated from step {from_step} to step {to_step}",
        {'from_step': from_step, 'to_step': to_step}
    )

def log_model_training(model_type, problem_type, dataset_shape, test_size, metrics=None):
    """Log model training events"""
    data = {
        'model_type': model_type,
        'problem_type': problem_type,
        'dataset_rows': dataset_shape[0],
        'dataset_columns': dataset_shape[1],
        'test_size': test_size
    }
    if metrics:
        data['metrics'] = metrics
    
    log_app_event("model_training", "Model training completed", data)

def log_model_error(model_type, problem_type, error):
    """Log model training errors"""
    log_error(
        "model_training_error",
        "Model training failed",
        error=error,
        data={'model_type': model_type, 'problem_type': problem_type}
    )

def log_download(download_type, filename):
    """Log file downloads"""
    log_user_activity(
        "download",
        f"File downloaded: {filename}",
        {'download_type': download_type, 'filename': filename}
    )

def get_log_stats():
    """Get basic statistics about logged events"""
    stats = {}
    
    for log_file in ['app.log', 'user_activity.log', 'performance.log', 'errors.log']:
        log_path = LOGS_DIR / log_file
        if log_path.exists():
            with open(log_path, 'r') as f:
                lines = f.readlines()
                stats[log_file] = {
                    'total_events': len(lines),
                    'file_size_mb': round(log_path.stat().st_size / (1024 * 1024), 2)
                }
        else:
            stats[log_file] = {'total_events': 0, 'file_size_mb': 0}
    
    return stats 