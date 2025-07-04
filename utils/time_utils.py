# utils/time_utils.py
import pytz
from datetime import datetime

class TimeUtils:
    IST = pytz.timezone('Asia/Kolkata')
    
    @classmethod
    def now_ist(cls):
        return datetime.now(cls.IST)
    
    @classmethod
    def to_ist(cls, dt):
        if dt.tzinfo is None:
            return cls.IST.localize(dt)
        return dt.astimezone(cls.IST)