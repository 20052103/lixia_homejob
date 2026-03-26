import sys
sys.path.insert(0, 'agent')
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Simulate calendar timezone = America/Los_Angeles
cal_tz = ZoneInfo("America/Los_Angeles")

# Import and test resolve_range
from gcal_tool import resolve_range, _week_bounds

start_dt, end_dt = resolve_range("this_week", cal_tz=cal_tz)
print(f"this_week: {start_dt.isoformat()} → {end_dt.isoformat()}")
print(f"  start weekday: {start_dt.strftime('%A')}")
print(f"  end weekday:   {end_dt.strftime('%A')}")

start_dt, end_dt = resolve_range("today", cal_tz=cal_tz)
print(f"today: {start_dt.isoformat()} → {end_dt.isoformat()}")

start_dt, end_dt = resolve_range("next_7_days", cal_tz=cal_tz)
print(f"next_7_days: {start_dt.isoformat()} → {end_dt.isoformat()}")
