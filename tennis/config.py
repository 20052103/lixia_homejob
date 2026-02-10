# Tennis Court Booking Configuration

# Website URL
BOOKING_URL = "https://bayclubconnect.com/racquet-sports/create-booking/a85fb33f-57b5-4283-9318-338530fe2fdd"

# Login credentials (set these or use environment variables)
# Option 1: Set here directly (less secure)
LOGIN_EMAIL = "62-6544217"  # or set to "your_email@example.com"
LOGIN_PASSWORD = "ZJXL8243304@sina"  # or set to "your_password"

# Option 2: Set via environment variables (more secure)
# Set in terminal: $env:TENNIS_EMAIL="your_email"; $env:TENNIS_PASSWORD="your_password"
# Then leave LOGIN_EMAIL and LOGIN_PASSWORD as None

# Auto-login settings
AUTO_LOGIN = True  # Set to False to manually log in
SKIP_TIME_WAIT = True  # Set to True if you're already ready on the website (don't wait for target time)

# Booking preferences
# TARGET_DATE options: "TODAY", "MO", "TU", "WE", "TH", "FR", "SA", "SU"
TARGET_DATE = "TH"  # Day of week (the website shows 7 days at a time)
TARGET_TIME = "8:00 am"  # Start time with am/pm (e.g., "8:00 am", "2:30 pm")
TARGET_COURT = None  # Set to specific court preference, or None for any available

# Browser settings
HEADLESS = False  # Set to True to run browser in headless mode

# Timing
REFRESH_INTERVAL = 0.25  # seconds between page refreshes (0.25s = 4 refreshes/sec, balanced speed)
POLLING_INTERVAL = 0.1  # deprecated - use REFRESH_INTERVAL instead
MAX_WAIT_TIME = 600  # max seconds to wait for availability (10 minutes)
