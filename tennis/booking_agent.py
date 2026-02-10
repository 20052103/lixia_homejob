"""
Tennis Court Booking Agent
Helps automatically book tennis courts on BayClubConnect
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TennisBookingAgent:
    def __init__(self, booking_url):
        self.booking_url = booking_url
        self.driver = None
        self.selected_date = None
        
    def initialize_browser(self, headless=False):
        """Initialize Edge webdriver"""
        options = webdriver.EdgeOptions()
        if headless:
            options.add_argument('--headless')
        
        options.add_argument('--start-maximized')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Edge(options=options)
        logger.info("Browser initialized")
    
    def navigate_to_booking_page(self):
        """Navigate to the booking page"""
        self.driver.get(self.booking_url)
        logger.info(f"Navigated to {self.booking_url}")
        time.sleep(2)  # Wait for page to load
    
    def auto_login(self, email, password):
        """Automatically log in with credentials"""
        try:
            logger.info("🔐 Attempting auto-login...")
            
            # Wait for login form to appear
            time.sleep(2)
            
            # Find email/username field
            email_inputs = self.driver.find_elements(By.XPATH, "//input[@type='email' or @type='text' or @name='email' or @name='username']")
            if not email_inputs:
                logger.warning("Could not find email input field")
                return False
            
            email_field = email_inputs[0]
            email_field.click()
            email_field.clear()
            email_field.send_keys(email)
            logger.info("✅ Entered email")
            
            # Find password field
            password_inputs = self.driver.find_elements(By.XPATH, "//input[@type='password']")
            if not password_inputs:
                logger.warning("Could not find password input field")
                return False
            
            password_field = password_inputs[0]
            password_field.click()
            password_field.clear()
            password_field.send_keys(password)
            logger.info("✅ Entered password")
            
            # Find and click login button
            login_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Login') or contains(text(), 'Sign In') or contains(text(), 'log in')]")
            if login_buttons:
                login_buttons[0].click()
                logger.info("✅ Clicked login button")
                time.sleep(3)  # Wait for login to complete
                return True
            else:
                logger.warning("Could not find login button")
                return False
                
        except Exception as e:
            logger.error(f"Auto-login failed: {e}")
            return False
    
    def wait_for_user_ready(self):
        """Wait for user to navigate to the booking page and confirm they're ready"""
        logger.info("\n" + "="*60)
        logger.info("📍 WAITING FOR YOU TO NAVIGATE TO THE BOOKING PAGE")
        logger.info("="*60)
        logger.info("Please:")
        logger.info("  1. Log in to the website")
        logger.info("  2. Navigate to the tennis court booking page")
        logger.info("  3. Make sure you can see the date/time selector")
        logger.info("="*60)
        
        input("\n✅ Once you're on the correct booking page, press ENTER to continue: ")
        logger.info("✅ Ready! Starting the booking monitor...")
        return True
    
    def check_target_time_status(self, target_time):
        """
        Check the status of target time in available slots
        Returns: ('available', element), ('locked', element), or ('not_found', None)
        """
        slots = self.find_available_time_slots()
        
        for slot in slots:
            slot_text = slot["time"]
            # Extract start time from slot (e.g., "5:30" from "5:30 - 7:00 pm")
            start_time_str = slot_text.split("-")[0].strip()
            
            # Match the target time with the start time
            # E.g., "5:30 pm" should match the "5:30" part of "5:30 - 7:00 pm"
            if target_time.lower() in (start_time_str + " pm").lower() or \
               target_time.lower() in (start_time_str + " am").lower() or \
               target_time.lower() in slot_text.lower():
                if slot["disabled"]:
                    return ('locked', slot["element"])
                else:
                    return ('available', slot["element"])
        
        return ('not_found', None)
    
    
    def select_date(self, target_day):
        """
        Select date by day of week
        target_day: "TODAY", "MO", "TU", "WE", "TH", "FR", "SA", "SU"
        """
        try:
            # Find date selector divs with class 'slider-item'
            date_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.slider-item.clickable")
            
            logger.info(f"Found {len(date_elements)} date elements")
            
            for elem in date_elements:
                text = elem.text.strip().upper()
                logger.info(f"  Checking date element: {text}")
                
                if target_day.upper() in text:
                    logger.info(f"✅ Clicking date: {text}")
                    self.selected_date = text  # Store selected date
                    elem.click()
                    time.sleep(2)  # Wait for time slots to load
                    return True
            
            logger.error(f"Could not find date button for {target_day}")
            logger.info(f"Available dates: {[elem.text.strip() for elem in date_elements]}")
            return False
            
        except Exception as e:
            logger.error(f"Error selecting date: {e}")
            return False
    
    def find_available_time_slots(self):
        """
        Get all available time slots currently displayed (MORNING, AFTERNOON, EVENING)
        Returns list of time slot elements with lock status
        """
        try:
            # Find all time slot divs - they may have class 'time-slot' or similar
            # Also check for lock icons to determine availability
            slots = self.driver.find_elements(By.CSS_SELECTOR, "div.time-slot, div[class*='time'], button[class*='time']")
            
            available = []
            for slot in slots:
                time_text = slot.text.strip()
                if not time_text:  # Skip empty elements
                    continue
                
                # Check multiple indicators of locked status
                slot_class = slot.get_attribute("class") or ""
                slot_html = slot.get_attribute("innerHTML") or ""
                
                # Check if disabled
                is_disabled = any([
                    "disabled" in slot_class.lower(),
                    "lock" in slot_html.lower(),
                    slot.get_attribute("disabled") is not None,
                    "opacity" in slot_class.lower()  # Sometimes locked slots have opacity classes
                ])
                
                # Check for lock icon inside the slot
                try:
                    lock_icon = slot.find_elements(By.XPATH, ".//*[contains(@class, 'lock') or contains(@class, 'icon-lock')]")
                    if lock_icon:
                        is_disabled = True
                except:
                    pass
                
                available.append({
                    "time": time_text,
                    "element": slot,
                    "disabled": is_disabled,
                    "class": slot_class
                })
            
            # Categorize by actual time of day (am/pm)
            morning = []   # 6am - 9am
            afternoon = [] # 2pm - 5pm
            evening = []   # 6pm - 11pm
            
            for s in available:
                time_lower = s["time"].lower()
                
                if "am" in time_lower:
                    morning.append(s)
                elif "pm" in time_lower:
                    # Further split pm times
                    # Extract hour (first number before colon or dash)
                    import re
                    hours = re.findall(r'(\d+):', time_lower)
                    if hours:
                        first_hour = int(hours[0])
                        if 2 <= first_hour <= 5:  # 2pm-5pm = afternoon
                            afternoon.append(s)
                        elif 6 <= first_hour <= 11:  # 6pm-11pm = evening
                            evening.append(s)
            
            available_count = sum(1 for s in available if not s['disabled'])
            disabled_count = sum(1 for s in available if s['disabled'])
            
            logger.info(f"\n📋 TIME SLOTS FOUND: {len(available)} total")
            logger.info(f"   ✅ Available: {available_count} | 🔒 Locked: {disabled_count}")
            
            if morning:
                morning_available = len([s for s in morning if not s['disabled']])
                logger.info(f"   🌅 MORNING ({morning_available} available):")
                for s in morning:
                    status = "✅" if not s['disabled'] else "🔒"
                    logger.info(f"      {status} {s['time']}")
            
            if afternoon:
                afternoon_available = len([s for s in afternoon if not s['disabled']])
                logger.info(f"   ☀️  AFTERNOON ({afternoon_available} available):")
                for s in afternoon:
                    status = "✅" if not s['disabled'] else "🔒"
                    logger.info(f"      {status} {s['time']}")
            
            if evening:
                evening_available = len([s for s in evening if not s['disabled']])
                logger.info(f"   🌙 EVENING ({evening_available} available):")
                for s in evening:
                    status = "✅" if not s['disabled'] else "🔒"
                    logger.info(f"      {status} {s['time']}")
            
            return available
        except Exception as e:
            logger.error(f"Error fetching time slots: {e}")
            return []
    
    def book_slot(self, target_day, target_time, skip_wait=False):
        """
        Click the time slot immediately when available and click next button
        User will handle the rest (payment, final confirmation, etc.)
        
        Only starts checking when:
        - skip_wait=True: Start immediately (you're already ready)
        - skip_wait=False: Wait until current time is within 1 minute of target time
        
        Args:
            target_day: Day of week ("TODAY", "MO", "TU", "WE", "TH", "FR", "SA", "SU")
            target_time: Time string with am/pm (e.g., "8:00 am", "2:30 pm")
            skip_wait: If True, skip time waiting and start polling immediately
        """
        from datetime import datetime
        from config import POLLING_INTERVAL, MAX_WAIT_TIME
        
        # Parse target time
        try:
            target_dt = datetime.strptime(target_time, "%I:%M %p")
            target_time_obj = target_dt.time()
        except:
            logger.error(f"Invalid time format: {target_time}. Use format: '8:00 am' or '2:30 pm'")
            return False
        
        logger.info(f"🎾 Target time: {target_time}")
        
        # Wait for polling window if not skipping
        if not skip_wait:
            logger.info("⏳ Waiting until 10 seconds before target time to start refreshing...")
            
            while True:
                now = datetime.now().time()
                time_diff = self._time_diff_minutes(now, target_time_obj)
                
                # Only show time diff when >= 1 minute or <= 15 seconds
                if time_diff >= 1 or time_diff <= 0.25:  # 15 seconds
                    logger.info(f"   Current time: {now.strftime('%H:%M:%S')} | Target: {target_time} | Time until: {time_diff*60:.1f} seconds")
                
                # Start refreshing when we're 10 seconds before target time
                if time_diff <= (10/60):  # Within 10 seconds
                    logger.info(f"🚀 10 SECONDS BEFORE TARGET! Starting continuous page refresh NOW!")
                    break
                
                if time_diff > 0.25:  # More than 15 seconds
                    wait_seconds = (time_diff * 60) - 10  # Wait until 10 seconds before
                    time.sleep(min(wait_seconds, 5))  # Check every 5 seconds max
                else:
                    time.sleep(0.1)  # Check every 100ms in final seconds
        else:
            logger.info("⏭️ Skipping time wait - starting continuous page refresh immediately!")
        
        logger.info(f"🎾 Searching for booking: {target_day} at {target_time}")
        logger.info("🚀 Rapid polling active - will click slot immediately when available...")
        
        # Step 1: Select the date
        if not self.select_date(target_day):
            logger.error("Failed to select date")
            return False
        
        # Step 2: Check initial status of target time
        logger.info(f"\n⏳ Checking target time '{target_time}' on {self.selected_date}...")
        status, slot_element = self.check_target_time_status(target_time)
        
        if status == 'not_found':
            logger.error(f"\n{'='*60}")
            logger.error(f"❌ TIME SLOT NOT FOUND!")
            logger.error(f"'{target_time}' does not exist in the schedule")
            logger.error(f"{'='*60}")
            logger.error(f"\n📍 Time slots on {self.selected_date}:")
            slots = self.find_available_time_slots()
            available_times = [s["time"] for s in slots if not s["disabled"]]
            locked_times = [s["time"] for s in slots if s["disabled"]]
            
            if available_times:
                logger.error(f"\n✅ AVAILABLE ({len(available_times)}):")
                for t in available_times:
                    logger.error(f"   • {t}")
            else:
                logger.error(f"\n✅ AVAILABLE: None")
            
            if locked_times:
                logger.error(f"\n🔒 LOCKED ({len(locked_times)}):")
                for t in locked_times:
                    logger.error(f"   • {t}")
            else:
                logger.error(f"\n🔒 LOCKED: None")
            
            logger.error(f"\n{'='*60}")
            logger.error(f"Please update TARGET_TIME in config.py to one of the above times")
            logger.error(f"{'='*60}\n")
            return False
        
        elif status == 'available':
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ '{target_time}' is AVAILABLE RIGHT NOW!")
            logger.info(f"{'='*60}")
            logger.info(f"Clicking immediately...")
            self._click_slot_and_next(slot_element, target_time)
            return True
        
        elif status == 'locked':
            logger.info(f"\n{'='*60}")
            logger.info(f"🔒 '{target_time}' is LOCKED")
            logger.info(f"Waiting for it to open...")
            logger.info(f"{'='*60}\n")
            self._wait_for_locked_slot_and_click(target_time, target_time_obj)
            return True
        
        return False
    
    def _click_slot_and_next(self, slot_element, target_time):
        """Click the slot and NEXT button, then wait for user"""
        try:
            slot_element.click()
            time.sleep(0.2)
            
            # Find and click NEXT button
            next_btns = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'NEXT')]")
            if not next_btns:
                next_btns = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Next')]")
            if not next_btns:
                next_btns = self.driver.find_elements(By.XPATH, "//*[contains(@class, 'btn') and contains(text(), 'NEXT')]")
            
            if next_btns:
                logger.info("✅ Clicked NEXT button!")
                next_btns[0].click()
                time.sleep(0.5)
            else:
                logger.warning("⚠️ Could not find NEXT button")
            
            logger.info("\n" + "="*60)
            logger.info("📍 YOUR TURN NOW!")
            logger.info("="*60)
            logger.info("Please complete:")
            logger.info("  - Select the court")
            logger.info("  - Enter member details")
            logger.info("  - Complete payment")
            logger.info("  - Finalize booking")
            logger.info("="*60 + "\n")
            
            input("➜ Press ENTER when booking is complete: ")
            logger.info("✅ Booking completed!")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            input("Press ENTER to continue: ")
    
    def _wait_for_locked_slot_and_click(self, target_time, target_time_obj):
        """Wait for locked slot to become available and click it"""
        from config import REFRESH_INTERVAL
        
        # Wait until 1 minute before target time
        logger.info("⏳ Waiting until 1 minute before target time to start monitoring...")
        
        while True:
            now = datetime.now().time()
            time_diff = self._time_diff_minutes(now, target_time_obj)
            
            if time_diff >= 1 or time_diff <= 0.25:
                logger.info(f"   Current: {now.strftime('%H:%M:%S')} | Target: {target_time} | Countdown: {time_diff*60:.1f}s")
            
            if time_diff <= (60/60):  # 1 minute = 60/60 minutes
                logger.info(f"\n⏱️  1 MINUTE! Starting countdown...\n")
                break
            
            if time_diff > 0.25:
                sleep_time = min((time_diff * 60) - 60, 5)
                time.sleep(sleep_time)
            else:
                time.sleep(0.1)
        
        # Now wait until 10 seconds before target time
        while True:
            now = datetime.now().time()
            time_diff = self._time_diff_minutes(now, target_time_obj)
            
            logger.info(f"   {time_diff*60:.1f}s remaining...")
            
            if time_diff <= (10/60):  # 10 seconds
                logger.info(f"\n🚀 10 SECONDS! Starting rapid refresh and polling...\n")
                break
            
            time.sleep(1)  # Check every second during the minute countdown
        
        # Final 10 seconds - rapid polling with view switching
        start_time = time.time()
        poll_count = 0
        
        while time.time() - start_time < 15:  # Poll for up to 15 seconds after target
            poll_count += 1
            elapsed = time.time() - start_time
            
            status, slot_element = self.check_target_time_status(target_time)
            
            if status == 'available':
                logger.info(f"\n✅ FOUND! '{target_time}' is now AVAILABLE!\n")
                self._click_slot_and_next(slot_element, target_time)
                return
            
            # Refresh by switching views
            if poll_count % 2 == 0:
                logger.info(f"🔄 Poll #{poll_count} ({elapsed:.1f}s) - Refreshing...")
                self.refresh_slots_by_switching_view()
            else:
                logger.info(f"⏳ Poll #{poll_count} ({elapsed:.1f}s) - Checking...")
            
            time.sleep(REFRESH_INTERVAL)
        
        logger.error(f"❌ Timeout! Slot never became available")
        input("Press ENTER to continue: ")
    
    def _time_diff_minutes(self, time1, time2):
        """Calculate difference between two time objects in minutes (time1 - time2)"""
        from datetime import datetime, timedelta
        
        # Convert to datetime for easier calculation
        today = datetime.now().date()
        dt1 = datetime.combine(today, time1)
        dt2 = datetime.combine(today, time2)
        
        diff = dt1 - dt2
        return diff.total_seconds() / 60
    
    def refresh_slots_by_switching_view(self):
        """
        Refresh available slots by clicking COURT VIEW then back to HOUR VIEW
        This avoids page going back (unlike driver.refresh())
        """
        try:
            # Find and click COURT VIEW button - try multiple selectors
            court_view_btns = self.driver.find_elements(By.XPATH, "//div[contains(text(), 'COURT VIEW')]")
            if not court_view_btns:
                court_view_btns = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'COURT VIEW') or contains(text(), 'Court View')]")
            if not court_view_btns:
                court_view_btns = self.driver.find_elements(By.CSS_SELECTOR, "[class*='view'], button[class*='court']")
            
            if court_view_btns:
                logger.info("🔄 Switching to COURT VIEW...")
                court_view_btns[0].click()
                time.sleep(0.5)  # Wait for view to switch
                
                # Find and click HOUR VIEW button to return
                hour_view_btns = self.driver.find_elements(By.XPATH, "//div[contains(text(), 'HOUR VIEW')]")
                if not hour_view_btns:
                    hour_view_btns = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'HOUR VIEW') or contains(text(), 'Hour View')]")
                if not hour_view_btns:
                    hour_view_btns = self.driver.find_elements(By.CSS_SELECTOR, "[class*='view'], button[class*='hour']")
                
                if hour_view_btns:
                    logger.info("🔄 Switching back to HOUR VIEW...")
                    hour_view_btns[0].click()
                    time.sleep(0.5)  # Wait for view to switch back
                    return True
                else:
                    logger.warning("⚠️ Could not find HOUR VIEW button - trying page refresh instead")
                    self.driver.refresh()
                    time.sleep(0.5)
                    return False
            else:
                logger.warning("⚠️ Could not find COURT VIEW button - trying page refresh instead")
                self.driver.refresh()
                time.sleep(0.5)
                return False
        except Exception as e:
            logger.error(f"Error switching views: {e}")
            logger.info("Falling back to page refresh...")
            try:
                self.driver.refresh()
                time.sleep(0.5)
                return False
            except:
                return False
    
    
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


def main():
    import os
    from config import BOOKING_URL, TARGET_DATE, TARGET_TIME, HEADLESS, AUTO_LOGIN, LOGIN_EMAIL, LOGIN_PASSWORD, SKIP_TIME_WAIT
    
    agent = TennisBookingAgent(BOOKING_URL)
    
    try:
        agent.initialize_browser(headless=HEADLESS)
        agent.navigate_to_booking_page()
        
        # Handle login
        if AUTO_LOGIN:
            # Try to get credentials from config or environment variables
            email = LOGIN_EMAIL or os.getenv('TENNIS_EMAIL')
            password = LOGIN_PASSWORD or os.getenv('TENNIS_PASSWORD')
            
            if email and password:
                logger.info("🔐 Auto-login enabled")
                if agent.auto_login(email, password):
                    logger.info("✅ Auto-login successful!")
                    time.sleep(2)
                else:
                    logger.warning("⚠️ Auto-login failed, please log in manually")
                    agent.wait_for_user_ready()
            else:
                logger.warning("⚠️ No credentials found. Set TENNIS_EMAIL and TENNIS_PASSWORD environment variables or config.LOGIN_EMAIL/PASSWORD")
                agent.wait_for_user_ready()
        else:
            # Manual login
            agent.wait_for_user_ready()
        
        # Start booking
        if TARGET_DATE and TARGET_TIME:
            agent.book_slot(TARGET_DATE, TARGET_TIME, skip_wait=SKIP_TIME_WAIT)
        else:
            logger.warning("TARGET_DATE and TARGET_TIME not set in config.py")
        
        # Don't close the browser - let user complete the booking
        logger.info("\n✅ Browser will stay open for you to complete the booking")
        input("Press ENTER in terminal when booking is complete: ")
        
    except KeyboardInterrupt:
        logger.info("\n✅ Booking process completed!")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
