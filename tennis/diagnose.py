"""
Diagnostic script to inspect the actual page structure
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json

def diagnose_page():
    """Open browser and inspect page structure"""
    options = webdriver.EdgeOptions()
    options.add_argument('--start-maximized')
    driver = webdriver.Edge(options=options)
    
    from config import BOOKING_URL
    driver.get(BOOKING_URL)
    
    print("✅ Browser opened - please log in and navigate to booking page")
    print("   Then press ENTER in the terminal...")
    
    input("\nPress ENTER once you're on the booking page: ")
    
    time.sleep(2)
    
    # Get page source
    print("\n" + "="*60)
    print("PAGE HTML (first 5000 chars):")
    print("="*60)
    html = driver.page_source
    print(html[:5000])
    
    # Find all elements with text
    print("\n" + "="*60)
    print("ALL TEXT ELEMENTS ON PAGE:")
    print("="*60)
    
    all_elements = driver.find_elements(By.XPATH, "//*[text()]")
    for i, elem in enumerate(all_elements[:30]):
        text = elem.text.strip()[:60]
        if text:
            print(f"{i:2d}. [{elem.tag_name}] {text}")
    
    # Find all buttons
    print("\n" + "="*60)
    print("ALL BUTTONS ON PAGE:")
    print("="*60)
    
    buttons = driver.find_elements(By.TAG_NAME, "button")
    for i, btn in enumerate(buttons):
        text = btn.text.strip()[:60]
        classes = btn.get_attribute("class")
        print(f"{i:2d}. TEXT: '{text}' | CLASS: {classes}")
    
    # Find all divs with 'date' or 'day' in class/id
    print("\n" + "="*60)
    print("DIVS WITH 'date', 'day', 'time' IN CLASS/ID:")
    print("="*60)
    
    date_divs = driver.find_elements(By.XPATH, "//*[@class or @id][contains(translate(@class|@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'date') or contains(translate(@class|@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'day') or contains(translate(@class|@id, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'time')]")
    
    for i, elem in enumerate(date_divs[:20]):
        text = elem.text.strip()[:40]
        tag = elem.tag_name
        classes = elem.get_attribute("class")
        elem_id = elem.get_attribute("id")
        print(f"{i:2d}. [{tag}] class='{classes}' id='{elem_id}' text='{text}'")
    
    input("\nPress ENTER to close browser: ")
    driver.quit()

if __name__ == "__main__":
    diagnose_page()
