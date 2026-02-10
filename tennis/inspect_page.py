"""
Helper script to inspect the website and find correct CSS selectors
Run this to explore the page structure
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def inspect_page(url):
    """Open browser and let you inspect the page structure"""
    options = webdriver.EdgeOptions()
    options.add_argument('--start-maximized')
    driver = webdriver.Edge(options=options)
    
    driver.get(url)
    print("✅ Browser opened - log in manually and navigate to the booking page")
    print("⏳ Waiting 10 minutes for you to be ready...")
    
    input("Press Enter once you're logged in and on the booking page: ")
    
    # Inspect date selectors
    print("\n" + "="*50)
    print("INSPECTING DATE SELECTORS")
    print("="*50)
    
    # Try different selectors for dates
    selectors = [
        ("div[class*='date']", "div with 'date' in class"),
        ("button[class*='date']", "button with 'date' in class"),
        (".date-selector", ".date-selector"),
        ("[class*='TODAY']", "elements with 'TODAY' in class"),
        ("div[class*='day-']", "div with 'day-' in class"),
    ]
    
    for selector, desc in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                print(f"\n✅ Found {len(elements)} elements with: {desc}")
                for i, elem in enumerate(elements[:3]):  # Show first 3
                    print(f"   [{i}] Text: '{elem.text}' | Tag: {elem.tag_name} | Class: {elem.get_attribute('class')}")
        except:
            pass
    
    # Inspect time slot selectors
    print("\n" + "="*50)
    print("INSPECTING TIME SLOT SELECTORS")
    print("="*50)
    
    selectors = [
        ("button", "all buttons"),
        ("div[class*='time']", "div with 'time' in class"),
        ("button[class*='slot']", "button with 'slot' in class"),
        ("[class*='morning'], [class*='afternoon'], [class*='evening']", "morning/afternoon/evening"),
    ]
    
    for selector, desc in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                print(f"\n✅ Found {len(elements)} elements with: {desc}")
                for i, elem in enumerate(elements[:5]):  # Show first 5
                    text = elem.text[:50] if elem.text else "(no text)"
                    print(f"   [{i}] Text: '{text}' | Class: {elem.get_attribute('class')}")
        except:
            pass
    
    # Get page HTML
    print("\n" + "="*50)
    print("PAGE SOURCE (first 3000 chars)")
    print("="*50)
    print(driver.page_source[:3000])
    
    input("\nPress Enter to close browser: ")
    driver.quit()

if __name__ == "__main__":
    from config import BOOKING_URL
    inspect_page(BOOKING_URL)
