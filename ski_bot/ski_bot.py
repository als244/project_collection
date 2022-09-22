import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import datetime
from getpass import getpass
from selenium.webdriver.support.ui import Select
import random

# account stuff
email = input("Enter email: ")
password = getpass("Enter password: ")

# look at line 50 for the options
day = input("Enter the day this month you want to reserve: ")
# day of this current month
resort = input("Enter resort you would like to reserve at: ")

book_now = True

# connect to browser
driver = webdriver.Firefox()

#go to sign in page
driver.get("https://www.epicpass.com/account/login.aspx?url=%2faccount%2fmy-account.aspx%3fma_1%3d4")
time.sleep(4)


# enter log-in info
email_form = driver.find_element_by_id("txtUserName_3")
pass_form = driver.find_element_by_id("txtPassword_3")
driver.execute_script("arguments[0].setAttribute('value', arguments[1])", email_form, email)
driver.execute_script("arguments[0].setAttribute('value', arguments[1])", pass_form, password)
time.sleep(3)

#accept cookie
cookie_accept_btn = driver.find_element_by_id("onetrust-accept-btn-handler")
cookie_accept_btn.click()
time.sleep(1)

# submit login form
login_button = driver.find_elements_by_class_name("accountLogin__cta")
login_button[1].click()
time.sleep(2)


# reservation page
driver.get("https://www.epicpass.com/plan-your-trip/lift-access/reservations.aspx")
time.sleep(5)


resort_val_mapping = {"vail":"1", "beaver creek": "2", "breck": "3", "keystone": "4", "mt snow": "74", "stowe": "85", "okemo": "86", "attitash":"203"}

done = False

while not done:

	print("searching...")
	resort_select = Select(driver.find_element_by_id("PassHolderReservationComponent_Resort_Selection"))

	resort_val = resort_val_mapping[resort]
	resort_select.select_by_value(resort_val)
	time.sleep(1)

	search_avail_btn = driver.find_element_by_id("passHolderReservationsSearchButton")
	search_avail_btn.click()

	time.sleep(2)

	day_buttons = driver.find_elements_by_class_name("passholder_reservations__calendar__day")

	# iterating over all the days this month

	for b in day_buttons:
		# checking for specific date user wants to reserve
		if b.text == day:
			if b.is_enabled():
				print(resort + "IS available on the " + day)
				if book_now:
					# choose date
					time.sleep(3)
					b.click()
					time.sleep(3)

					# assign passholder
					passholder_checkboxes = driver.find_elements_by_class_name("passholder_reservations__assign_passholder_modal__label")
			
					# assume only 1 valid passholder
					# get the span element, then find first child that has label tag and click it
					assign_checkbox = passholder_checkboxes[0]
					labels = assign_checkbox.find_elements_by_tag_name("label")
					labels[0].click()

					time.sleep(2)
					# submit assignment
					primaryCTA_els = driver.find_elements_by_class_name("primaryCTA")
					for e in primaryCTA_els:
						print(e.text)
						if e.text == "ASSIGN PASS HOLDERS":
							e.click()
							time.sleep(5)
							break

					# accept terms
					accept_terms_checkbox = driver.find_elements_by_class_name("passholder_reservations__completion__terms_checkbox")
					accept_terms_checkbox[0].click()
					time.sleep(1)

					primaryCTA_els = driver.find_elements_by_class_name("primaryCTA")
					for e in primaryCTA_els:
						if e.text == "COMPLETE RESERVATION":
							e.click()
							time.sleep(5)
							exit()
					done = True
			else:
				print(resort + " IS NOT available on the " + day)
				print("sleeping...")
				time.sleep(random.randint(6, 30))
				print("refreshing...")
				driver.get(driver.current_url)
				driver.refresh()
				time.sleep(10)
				# exit iterating over buttons and go to start of while loop
				break


