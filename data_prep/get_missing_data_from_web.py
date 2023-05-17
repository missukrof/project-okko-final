import os

from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import ui
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from configs.config import settings
from utils.utils import (
    save_pickle,
    load_pickle
)

def find_missing_titles(missing_ids_list):
    
    parent_directory = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

    if len(os.listdir(f"{parent_directory}{settings.NAN_MAPPING_PATH.FOLDER}")) > 0:
        missing_ids_names_mapping = load_pickle(
            path=f"{parent_directory}{settings.NAN_MAPPING_PATH.FOLDER}{settings.NAN_MAPPING_PATH.FILE_NAME}"
        )
    else:
        missing_ids_names_mapping = {}

        chrome_options = ChromeOptions()
        for option in settings.DRIVER.CHROME_OPTIONS:
            chrome_options.add_argument(option)
        chrome_options.add_argument(f'user-agent={settings.DRIVER.USER_AGENT}')

        for movie_id in missing_ids_list:

            URL = f'{settings.WEBSITE.URL_PATTERN}{movie_id}'
            DRIVER_PATH = settings.DRIVER.DRIVER_PATH

            browser = Chrome(executable_path=DRIVER_PATH, chrome_options=chrome_options)
            browser.get(URL)
            
            try:
                element = ui.WebDriverWait(browser, 2)\
                    .until(EC.visibility_of_all_elements_located((By.CLASS_NAME, settings.WEBSITE.CLASS_TEXT)))[0]\
                        .text
            except TimeoutException:
                element = ui.WebDriverWait(browser, 20)\
                    .until(EC.visibility_of_element_located((By.CLASS_NAME, settings.WEBSITE.CLASS_IMG)))\
                        .get_attribute('alt')

            missing_ids_names_mapping[movie_id] = element
            
            browser.quit()
        
        save_pickle(
            data=missing_ids_names_mapping, 
            path=f"{settings.NAN_MAPPING_PATH.FOLDER}{settings.NAN_MAPPING_PATH.FILE_NAME}"
            )

    return missing_ids_names_mapping