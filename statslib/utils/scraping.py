import urllib
from bs4 import BeautifulSoup
import pandas as pd

from statslib.utils.log import Logger


def parce_web_table_to_df(ths_url, ths_table_number=0):
    logger = Logger().get_logger()
    if not ths_url:
        logger.info("Please provide the url!")
        return
    try:
        n = ths_table_number
        openHtml = urllib.request.urlopen(ths_url).read()
        soup = BeautifulSoup(openHtml, 'lxml')  # Parse the HTML as string
        logger.debug("Opened HTML: {}".format(ths_url))
        table = soup.find_all('table')[n]  # Grab the fist table!
        field_names = []
        row = table.find_all('tr')[0]
        columns = row.find_all('th')
        for column in columns:
            field_names.append(column.get_text())

        all_entries = []
        for row in table.find_all('tr')[1:]:
            symbol_string = []
            columns = row.find_all('td')
            for column in columns:
                row_value = column.get_text()
                if '\n' in row_value:
                    row_value = row_value.replace('\n', '')
                symbol_string.append(row_value)
            all_entries.append(symbol_string)

        ths_parsed_dataframe = pd.DataFrame(all_entries, columns=field_names)
        return ths_parsed_dataframe
    except Exception as e:
        print("Can't parce the url{}! "
              "\nError message in parsing: {}".format(ths_url, e.args))


def save_file_from_web(url, file_path):
    urllib.request.urlretrieve(url, file_path)
