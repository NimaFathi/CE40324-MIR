import json
import re
import time

from bs4 import BeautifulSoup
from selenium import webdriver


class Crawler:
    def __init__(self, limit=5000, wait=2):
        self.driver = None
        self.reload_driver()
        self.crawl_limit = limit
        self.max_wait_time = wait
        self.queue = []
        self.explored = set()

        with open('./start.txt') as file:
            init_urls = file.readlines()

        for url in init_urls:
            paper_id = self.search_paper(pattern=r'https://academic.microsoft.com/paper/(\d+)', string=url)
            if paper_id is not None:
                self.queue.append(paper_id)

    def reload_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument('--user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) ' +
                             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"')
        options.add_argument(r"user-data-dir=.\cookies\\")
        self.driver = webdriver.Chrome(options=options)

    def crawl(self):
        with open('papers.json', 'a') as file:
            file.write("[\n")
        index = 0
        add_to_queue = True
        while index < self.crawl_limit:
            paper_id = self.queue.pop(0)
            if paper_id in self.explored:
                continue

            paper = self.get_paper(paper_id)
            if paper is None:
                print('Reload driver ... Paper: {}/{} - Paper_id: {}'.format(index, self.crawl_limit, paper_id))
                self.driver.quit()
                self.reload_driver()
                self.queue.insert(0, paper_id)
                continue

            self.explored.add(paper_id)
            index += 1
            if index % 100 == 0:
                print('Paper: {}/{} - Paper_id: {}'.format(index, self.crawl_limit, paper_id))
            if add_to_queue:
                self.queue.extend(paper['references'])
                add_to_queue = True if len(set(self.queue + list(self.explored))) <= self.crawl_limit else False
        with open('papers.json', 'a') as file:
            file.write("]")  # create a file that contains multiple json objects
        self.driver.quit()

    def delayed(self):
        now = time.time()
        loaded = False
        while not loaded and time.time() - now < self.max_wait_time:
            time.sleep(0.5)
            loaded = '<div class="primary_paper">' in self.driver.page_source
        return loaded

    def get_html(self, paper_id):
        self.driver.get('https://academic.microsoft.com/paper/' + paper_id)
        self.delayed()
        if self.delayed():
            return BeautifulSoup(self.driver.page_source, features='html.parser')
        else:
            return None

    def get_paper(self, paper_id):
        parsed_html = self.get_html(paper_id)
        if parsed_html is None:
            return None
        paper_json = self.make_json_object(parsed_html, paper_id)
        with open('papers.json', 'a') as file:
            json.dump(paper_json, file, indent=2)
            file.write(", ")
        return paper_json

    def make_json_object(self, html, paper_id):
        title = html.head.find('title').text
        abstract = html.body.find('p', attrs={'class': None}).text
        date = html.body.find('span', attrs={'class': 'year'}).text
        authors = [
            a_tag.text for a_tag in
            html.find('div', attrs={'class': 'authors'}).find_all('a', attrs={'class': 'au-target author link'})
        ]
        return {
            'id': paper_id,
            'title': title,
            'abstract': abstract,
            'date': date,
            'authors': authors,
            'references': self.get_references(html)
        }

    def get_references(self, parsed_html):
        ref_ids = []
        processed_refs = 0
        primary_ref_lst = parsed_html.find_all('div', attrs={'class': 'primary_paper'})
        for ref in primary_ref_lst:
            if processed_refs >= 10:
                break
            ref = ref.find('a', attrs={'class': 'title au-target'})
            if ref is None or ref.get('href', None) is None:
                continue
            ref_id = self.search_paper(r'paper/(\d+)(/reference)?', ref['href'])
            if ref_id is not None:
                ref_ids.append(ref_id)
                processed_refs += 1

        return ref_ids

    @staticmethod
    def search_paper(pattern, string):
        found = re.search(pattern, string)
        if found:
            return found.group(1)
        return None


if __name__ == '__main__':
    crawler = Crawler(limit=500)
    crawler.crawl()
