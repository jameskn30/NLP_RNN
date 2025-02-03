import scrapy
import os
import logging
from pprint import pprint
import sys

class YcScrapySpider(scrapy.Spider):
    name = "yc_scrapy"
    allowed_domains = ["ycombinator.com"]
    
    def __init__(self, urls=None, *args, **kwargs):
        super(scrapy.Spider, self).__init__(*args, **kwargs)

        self.start_urls = []
        print('cwd ', os.getcwd())

        for html_file in os.listdir("output"):
            yc, season, year = html_file.split("_")
            year = year.split('.')[0]
        
            file_path = os.path.join("output", html_file)
            logging.info('loading seed links from ' + file_path)

            if os.path.exists(file_path) == False:
                logging.error(f'path not found: {file_path}')
                raise AssertionError(f'path not found: {file_path}')

            with open(file_path, 'r') as file:
                links = [line.strip() for line in file.readlines()]
                self.start_urls.append({
                    'season': season,
                    'year': year,
                    'links': links
                })
                logging.info(f"link count = {len(self.start_urls)}")
    
    def start_requests(self):
        for i, data in enumerate(self.start_urls):
            self.season = data['season']
            self.year = data['year']
            links = data['links']
            for url in links:
                logging.info(f"{i}, scraping link {url}")
                yield scrapy.Request(url = url, callback = self.parse)

    def parse(self, response):
        #put parsing the company page here
        logging.info('start parsing ' + response.url)

        #company summary
        card_div = response.xpath("//div[@class='ycdc-card space-y-1.5 sm:w-[300px]']")
        if card_div:
            founded_year = card_div.xpath(".//span[contains(., 'Founded:')]/following-sibling::span/text()").get()
            team_size = card_div.xpath(".//span[contains(., 'Team Size:')]/following-sibling::span/text()").get()
            location = card_div.xpath(".//span[contains(., 'Location:')]/following-sibling::span/text()").get()
            group_partner = card_div.xpath(".//span[contains(., 'Group Partner:')]/following-sibling::a/text()").get()  # Check for link first
            group_partner_link = card_div.xpath(".//span[contains(., 'Group Partner:')]/following-sibling::a/@href").get()

            # logging.info("founded year " + founded_year)
            # logging.info("team size " + team_size)
            # logging.info("location " + location)
            # logging.info("group_partner " + group_partner)
            # logging.info("group_partner link " + group_partner_link)
        else:
            logging.error(f'url error, {response.url}')
            return
        
        # get the description
        description_div = response.xpath("//div[@class='prose max-w-full whitespace-pre-line']")

        if description_div:
            description = description_div.xpath(".//text()").getall()  # Get all text content within the div
            description = "".join(description).strip()
            # logging.info('description = ' + description)
        else:
            logging.error(f'url error, {response.url}')
            return

        founder_section = response.xpath("//section[@class='relative isolate z-0 border-retro-sectionBorder sm:pr-[13px] ycdcPlus:pr-0 pt-1 sm:pt-2 lg:pt-3 pb-1 sm:pb-2 lg:pb-3']")

        if founder_section:
            founders = []  # List to store all founders' data

            founder_divs = founder_section.xpath(".//div[@class='flex flex-row flex-col items-start gap-3 md:flex-row']")  # Select all individual founder divs

            for founder_div in founder_divs:
                name = founder_div.xpath(".//h3[@class='text-lg font-bold']/text()").get().strip()
                background = founder_div.xpath(".//div[@class='prose max-w-full whitespace-pre-line']/text()").getall()
                background = "".join(background).strip()

                founder_data = {
                    "name": name,
                    "background": background,
                }
                founders.append(founder_data)
                # logging.info(f"founder name = {name}, background = {background}")

        else:
            logging.error(f'url error, {response.url}')
            return

        #Get the linkedin link

        founder_social_section = response.xpath("//section[@class='relative isolate z-0 border-retro-sectionBorder sm:pr-[13px] ycdcPlus:pr-0 pt-1 sm:pt-2 lg:pt-3 pb-1 sm:pb-2 lg:pb-3']")

        if founder_social_section:
            founders_socials = []

            founder_social_divs = founder_section.xpath(".//div[@class='flex flex-row flex-col items-start gap-3 md:flex-row']")

            for founder_social_div in founder_social_divs:
                name = founder_social_div.xpath(".//h3[@class='text-lg font-bold']/text()").get().strip()
                linkedin_url = founder_social_div.xpath(".//a[contains(@href, 'linkedin.com')]/@href").get() #Gets the href of the link that contains linkedin.com

                founder_social_data = {
                    "name": name,
                    "linkedin_url": linkedin_url,
                }
                founders_socials.append(founder_social_data)

                logging.info('founder social data ')
                logging.info(founder_social_data)

        else:
            logging.error(f'url error, {response.url}')
            return

        item = {
            'url':  response.url,
            "year": founded_year,
            "team_size": team_size,
            'location': location,
            "group_partner":group_partner,
            "group_partner link ": group_partner_link,
            'description': description,
            'founders': founders,
            'founders_socials': founders_socials,
            'season': self.season,
            'yc_year': self.year,
        }

        # logging.info('item ', item)
        pprint(item)

        logging.info('OK, parsed' + response.url)
        logging.info('####')

        yield item
    
