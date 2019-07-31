import logging
import sys
import base64
from datetime import date
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler
from icrawler import ImageDownloader
from icrawler.builtin import GoogleImageCrawler
from six.moves.urllib.parse import urlparse

class PrefixNameDownloader(ImageDownloader):

    def get_filename(self, task, default_ext):
        filename = super(PrefixNameDownloader, self).get_filename(
            task, default_ext)
        return 'prefix_' + filename


class Base64NameDownloader(ImageDownloader):

    def get_filename(self, task, default_ext):
        url_path = urlparse(task['file_url'])[2]
        if '.' in url_path:
            extension = url_path.split('.')[-1]
            if extension.lower() not in [
                    'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'ppm', 'pgm'
            ]:
                extension = default_ext
        else:
            extension = default_ext
        filename = base64.b64encode(url_path.encode()).decode()
        return '{}.{}'.format(filename, extension)

def test_google(dir,keyword):
    print('启用google爬虫')
    google_crawler = GoogleImageCrawler(parser_threads=20,
                                        downloader_threads=20,
                                        downloader_cls=Base64NameDownloader,
                                        storage={'root_dir': dir},
                                        log_level = logging.INFO)
    google_crawler.crawl(keyword=keyword, offset=0, max_num=1000,min_size=(200,200), max_size=None)


def test_bing(dir,keyword):
    keyword = keyword.replace(': flickr.com', '')
    print('启用bing爬虫',keyword)
    bing_crawler = BingImageCrawler(
                                    # parser_threads=16,
                                    downloader_cls=Base64NameDownloader,
                                    downloader_threads=16,
                                    storage={'root_dir': dir},
                                    log_level=logging.DEBUG)
    bing_crawler.crawl(keyword=keyword,offset=0, max_num=1000,min_size=None,max_size=None)

def test_baidu(dir,keyword):
    keyword = keyword.replace(': flickr.com', '')
    print('启用百度爬虫',keyword)
    baidu_crawler = BaiduImageCrawler(
                                    # parser_threads=16,
                                    # downloader_threads=16,
                                    downloader_cls=Base64NameDownloader,
                                    storage={'root_dir': dir},
                                    log_level = logging.DEBUG)
    baidu_crawler.crawl(keyword=keyword, offset=0,max_num=1000,min_size=None,max_size=None)


def main():
##################################################################
            keyword='supplement facts image'
            base_dir='C:/carlos/image_download/supplement_facts_image'
            if len(sys.argv) == 1:
                dst = 'all'
            else:
                dst = sys.argv[1:]
            if 'all' in dst:
                dst = ['google', ]
            if 'google' in dst:
                test_google(base_dir,keyword)
            if 'bing' in dst:
                test_bing(base_dir,keyword)
            if 'baidu' in dst:
                test_baidu(base_dir,keyword)


if __name__ == '__main__':
    main()