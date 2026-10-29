from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from threading import Thread
import unittest
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


class PageParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class ProjectPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT))
        )
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # A non-root URL catches paths that would break a GitHub Pages subdirectory.
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/docs/"
        with urlopen(cls.url) as response:
            cls.page = PageParser(response.read().decode())

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def test_local_links_and_assets_are_served_under_a_subdirectory(self):
        ids = {attrs["id"] for _, attrs in self.page.elements if "id" in attrs}
        links = {
            attrs[key]
            for _, attrs in self.page.elements
            for key in ("src", "href")
            if key in attrs
        }
        for link in sorted(links):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            with self.subTest(link=link):
                if parsed.fragment and not parsed.path:
                    self.assertIn(unquote(parsed.fragment), ids)
                    continue
                self.assertFalse(parsed.path.startswith("/"), "Use relative site paths")
                with urlopen(urljoin(self.url, link)) as response:
                    self.assertEqual(response.status, 200)
                    content = response.read()
                    self.assertTrue(content)
                    if parsed.path.endswith(".pdf"):
                        self.assertTrue(content.startswith(b"%PDF-"))
                    if parsed.path.endswith(".webp"):
                        self.assertEqual(content[8:12], b"WEBP")
                    if parsed.path.endswith(".css"):
                        for asset in re.findall(r'url\(["\']?([^\)"\']+)', content.decode()):
                            with urlopen(urljoin(response.url, asset)) as font:
                                self.assertEqual(font.status, 200)
                                self.assertTrue(font.read())

    def test_figures_and_tabs_have_accessible_labels_and_targets(self):
        elements = self.page.elements
        ids = [attrs["id"] for _, attrs in elements if "id" in attrs]
        self.assertEqual(len(ids), len(set(ids)), "IDs must be unique")
        by_id = {attrs["id"]: attrs for _, attrs in elements if "id" in attrs}
        images = [attrs for tag, attrs in elements if tag == "img"]
        self.assertTrue(images)
        for image in images:
            with self.subTest(image=image["src"]):
                self.assertTrue(image.get("alt", "").strip())
                self.assertGreater(int(image["width"]), 0)
                self.assertGreater(int(image["height"]), 0)
        tabs = [attrs for _, attrs in elements if attrs.get("role") == "tab"]
        self.assertGreater(len(tabs), 1)
        self.assertEqual(sum(tab.get("aria-selected") == "true" for tab in tabs), 1)
        for tab in tabs:
            with self.subTest(tab=tab["id"]):
                panel = by_id[tab["aria-controls"]]
                self.assertEqual(panel["role"], "tabpanel")
                self.assertEqual(panel["aria-labelledby"], tab["id"])
                self.assertNotIn("hidden", panel, "Figures remain readable without JavaScript")


if __name__ == "__main__":
    unittest.main()
