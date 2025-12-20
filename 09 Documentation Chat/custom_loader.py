from langchain_core.documents import Document
from langchain_community.document_loaders import BSHTMLLoader
from bs4 import BeautifulSoup

class UTF8BSHTMLLoader(BSHTMLLoader):
    def lazy_load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)
            text = soup.get_text()
            metadata = {"source": str(self.file_path)}
            yield Document(page_content=text, metadata=metadata)
