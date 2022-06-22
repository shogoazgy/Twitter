import requests
from PIL import Image
from io import BytesIO
import time
from urllib.parse import urlparse

"""
i = Image.open(BytesIO(res.content))
i.save(f"test.{i.format.lower()}")
"""

id = urlparse('https://pbs.twimg.com/media/EPoI73mXkAEWdOX.jpg').path[7:]
print(id)