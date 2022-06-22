import requests
from PIL import Image
from io import BytesIO
import time


start_time = time.perf_counter()
res = requests.get('https://pbs.twimg.com/media/EPstuxsUEAER6KW.jpg')
end_time = time.perf_counter()
print(end_time - start_time)
i = Image.open(BytesIO(res.content))
i.save(f"test.{i.format.lower()}")