import urllib.request
import re

url = "http://pythonprogramming.net/parse-website-using-regular-expressions-urllib/"

req = urllib.request.Request(url)
resp = urllib.request.urlopen(req)
respData = resp.read()

paragraphs = re.findall(r'<p>(.*?)</p>', str(respData))

for paragraph in paragraphs:
	print(paragraph)
