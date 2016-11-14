from bs4 import BeautifulSoup
import urllib
r = urllib.urlopen('https://www.gametabs.net/elder-scrolls-v-skyrim/sons-skyrim').read()
soup = BeautifulSoup(r)
print type(soup)

''' Access denied.  Cloudflare?
'''
