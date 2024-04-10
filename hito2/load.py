import gdown

url = 'https://drive.google.com/uc?id=1XkSp7vMsTrp9aTY40RtZhVqdZPMyguwM'
output = 'pavia.txt'

gdown.download(url, output, quiet=False)