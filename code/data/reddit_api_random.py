import requests, re, csv, time, random, os
from langdetect import detect
from datetime import datetime

def to_skip(object, lang="en"):

    try:
        language = detect(object['body'])
    except:
        language = "error"

    return any([
        language != lang,
        object['author'] == 'AutoModerator',
        object['body'] in ['[deleted]', '[removed]', ''],
        not re.search('[a-zA-Z]', object['body'])
    ])


def main(datafile, sample_size):

    counter = 0
    url = f"https://api.pushshift.io/reddit/search/comment/?size=500&order=asc&after="
    now = int(time.time())
    file_exists = os.path.isfile(datafile)

    with open(datafile, 'a') as outfile: 
        header = (['text', 'id', 'author', 'created_utc', 'url']) 
        writer = csv.DictWriter(outfile, fieldnames=header)
        if not file_exists:
            writer.writeheader() 

        while counter < sample_size:

            utc = random.randrange(1136070000, now) # 01-01-2006 to now
            response = requests.get(url+str(utc))
            
            while response.status_code in [500, 524, 429]:
                print("HTTP response status code: ", response.status_code)
                time.sleep(1)
                response = requests.get(url)

            if response.status_code == 200:

                json_data = response.json()
                objects = json_data['data']

                for object in objects:
                    comment_url = None if 'permalink' not in object else object['permalink']
                    if not to_skip(object):
                        comment_dict = {'text': object['body'],
                                    'id': object['id'],
                                    'author': object['author'],
                                    'created_utc': datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"),
                                    'url': comment_url}
                        writer.writerow(comment_dict)
                        counter += 1
                        print(counter)

            else:
                print("HTTP response status code: ", response.status_code)
                break
        
        print('now = ', now)

if __name__ == '__main__':

    datafile = '../../Data/RandomReddit/sample.csv'
    sample_size = 10000000
    main(datafile, sample_size)