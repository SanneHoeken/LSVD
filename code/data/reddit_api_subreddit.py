import requests, time, csv, re
from datetime import datetime

def to_skip(object):

    return any([
        object['author'] == 'AutoModerator',
        object['body'] in ['[deleted]', '[removed]', ''],
        not re.search('[a-zA-Z]', object['body'])
    ])

def main(subreddit, datafile):

    count = 0
    url = f"https://api.pushshift.io/reddit/comment/search/?subreddit={subreddit}&size=500&order=asc&after="
    #start_time = datetime.utcnow()
    #timestamp = int(start_time.timestamp())
    timestamp = 1677103989

    with open(datafile, 'a') as outfile: 
        header = (['text', 'id', 'author', 'created_utc', 'url']) 
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader() 
        
        while True:
            request_url = url+str(timestamp)
            response = requests.get(request_url)
            
            while response.status_code in [500, 524, 429]:
                print("HTTP response status code: ", response.status_code)
                time.sleep(1)
                response = requests.get(request_url)

            if response.status_code == 200:

                json_data = response.json()
                objects = json_data['data']

                for object in objects:
                    timestamp = object['created_utc']
                    comment_url = None if 'permalink' not in object else object['permalink']
                    if not to_skip(object):
                        comment_dict = {'text': object['body'],
                                    'id': object['id'],
                                    'author': object['author'],
                                    'created_utc': datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"),
                                    'url': comment_url}
                        writer.writerow(comment_dict)
                        count += 1
                        print(count, timestamp)

            else:
                print("HTTP response status code: ", response.status_code)
                print(timestamp)
                break
                           
if __name__ == '__main__':

    subreddit = 'hillaryclinton'
    datafile = f'../../Data/hillaryclinton_comments/hillaryclinton_comments_afterEpoch=1677103989.csv'
    main(subreddit, datafile)
            