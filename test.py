import datetime
s = '2022-07-28'
s2 = '2022-08-02'
start_date = datetime.datetime.strptime(s.split('/')[-1].split('.')[0], '%Y-%m-%d')
date = datetime.datetime.strptime(s2.split('/')[-1].split('.')[0], '%Y-%m-%d')

print(start_date)
print(date)
print(date.day - start_date.day)
print(date.day)
print((start_date - date).days)