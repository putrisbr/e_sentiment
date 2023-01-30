import twint
import datetime

start_date = datetime.date(2020, 3, 9) #datetime.date(year, month, day)
end_date = datetime.date(2022, 2, 2)
delta = datetime.timedelta(days=1)

while (start_date <= end_date): #INI UNTUK MENGATUR BERAPA LAMA CRAWL DATA
    n = 1
    while n < 3:
        c = twint.Config()
        c.Search = "ETH"
        c.Since = "2020-03-08"
        c.Until = str(start_date)
        c.Store_csv = True
        c.Verified = False
        c.Limit = 10
        c.Lang = "en"
        c.Output = "eth.xls"
        c.Pandas = True
        twint.run.Search(c)
        n+=1
        print(start_date)
    start_date += delta