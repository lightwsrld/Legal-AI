# 법률저널 기사 크롤링 스크립트

``` bash

python3 lawtimes_case_crawler.py \
    --url "https://www.lawtimes.co.kr/Case-curation?page=1&con=%ED%8C%90%EA%B2%B0%EA%B8%B0%EC%82%AC&cat=" \
    --out lawtimes_all.csv
    --start 1 \
    --end 0 \ # end가 0이면 제한없음

# https://www.lawtimes.co.kr/Case-curation?page=1&con=%ED%8C%90%EA%B2%B0%EA%B8%B0%EC%82%AC&cat=%ED%98%95%EC%82%AC%EC%9D%BC%EB%B0%98
```