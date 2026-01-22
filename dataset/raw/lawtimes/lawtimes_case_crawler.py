#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lawtimes 판결큐레이션(민사일반) 전체 페이지 크롤러

- 본문: 반드시 div.css-1rywr2z.e1ogx6dn0 에서 우선 추출
- 컨테이너가 비거나 텍스트가 없으면 <meta> description 계열을 폴백으로 사용
- --end 0 이면 새 링크가 2페이지 연속 안 나오면 종료(상한 없음)
- --render 주면 본문 비었을 때 JS 렌더링으로 재시도
- CSV: url,title,content (UTF-8 BOM)

python3 lawtimes_case_crawler.py \
    --url "https://www.lawtimes.co.kr/Case-curation?page=1&con=%ED%8C%90%EA%B2%B0%EA%B8%B0%EC%82%AC&cat=" \
    --out lawtimes_kiup.csv \
    --start 1 \
    --end 0 \
    --out lawtimes_all.csv

https://www.lawtimes.co.kr/Case-curation?page=1&con=%ED%8C%90%EA%B2%B0%EA%B8%B0%EC%82%AC&cat=%ED%98%95%EC%82%AC%EC%9D%BC%EB%B0%98
"""

import argparse
import csv
import sys
import time
import re
import html as htmlmod
from typing import List, Tuple, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import requests
from tqdm import tqdm as tqdm
from bs4 import BeautifulSoup

BASE = "https://www.lawtimes.co.kr"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def get_soup(url: str, html: Optional[str] = None) -> BeautifulSoup:
    if html is None:
        resp = SESSION.get(url, timeout=25)
        resp.raise_for_status()
        if resp.encoding is None:
            resp.encoding = "utf-8"
        html = resp.text
    return BeautifulSoup(html, "lxml")

def set_query_param(url: str, key: str, value: str) -> str:
    pu = urlparse(url)
    q = parse_qs(pu.query, keep_blank_values=True)
    q[key] = [str(value)]
    new_query = urlencode({k: v[-1] for k, v in q.items()})
    return urlunparse((pu.scheme, pu.netloc, pu.path, pu.params, new_query, pu.fragment))


def extract_article_links_from_soup(soup: BeautifulSoup) -> List[str]:
    links: Set[str] = set()
    for a in soup.select('a[href^="/Case-curation/"]'):
        href = a.get("href", "").strip()
        if re.fullmatch(r"/Case-curation/\d+", href):
            links.add(urljoin(BASE, href))
    if not links:
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if re.fullmatch(r"/Case-curation/\d+", href):
                links.add(urljoin(BASE, href))
    return sorted(links)

def extract_article_links_across_pages(listing_url: str, start: int, end: Optional[int], sleep: float) -> List[str]:
    """
    start~end 페이지 순회. end=None(--end 0)이면 '새 링크 0개'가 2페이지 연속 발생 시 종료.
    """
    collected: List[str] = []
    seen: Set[str] = set()
    empty_streak = 0
    page = max(1, start)

    while True:
        curr_url = set_query_param(listing_url, "page", str(page))
        try:
            soup = get_soup(curr_url)
        except Exception as e:
            print(f"[WARN] listing {curr_url} 요청 실패: {e}", file=sys.stderr)
            break

        links = extract_article_links_from_soup(soup)
        new_links = [u for u in links if u not in seen]
        for u in new_links:
            seen.add(u)
            collected.append(u)

        print(f"[INFO] page={page} links={len(links)} new={len(new_links)} total={len(collected)}", file=sys.stderr)

        if sleep > 0:
            time.sleep(sleep)

        if end is not None:
            if page >= end:
                break
        else:
            empty_streak = empty_streak + 1 if len(new_links) == 0 else 0
            if empty_streak >= 2:
                break

        page += 1

    return collected

def find_title(soup: BeautifulSoup) -> Optional[str]:
    title_div = soup.select_one("div.css-1jk5fvy.e16ienf60")
    if not title_div:
        title_div = soup.find(
            "div",
            class_=lambda c: isinstance(c, str) and (
                "e16ienf60" in c.split() or "css-1jk5fvy" in c.split()
            ),
        )
    if title_div:
        txt = title_div.get_text(strip=True)
        if txt:
            return txt
    h1 = soup.find(["h1", "h2"], string=True)
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    return None

def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def _extract_from_meta(soup: BeautifulSoup) -> Optional[str]:
    """<meta> description 계열에서 텍스트 복원(HTML 엔티티 디코딩)."""
    for attrs in (
        {"name": "description"},
        {"property": "og:description"},
        {"itemprop": "description"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            raw = tag["content"]
            decoded = htmlmod.unescape(raw)
            return _clean_text(decoded)
    return None

def extract_body_text_fixed(soup: BeautifulSoup) -> Optional[str]:
    """
    본문: div.css-1rywr2z.e1ogx6dn0 → (없거나 비면) meta description 폴백
    """
    node = soup.select_one("div.css-1rywr2z.e1ogx6dn0")
    if node:
        for t in node.find_all(["script", "style", "svg", "button", "noscript"]):
            t.decompose()
        txt = node.get_text("\n", strip=True)
        if txt and txt.strip():
            return _clean_text(txt)
    return _extract_from_meta(soup)


def render_with_requests_html(url: str, timeout: int = 25, wait: float = 1.5) -> Optional[str]:
    try:
        from requests_html import HTMLSession  # type: ignore
    except Exception as e:
        print(f"[INFO] requests_html 사용 불가: {e}", file=sys.stderr)
        return None
    sess = HTMLSession()
    try:
        r = sess.get(url, headers=HEADERS, timeout=timeout)
        r.html.render(wait=wait, sleep=wait, reload=False, scrolldown=0)
        return r.html.html
    except Exception as e:
        print(f"[WARN] JS 렌더링 실패: {e}", file=sys.stderr)
        return None
    finally:
        try:
            sess.close()
        except Exception:
            pass


def extract_article(article_url: str, use_render: bool = False, sleep: float = 0.5) -> Tuple[str, Optional[str], Optional[str]]:
    try:
        soup = get_soup(article_url)
        title = find_title(soup)
        content = extract_body_text_fixed(soup)
        if (not content) and use_render:
            print(f"[INFO] 정적 본문 없음 → JS 렌더링 재시도: {article_url}", file=sys.stderr)
            html2 = render_with_requests_html(article_url)
            if html2:
                soup2 = get_soup(article_url, html=html2)
                content = extract_body_text_fixed(soup2)
        if sleep > 0:
            time.sleep(sleep)
        return article_url, title, content
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed: {article_url} -> {e}\n")
        return article_url, None, None

def save_csv(rows: List[Tuple[str, Optional[str], Optional[str]]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["url", "title", "content"])
        for url, title, content in rows:
            writer.writerow([url or "", title or "", content or ""])

def main():
    p = argparse.ArgumentParser(description="Lawtimes 판결큐레이션(민사일반) 전체 페이지 크롤러")
    p.add_argument("--url", required=True, help="리스트 페이지 URL (예: page=1 포함)")
    p.add_argument("--out", default="lawtimes_minsa.csv", help="출력 CSV 경로")
    p.add_argument("--sleep", type=float, default=0.5, help="요청 간 sleep 초")
    p.add_argument("--start", type=int, default=1, help="시작 페이지")
    p.add_argument("--end", type=int, default=0, help="종료 페이지(0=자동 종료)")
    p.add_argument("--render", action="store_true", help="본문 비면 JS 렌더링 재시도")
    args = p.parse_args()

    listing_url = args.url
    end = args.end if args.end > 0 else None

    article_links = extract_article_links_across_pages(listing_url, start=args.start, end=end, sleep=args.sleep)
    if not article_links:
        sys.stderr.write("[INFO] 상세 기사 링크를 찾지 못했습니다. 선택자/URL을 확인하세요.\n")

    rows: List[Tuple[str, Optional[str], Optional[str]]] = []
    _total = len(article_links)
    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"

    with tqdm(total=_total, unit="art", dynamic_ncols=True, bar_format=bar_fmt) as pbar:
        for i, url in enumerate(article_links, start=1):
            pbar.set_description(f"Fetch {i}/{_total}")
            pbar.set_postfix_str(url if len(url) <= 100 else url[:97] + "...")
            rows.append(extract_article(url, sleep=args.sleep))
            pbar.update(1)
    save_csv(rows, args.out)
    print(f"[DONE] Saved -> {args.out} (rows={len(rows)})")

if __name__ == "__main__":
    main()
