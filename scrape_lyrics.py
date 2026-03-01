"""
Scrape Taylor Swift lyrics from Genius.com.

Usage:
    python scrape_lyrics.py

Writes output to data/lyrics.txt.
"""

import os
import re
import time
import random
import requests
from bs4 import BeautifulSoup

OUTPUT_PATH = "data/lyrics.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Key songs from each major album — enough for good training data
SONGS = [
    # Taylor Swift (Debut)
    "tim-mcgraw", "teardrops-on-my-guitar", "our-song", "picture-to-burn",
    "shouldve-said-no", "cold-as-you", "stay-beautiful", "mary-s-song-oh-my-my-my",
    # Fearless
    "fearless", "fifteen", "love-story", "white-horse", "you-belong-with-me",
    "breathe", "tell-me-why", "youre-not-sorry", "the-way-i-loved-you",
    "forever-and-always", "the-best-day", "jump-then-fall",
    # Speak Now
    "mine", "sparks-fly", "back-to-december", "speak-now", "dear-john",
    "mean", "the-story-of-us", "never-grow-up", "enchanted", "last-kiss",
    "long-live", "haunted", "innocent",
    # Red
    "state-of-grace", "red", "treacherous", "i-knew-you-were-trouble",
    "all-too-well", "22", "i-almost-do", "we-are-never-ever-getting-back-together",
    "stay-stay-stay", "the-last-time", "holy-ground", "sad-beautiful-tragic",
    "the-lucky-one", "everything-has-changed", "starlight", "begin-again",
    # 1989
    "welcome-to-new-york", "blank-space", "style", "out-of-the-woods",
    "shake-it-off", "i-wish-you-would", "bad-blood", "wildest-dreams",
    "how-you-get-the-girl", "this-love", "clean", "new-romantics",
    # Reputation
    "ready-for-it", "end-game", "i-did-something-bad",
    "dont-blame-me", "delicate", "look-what-you-made-me-do",
    "so-it-goes", "gorgeous", "getaway-car", "king-of-my-heart",
    "dancing-with-our-hands-tied", "dress", "this-is-why-we-cant-have-nice-things",
    "call-it-what-you-want", "new-years-day",
    # Lover
    "i-forgot-that-you-existed", "cruel-summer", "lover", "the-man",
    "the-archer", "i-think-he-knows", "miss-americana-and-the-heartbreak-prince",
    "paper-rings", "cornelia-street", "death-by-a-thousand-cuts",
    "london-boy", "soon-youll-get-better", "afterglow",
    "me", "you-need-to-calm-down", "daylight",
    # Folklore
    "the-1", "cardigan", "the-last-great-american-dynasty",
    "exile", "my-tears-ricochet", "mirrorball", "seven",
    "august", "this-is-me-trying", "illicit-affairs",
    "invisible-string", "mad-woman", "epiphany",
    "betty", "peace", "hoax",
    # Evermore
    "willow", "champagne-problems", "gold-rush",
    "tis-the-damn-season", "tolerate-it", "no-body-no-crime",
    "happiness", "dorothea", "coney-island",
    "ivy", "cowboy-like-me", "long-story-short",
    "marjorie", "closure", "evermore",
    # Midnights
    "lavender-haze", "maroon", "anti-hero",
    "snow-on-the-beach", "youre-on-your-own-kid", "midnight-rain",
    "question", "vigilante-shit", "bejeweled",
    "labyrinth", "karma", "sweet-nothing",
    "mastermind", "would-ve-could-ve-should-ve",
]


def get_lyrics(song_slug):
    """Fetch lyrics from a Genius song page."""
    url = f"https://genius.com/Taylor-swift-{song_slug}-lyrics"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return None, None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Get clean title from the page
    title = song_slug.replace("-", " ").title()
    title_tag = soup.find("h1")
    if title_tag:
        # Extract just the song name, stripping " Lyrics" suffix
        t = title_tag.get_text(strip=True)
        t = re.sub(r"\s*Lyrics$", "", t)
        if t:
            title = t

    # Use Lyrics__Root to get clean lyrics text
    lyrics_root = soup.select_one("[class*='Lyrics__Root']")
    if not lyrics_root:
        return title, None

    # Replace <br> with newlines
    for br in lyrics_root.find_all("br"):
        br.replace_with("\n")
    # Remove script/style tags
    for tag in lyrics_root.find_all(["script", "style"]):
        tag.decompose()

    text = lyrics_root.get_text(separator="")

    # Strip contributor/about preamble (everything before the first [section] marker)
    match = re.search(r"\[", text)
    if match:
        text = text[match.start():]

    # Remove [Verse 1], [Chorus], etc.
    text = re.sub(r"\[.*?\]", "", text)
    # Clean up multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return title, text


def main():
    os.makedirs("data", exist_ok=True)

    all_lyrics = []
    failed = []

    for i, slug in enumerate(SONGS):
        print(f"  [{i+1}/{len(SONGS)}] {slug}...", end=" ", flush=True)
        try:
            title, lyrics = get_lyrics(slug)
            if lyrics and len(lyrics) > 50:
                all_lyrics.append(f"[{title}]\n{lyrics}\n")
                print(f"OK ({len(lyrics)} chars)")
            else:
                print("(empty/short)")
                failed.append(slug)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(slug)

        # Be polite
        time.sleep(random.uniform(1.5, 3.0))

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n\n".join(all_lyrics))

    size = os.path.getsize(OUTPUT_PATH)
    print(f"\nDone! Saved {len(all_lyrics)} songs to {OUTPUT_PATH} ({size:,} bytes)")
    if failed:
        print(f"Failed songs ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
