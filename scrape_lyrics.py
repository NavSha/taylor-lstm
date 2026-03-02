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

# Comprehensive Taylor Swift discography for maximum training data
SONGS = [
    # Taylor Swift (Debut)
    "tim-mcgraw", "teardrops-on-my-guitar", "our-song", "picture-to-burn",
    "shouldve-said-no", "cold-as-you", "stay-beautiful", "marys-song-oh-my-my-my",
    "a-place-in-this-world", "the-outside", "tied-together-with-a-smile",
    # Fearless
    "fearless", "fifteen", "love-story", "white-horse", "you-belong-with-me",
    "breathe", "tell-me-why", "youre-not-sorry", "the-way-i-loved-you",
    "forever-and-always", "the-best-day", "jump-then-fall",
    # Fearless TV vault tracks
    "you-all-over-me-taylors-version-from-the-vault",
    "mr-perfectly-fine-taylors-version-from-the-vault",
    "thats-when-taylors-version-from-the-vault",
    "we-were-happy-taylors-version-from-the-vault",
    "dont-you-taylors-version-from-the-vault",
    "bye-bye-baby-taylors-version-from-the-vault",
    # Speak Now
    "mine", "sparks-fly", "back-to-december", "speak-now", "dear-john",
    "mean", "the-story-of-us", "never-grow-up", "enchanted", "last-kiss",
    "long-live", "haunted", "innocent",
    # Speak Now TV vault tracks
    "electric-touch-taylors-version-from-the-vault",
    "when-emma-falls-in-love-taylors-version-from-the-vault",
    "i-can-see-you-taylors-version-from-the-vault",
    "castles-crumbling-taylors-version-from-the-vault",
    "foolish-one-taylors-version-from-the-vault",
    "timeless-taylors-version-from-the-vault",
    # Red
    "state-of-grace", "red", "treacherous", "i-knew-you-were-trouble",
    "all-too-well", "22", "i-almost-do", "we-are-never-ever-getting-back-together",
    "stay-stay-stay", "the-last-time", "holy-ground", "sad-beautiful-tragic",
    "the-lucky-one", "everything-has-changed", "starlight", "begin-again",
    # Red TV vault tracks
    "better-man-taylors-version-from-the-vault",
    "nothing-new-taylors-version-from-the-vault",
    "babe-taylors-version-from-the-vault",
    "message-in-a-bottle-taylors-version-from-the-vault",
    "i-bet-you-think-about-me-taylors-version-from-the-vault",
    "forever-winter-taylors-version-from-the-vault",
    "run-taylors-version-from-the-vault",
    "the-very-first-night-taylors-version-from-the-vault",
    "all-too-well-10-minute-version-taylors-version-from-the-vault",
    # 1989
    "welcome-to-new-york", "blank-space", "style", "out-of-the-woods",
    "shake-it-off", "i-wish-you-would", "bad-blood", "wildest-dreams",
    "how-you-get-the-girl", "this-love", "clean", "new-romantics",
    "all-you-had-to-do-was-stay", "you-are-in-love", "wonderland",
    # 1989 TV vault tracks
    "is-it-over-now-taylors-version-from-the-vault",
    "now-that-we-dont-talk-taylors-version-from-the-vault",
    "say-dont-go-taylors-version-from-the-vault",
    "suburban-legends-taylors-version-from-the-vault",
    "slut-taylors-version-from-the-vault",
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
    "false-god", "its-nice-to-have-a-friend",
    # Folklore
    "the-1", "cardigan", "the-last-great-american-dynasty",
    "exile", "my-tears-ricochet", "mirrorball", "seven",
    "august", "this-is-me-trying", "illicit-affairs",
    "invisible-string", "mad-woman", "epiphany",
    "betty", "peace", "hoax", "the-lakes",
    # Evermore
    "willow", "champagne-problems", "gold-rush",
    "tis-the-damn-season", "tolerate-it", "no-body-no-crime",
    "happiness", "dorothea", "coney-island",
    "ivy", "cowboy-like-me", "long-story-short",
    "marjorie", "closure", "evermore",
    "right-where-you-left-me", "its-time-to-go",
    # Midnights
    "lavender-haze", "maroon", "anti-hero",
    "snow-on-the-beach", "youre-on-your-own-kid", "midnight-rain",
    "question", "vigilante-shit", "bejeweled",
    "labyrinth", "karma", "sweet-nothing",
    "mastermind", "wouldve-couldve-shouldve",
    # The Tortured Poets Department (standard)
    "fortnight", "the-tortured-poets-department",
    "my-boy-only-breaks-his-favorite-toys", "down-bad",
    "so-long-london", "but-daddy-i-love-him",
    "fresh-out-the-slammer", "florida",
    "guilty-as-sin", "whos-afraid-of-little-old-me",
    "i-can-fix-him-no-really-i-can", "loml",
    "i-can-do-it-with-a-broken-heart",
    "the-smallest-man-who-ever-lived",
    "the-alchemy", "clara-bow",
    # The Tortured Poets Department (Anthology)
    "the-black-dog", "imgonnagetyouback", "the-albatross",
    "chloe-or-sam-or-sophia-or-marcus", "how-did-it-end",
    "so-high-school", "i-hate-it-here", "thank-you-aimee",
    "i-look-in-peoples-windows", "the-prophecy",
    "cassandra", "peter", "the-bolter", "robin", "the-manuscript",
    # Soundtrack / standalone singles
    "safe-and-sound", "eyes-open", "beautiful-ghosts",
    "only-the-young", "christmas-tree-farm",
    # Holiday
    "christmases-when-you-were-mine", "christmas-must-be-something-more",
    "last-christmas", "santa-baby", "silent-night", "white-christmas",
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
