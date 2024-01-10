import torch
import psutil


def count_metadata(metadata):
    counter = {
        "artist": {},
        "character": {},
        "style": {},
        "quality": {},
    }
    for img_key, md in metadata.items():
        artist, characters, styles, quality = md.get('artist'), md.get('characters'), md.get('styles'), md.get('quality')
        if artist:
            if artist not in counter["artist"]:
                counter["artist"][artist] = 0
            counter["artist"][artist] += 1
        if characters:
            for character in characters:
                if character not in counter["character"]:
                    counter["character"][character] = 0
                counter["character"][character] += 1
        if styles:
            for style in styles:
                if style not in counter["style"]:
                    counter["style"][style] = 0
                counter["style"][style] += 1
        if quality:
            if quality not in counter["quality"]:
                counter["quality"][quality] = 0
            counter["quality"][quality] += 1
    return counter


def get_num_repeats(img_key, img_md, counter, benchmark):
    least_num_repeats = 1
    num_repeats = least_num_repeats
    artist = img_md.get('artist')
    if artist:
        num_repeats *= min(benchmark / counter["artist"][artist], 5)
    num_repeats = int(num_repeats)
    num_repeats = max(least_num_repeats, num_repeats)
    return num_repeats


AESTHETIC_TAGS = {
    'aesthetic',
    'beautiful_color',
    'beautiful',
    'detailed',
}


def process_caption(caption, fixed_tag_dropout_rate, flex_tag_dropout_rate, shuffle_caption, artist_tag_weight=0.4, style_tag_weight=0.3, character_tag_weight=0.3):
    import random
    tags = caption.split(",")
    tags = [tag.strip().replace(' ', '_').replace(':_', ':') for tag in tags]
    n = len(tags)
    fixed_bitmap = [tag.startswith(("artist:", "character:", "style:")) or 'quality' in tag or tag in AESTHETIC_TAGS for tag in tags]
    fixed_tags, flex_tags = [], []
    for tag, is_fixed in zip(tags, fixed_bitmap):
        if is_fixed:
            fixed_tags.append(tag)
        else:
            flex_tags.append(tag)

    if fixed_tag_dropout_rate:
        fixed_tags = [tag for tag in fixed_tags if random.random() > fixed_tag_dropout_rate]
    if flex_tag_dropout_rate:
        flex_tags = [tag for tag in flex_tags if random.random() > flex_tag_dropout_rate]

    if shuffle_caption:
        random.shuffle(flex_tags)

    tags = fixed_tags + flex_tags
    # every fixed tag has a probability to be the first tag
    # the probability is proportional to its weight
    # total probability is 1
    artist_tags, character_tags, style_tags = [], [], []
    for tag in fixed_tags:
        if tag.startswith("artist:"):
            artist_tags.append(tag)
        elif tag.startswith("character:"):
            character_tags.append(tag)
        elif tag.startswith("style:"):
            style_tags.append(tag)

    total_weight = (artist_tag_weight if len(artist_tags) > 0 else 0) + (style_tag_weight if len(style_tags) > 0 else 0) + (character_tag_weight if len(character_tags) > 0 else 0)
    if total_weight != 0:
        artist_tag_weight /= total_weight
        style_tag_weight /= total_weight
        character_tag_weight /= total_weight

        coin = random.random()
        if coin < artist_tag_weight:
            for tag in artist_tags:
                tags.remove(tag)
                tags.insert(0, tag)
        elif coin < artist_tag_weight + style_tag_weight:
            for tag in style_tags:
                tags.remove(tag)
                tags.insert(0, tag)
        else:
            for tag in character_tags:
                tags.remove(tag)
                tags.insert(0, tag)

    if 'worst quality' in tags:
        tags.remove('worst quality')
        tags.insert(0, 'worst quality')
    if 'low quality' in tags:
        tags.remove('low quality')
        tags.insert(0, 'low quality')
    if 'amazing quality' in tags:
        tags.remove('amazing quality')
        tags.insert(0, 'amazing quality')

    return ', '.join(tags)
