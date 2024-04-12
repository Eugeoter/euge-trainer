# ! You can modify these functions to use your own custom dataset management
import random
import os
from pathlib import Path


def fmt2dan(tag):
    if not tag:
        return tag
    elif isinstance(tag, list):
        return [fmt2dan(t) for t in tag]
    else:
        tag = tag.lower().strip()
        tag = tag.replace(' ', '_').replace('\\(', '(').replace('\\)', ')')
        return tag


def count_metadata(metadata):
    counter = {
        "category": {},
        "artist": {},
        "character": {},
        "style": {},
        "quality": {},
    }
    for img_key, img_md in metadata.items():
        num_repeats = img_md.get('num_repeats', 1)
        category = os.path.basename(os.path.dirname(img_md['image_path']))
        if category not in counter["category"]:
            counter["category"][category] = 0
        counter["category"][category] += num_repeats

        artist, characters, styles, quality = img_md.get('artist'), img_md.get('characters'), img_md.get('styles'), img_md.get('quality')

        if artist:
            artist = fmt2dan(artist)
            if artist not in counter["artist"]:
                counter["artist"][artist] = 0
            counter["artist"][artist] += num_repeats
        if characters:
            for character in characters.split(', '):
                character = fmt2dan(character)
                if character not in counter["character"]:
                    counter["character"][character] = 0
                counter["character"][character] += num_repeats
        if styles:
            for style in styles.split(', '):
                style = fmt2dan(style)
                if style not in counter["style"]:
                    counter["style"][style] = 0
                counter["style"][style] += num_repeats
        if quality:
            if quality not in counter["quality"]:
                counter["quality"][quality] = 0
            counter["quality"][quality] += num_repeats
    return counter


AESTHETIC_TAGS = {
    'aesthetic',
    'beautiful color',
    'beautiful',
    'detailed',
}


def get_num_repeats(img_key, img_md, artist_benchmark=100, character_benchmark=1000, **kwargs):
    r"""
    Num repeats getter of AIDXL.
    """
    counter = kwargs.get('counter', None)
    if not counter:
        raise ValueError("counter is required for AIDXL num_repeats getter")

    if img_key.startswith(('celluloid_animation_screenshot')):
        return 0
    least_num_repeats = 1
    max_concept_multiple = 10
    num_repeats = least_num_repeats
    concept_multiple = num_repeats
    artist = fmt2dan(img_md.get('artist'))
    if artist and (num_artist := counter["artist"][artist]) >= 20:
        concept_multiple *= min(artist_benchmark / num_artist, 5)

    characters = img_md.get('characters')
    if characters:
        characters = characters.split(', ')
        for character in characters:
            character = fmt2dan(character)
            if (num_character := counter["character"][character]) >= 50:
                concept_multiple *= min(character_benchmark / num_character, 10)

    num_repeats *= min(concept_multiple, max_concept_multiple)
    quality = img_md.get('quality', 'normal')

    # if quality not in ('best', 'amazing'):
    #     return 0

    if quality == 'normal':
        num_repeats *= 1
    elif quality == 'high':
        num_repeats *= 1.5
    elif quality == 'best':
        num_repeats *= 2
    elif quality == 'amazing':
        num_repeats *= 5
    elif quality == 'low' or quality == 'worst':
        return 0

    caption = img_md.get('tags', '')
    if any(tag in caption for tag in ('beautiful color', 'beautiful')):
        num_repeats *= 1.5

    source = Path(img_md['image_path']).parent.parent.name
    root = Path(img_md['image_path']).parent.parent.parent.name
    if source == 'viscept':  # neta
        category = os.path.basename(os.path.dirname(img_md['image_path']))
        if category in ('impasto', 'gufeng'):
            return 0
        return 1  # fix to 1
    elif source in ('original', 'dataset'):  # aid-0
        pass
    elif source in ('download-5', 'preparation-5'):
        num_repeats *= 2
    elif source in ('download-4', 'preparation-4'):  # aid-3
        num_repeats *= 1
    elif source in ('download-3', 'preparation-3'):  # aid-2
        num_repeats *= 1
    elif source in ('download', 'preparation', 'download-2', 'preparation-2'):  # aid-1
        num_repeats *= 1
    elif root == 'cosplay':  # cosplay
        cosplay_benchmark = 250
        category = os.path.basename(os.path.dirname(img_md['image_path']))
        num_repeats *= min(cosplay_benchmark / counter["category"][category], 5)
        if 'hane ame' in characters:
            if random.random() < 0.9:
                return 0
        return 1  # fix to 1
    else:
        raise ValueError(f"unknown data source: `{source}` | img_path: `{img_md['image_path']}`")

    num_repeats = int(num_repeats)
    num_repeats = max(least_num_repeats, num_repeats)

    return num_repeats

# For neta


# def get_num_repeats(img_key, img_md, counter, artist_benchmark, character_benchmark, reg_metadata):
#     if img_key.startswith(('celluloid_animation_screenshot')):
#         return 0
#     least_num_repeats = 1
#     num_repeats = least_num_repeats
#     num_repeats *= 2
#     # artist = img_md.get('artist')
#     # if artist and counter["artist"][artist] >= 10:
#     #     num_repeats *= min(benchmark / counter["artist"][artist], 10)

#     quality = img_md.get('quality', 'normal')

#     # if quality not in ('best', 'amazing'):
#     #     return 0

#     if quality == 'normal':
#         num_repeats *= 1
#     elif quality == 'high':
#         num_repeats *= 1.5
#     elif quality == 'best':
#         num_repeats *= 2
#     elif quality == 'amazing':
#         num_repeats *= 5
#     elif quality == 'low' or quality == 'worst':
#         return 0

#     caption = img_md.get('tags', '')
#     if any(tag in caption for tag in ('beautiful color', 'beautiful')):
#         num_repeats *= 1.5

#     source = Path(img_md['image_path']).parent.parent.stem
#     root = Path(img_md['image_path']).parent.parent.parent.stem
#     if source == 'viscept':
#         aesthetic_score = img_md.get('aesthetic_score', None)
#         if aesthetic_score is not None and aesthetic_score <= 1.5:
#             return 0
#         # styles = img_md.get('styles', '').split(', ')
#         # if 'kr-ani' in styles or 'celluloid' in styles or 'helltaker maker' in styles:
#         #     num_repeats /= 2
#         # elif 'pixel-art' in styles or 'tiaotiaotang' in styles or 'line-draft' in styles:
#         #     num_repeats *= 2
#     else:
#         aesthetic_score = img_md.get('aesthetic_score', None)
#         if quality not in ('amazing', 'best') and (aesthetic_score is None or aesthetic_score <= 8.0):
#             return 0

#     num_repeats = int(num_repeats)
#     num_repeats = max(least_num_repeats, num_repeats)

#     return num_repeats


def score2quality(score):
    if score >= 9.5:
        return 'amazing'
    elif score >= 8.0:
        return 'best'
    elif score >= 7.0:
        return 'high'
    elif score >= 4.0:
        return 'normal'
    elif score >= 2.5:
        return 'low'
    elif score >= 1.0:
        return 'worst'
    else:
        return 'horrible'


def process_caption(
    img_info,
    fixed_tag_dropout_rate=0,
    flex_tag_dropout_rate=0,
    tags_shuffle_prob=0.0,
    tags_shuffle_rate=1.0,
    **kwargs,
):
    caption = img_info.caption
    img_md = img_info.metadata
    counter = kwargs.get('counter', None)

    tags = caption.split(", ")
    fixed_bitmap = [tag.startswith(("artist:", "character:", "style:")) or 'quality' in tag or tag in AESTHETIC_TAGS for tag in tags]
    fixed_tags, flex_tags = [], []
    for i, (tag, is_fixed) in enumerate(zip(tags, fixed_bitmap)):
        if is_fixed or random.random() < tags_shuffle_rate:
            fixed_tags.append(tag)
            fixed_bitmap[i] = True
        else:
            flex_tags.append(tag)

    if random.random() < tags_shuffle_prob:
        random.shuffle(flex_tags)

    proc_tags = []
    for is_fixed in fixed_bitmap:
        if is_fixed:
            tag = fixed_tags.pop(0)
            if random.random() > fixed_tag_dropout_rate:
                proc_tags.append(tag)
        else:
            tag = flex_tags.pop(0)
            if random.random() > flex_tag_dropout_rate:
                proc_tags.append(tag)
    tags = proc_tags

    artist = img_md.get('artist')
    characters = img_md.get('characters')
    # styles = fmt2dan(img_md.get('styles'))
    quality = img_md.get('quality')
    aesthetic_score = img_md.get('aesthetic_score')

    if not quality and aesthetic_score is not None:
        quality = score2quality(aesthetic_score)

    # dropout artist tags with high frequency
    if counter:
        if artist:
            artist_tag = f"artist: {artist}"
            if artist_tag not in tags:
                artist_tag = f"artist:{artist}"
            cnt = counter["artist"][fmt2dan(artist)]
            if cnt <= 10:
                try:
                    tags.remove(artist_tag)
                except:
                    print(f"artist_tag: `{artist_tag}` not in tags: {tags}")
            elif random.random() < 0.75:  # 75% chance to drop artist tag
                artist_benchmark = 1000
                artist_tag_dropout_rate = max(0, 1 - (artist_benchmark / cnt))
                if artist_tag_dropout_rate > 0 and random.random() < artist_tag_dropout_rate:  # dropout artist tag
                    try:
                        tags.remove(artist_tag)
                    except:
                        print(f"artist_tag: {artist_tag} not in tags: {tags}")

    if characters:
        # 50% chance to prepend a character tag
        for character in characters.split(", "):
            if random.random() < 0.5:
                tags.insert(0, f"character: {character}")
                if f"character: {character}" in tags:
                    tags.remove(f"character: {character}")

    tags = ['by ' + tag[7:].strip() if tag.startswith("artist:") else tag[9:].strip() if tag.startswith(
        "character:") else tag[6:].strip() if tag.startswith("style:") else tag for tag in tags]

    if quality:
        tags.append(quality + " quality")

    caption = ', '.join(tags)
    return caption


def process_description(
    img_info,
    **kwargs,
):
    desc = img_info.description
    flip_aug = kwargs.get('flip_aug', False)
    if flip_aug:
        desc = desc.replace("left", "right").replace("right", "left")
    return desc


def get_artist_tag(artist):
    coin = random.randint(1, 6)
    if coin == 1:
        return f"artist: {artist}"
    elif coin == 2:
        return f"art by {artist}"
    elif coin == 3:
        return f"drawn by {artist}"
    elif coin == 4:
        return f"illustrated by {artist}"
    elif coin == 5:
        return f"by {artist}"
    elif coin == 6:
        return f"{artist} style"
    else:
        raise ValueError(f"Invalid coin value: {coin}")


def get_character_tag(character):
    coin = random.randint(1, 4)
    if coin == 1:
        return f"character: {character}"
    elif coin == 2:
        return character
    elif coin == 3:
        return character.capitalize()
    elif coin == 4:
        return f"character named {character}"
    else:
        raise ValueError(f"Invalid coin value: {coin}")


def get_style_tag(style):
    coin = random.randint(1, 6)
    if coin == 1:
        return f"style: {style}"
    elif coin == 2:
        return style
    elif coin == 3:
        return f"drawn in {style} style"
    elif coin == 4:
        return f"in {style} style"
    elif coin == 5:
        return f"{style} style"
    elif coin == 6:
        return f"art in {style} style"
    else:
        raise ValueError(f"Invalid coin value: {coin}")
