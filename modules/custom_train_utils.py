# ! You can modify these functions to use your own custom dataset management
import random


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


AESTHETIC_TAGS = {
    'aesthetic',
    'beautiful color',
    'beautiful',
    'detailed',
}


def get_num_repeats(img_key, img_md, counter, benchmark, reg_metadata):
    if img_key.startswith(('celluloid_animation_screenshot', 'dainty_wilder')):
        return 0
    least_num_repeats = 1
    num_repeats = least_num_repeats
    artist = img_md.get('artist')
    if artist and counter["artist"][artist] >= 10:
        num_repeats *= min(benchmark / counter["artist"][artist], 10)

    # if img_key in reg_metadata:
    #     return 1

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

    num_repeats = int(num_repeats)
    num_repeats = max(least_num_repeats, num_repeats)

    return num_repeats


def process_caption(
    caption,
    img_md,
    counter,
    fixed_tag_dropout_rate=0,
    flex_tag_dropout_rate=0,
    tags_shuffle_prob=0.0,
    tags_shuffle_rate=1.0,
):
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
    # artist_tag = f"by {artist}" if artist else []
    characters = img_md.get('characters')
    # character_tags = [character for character in characters.split(', ')] if characters else []
    styles = img_md.get('styles')
    # style_tags = [f"{style}" for style in styles.split(', ')] if styles else []
    # fixed_tags = [artist_tag] + character_tags + style_tags

    # flex_tags = [tag for tag in tags if tag not in fixed_tags]
    # if fixed_tag_dropout_rate > 0:
    #     fixed_tags = [tag for tag in fixed_tags if random.random() > fixed_tag_dropout_rate]
    # if flex_tag_dropout_rate > 0:
    #     flex_tags = [tag for tag in flex_tags if random.random() > flex_tag_dropout_rate]

    # if caption_shuffle_prob > 0 and random.random() < caption_shuffle_prob:  # shuffle caption with some probability
    # random.shuffle(flex_tags)

    # dropout artist tags with high frequency
    if artist:
        if random.random() < 0.25:  # 25% chance to drop artist tag
            artist_benchmark = 250
            artist_tag_dropout_rate = max(0, 1 - (artist_benchmark / counter["artist"][artist]))
            if artist_tag_dropout_rate > 0 and random.random() < artist_tag_dropout_rate:  # dropout artist tag
                artist_tag = f"artist: {artist}"
                try:
                    tags.remove(artist_tag)
                except:
                    print(f"artist_tag: {artist_tag} not in tags: {tags}")
            elif styles:
                styles = styles.split(", ")
                for style in styles:
                    try:
                        style_tag = f"style: {style}"
                        tags.remove(style_tag)
                    except:
                        print(f"style_tag: {style_tag} not in tags: {tags}")

    # every fixed tag has a probability to be the first tag
    # the probability is proportional to its weight
    # total probability is 1
    # artist_tags, character_tags, style_tags = [], [], []
    # for tag in fixed_tags:
    #     if tag.startswith("artist:"):
    #         artist_tags.append(tag)
    #     elif tag.startswith("character:"):
    #         character_tags.append(tag)
    #     elif tag.startswith("style:"):
    #         style_tags.append(tag)

    # total_weight = (artist_tag_weight if len(artist_tags) > 0 else 0) + (style_tag_weight if len(style_tags) > 0 else 0) + (character_tag_weight if len(character_tags) > 0 else 0)
    # if total_weight != 0:
    #     artist_tag_weight /= total_weight
    #     style_tag_weight /= total_weight
    #     character_tag_weight /= total_weight

    #     coin = random.random()
    #     if coin < artist_tag_weight:
    #         for tag in artist_tags:
    #             tags.remove(tag)
    #             tags.insert(0, tag)
    #     elif coin < artist_tag_weight + style_tag_weight:
    #         for tag in style_tags:
    #             tags.remove(tag)
    #             tags.insert(0, tag)
    #     else:
    #         for tag in character_tags:
    #             tags.remove(tag)
    #             tags.insert(0, tag)

    tags = [f"by {artist}" if tag.startswith("artist:") else tag[9:] if tag.startswith("character:") else tag[6:] if tag.startswith("style:") else tag for tag in tags]
    caption = ', '.join(tags)
    # caption = caption.replace("artist:", "by ").replace("character:", "").replace("style:", "")
    return caption


def process_nl_caption(
    nl_caption,
    flip_aug=False,
):
    if flip_aug:
        nl_caption = nl_caption.replace("left", "right").replace("right", "left")

    return nl_caption
