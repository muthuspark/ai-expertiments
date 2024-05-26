import os
import random
import string

label_color_map = {}


def generate_random_name(length=8, prefix='', suffix=''):
    """
    Generate a random name with a specified length.

    Args:
        length (int): The desired length of the name (default: 8).
        prefix (str): A prefix to add before the random name (default: '').
        suffix (str): A suffix to add after the random name (default: '').

    Returns:
        str: A random name with the specified length, prefix, and suffix.
    """
    # Generate a random string of letters and digits
    characters = string.ascii_letters + string.digits
    random_chars = ''.join(random.choice(characters) for _ in range(length))

    # Combine the prefix, random string, and suffix
    random_name = prefix + random_chars + suffix

    return random_name


def get_file_name(file_path):
    filename = os.path.basename(file_path)

    # Extract the name without extension (assuming extension starts with '.')
    name_without_extension = filename.split('.')[0]
    return name_without_extension


def get_unique_color(label):
    global label_color_map
    if label not in label_color_map:
        label_color_map[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return label_color_map[label]
