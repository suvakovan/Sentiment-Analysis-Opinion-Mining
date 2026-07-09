"""
Mock Data Generator — creates realistic synthetic Amazon reviews.
"""

import pandas as pd
import os
import random
import string

# Sentence templates for more realistic synthetic reviews
POSITIVE_TEMPLATES = [
    "I absolutely love this {noun}. {adj} quality and fast {action}.",
    "This is the best {noun} I have ever purchased. Highly recommend!",
    "Great {noun}! The {feature} is {adj} and it works perfectly.",
    "{adj} product. Exactly what I needed. Will buy again.",
    "Five stars! This {noun} exceeded my expectations. Very {adj}.",
    "Impressed with the {feature}. {adj} build and the {action} was quick.",
    "My family loves this {noun}. The {feature} is outstanding.",
    "Perfect {noun} for the price. {adj} value and excellent {feature}.",
    "Amazing {noun}! Not a single complaint. The {feature} is top notch.",
    "So happy with this purchase. The {noun} is {adj} and well made.",
]

NEGATIVE_TEMPLATES = [
    "Terrible {noun}. The {feature} broke after one day. Very disappointed.",
    "Worst purchase ever. This {noun} is completely useless.",
    "Do not buy this {noun}. Poor {feature} and awful {action}.",
    "I hate this {noun}. It stopped working immediately. Total waste.",
    "The {feature} is horrible. Returned this {noun} right away.",
    "Very poor quality {noun}. The {feature} fell apart within a week.",
    "Not worth the money. This {noun} is cheaply made and {adj}.",
    "Disappointing. The {noun} does not match the description at all.",
    "Broken on arrival. The {feature} was damaged and {action} took forever.",
    "One star. This {noun} is the worst I have owned. Do not recommend.",
]

NOUNS = ["product", "item", "device", "gadget", "tool", "unit", "purchase", "accessory"]
POS_ADJS = ["excellent", "amazing", "fantastic", "wonderful", "great", "superb", "solid", "premium"]
NEG_ADJS = ["terrible", "awful", "horrible", "flimsy", "defective", "cheap", "broken", "useless"]
FEATURES = ["quality", "design", "battery", "material", "screen", "build", "packaging", "finish"]
ACTIONS = ["shipping", "delivery", "setup", "installation", "assembly"]


def _fill_template(template, adj_pool):
    return template.format(
        noun=random.choice(NOUNS),
        adj=random.choice(adj_pool),
        feature=random.choice(FEATURES),
        action=random.choice(ACTIONS),
    )


def generate_mock_dataset(num_samples=10000, output_path='data/raw_reviews.csv',
                          seed=42):
    """Generate a mock Amazon reviews dataset.

    Args:
        num_samples: Number of reviews to generate.
        output_path: CSV output path (relative or absolute).
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    print(f"Generating {num_samples} mock reviews (seed={seed}) ...")

    data = []
    for i in range(num_samples):
        roll = random.random()

        if roll > 0.4:                          # ~60 % positive
            rating = random.choice([4, 5])
            text = _fill_template(random.choice(POSITIVE_TEMPLATES), POS_ADJS)
            title_words = random.sample(POS_ADJS, 2)
        elif roll > 0.1:                        # ~30 % negative
            rating = random.choice([1, 2])
            text = _fill_template(random.choice(NEGATIVE_TEMPLATES), NEG_ADJS)
            title_words = random.sample(NEG_ADJS, 2)
        else:                                   # ~10 % neutral
            rating = 3
            text = _fill_template(random.choice(POSITIVE_TEMPLATES), NEG_ADJS)
            title_words = ["okay", "average"]

        # Occasionally duplicate the text to simulate longer reviews
        if random.random() > 0.6:
            text += " " + _fill_template(random.choice(
                POSITIVE_TEMPLATES if rating >= 4 else NEGATIVE_TEMPLATES
            ), POS_ADJS if rating >= 4 else NEG_ADJS)

        data.append({
            'review_id': f"R{i:06d}",
            'rating': rating,
            'title': " ".join(title_words).title(),
            'text': text,
        })

    df = pd.DataFrame(data)

    # Ensure output directory exists (handles both relative and absolute paths)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} reviews to {output_path}")
    print("\nRating distribution:")
    print(df['rating'].value_counts().sort_index())


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == 'src':
        os.chdir('..')
    generate_mock_dataset()
