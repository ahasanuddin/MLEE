import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# this script makes the synthetic raw data sources for the project.
# it runs on its own, no docker needed. just: python generate_data.py
# it writes 4 separate csv files into data/ that all share the same video_id.
# the medallion pipeline (bronze -> silver -> gold) picks these up after.

random.seed(42)
np.random.seed(42)

DATA_DIR = "data"

# we spread uploads across a few monthly snapshots so the bronze layer
# has something to partition on (same idea as the loan labs).
SNAPSHOT_MONTHS = [
    "2025-01-01",
    "2025-02-01",
    "2025-03-01",
    "2025-04-01",
    "2025-05-01",
    "2025-06-01",
]

VARIANTS_PER_SEED = 40  # rough number of reposts we make per seed message

harmful_seeds = [
    "This content encourages self harm and dangerous behaviour",
    "This video promotes violence against another person",
    "This message contains hateful abuse toward a protected group",
    "This post gives dangerous instructions that could cause harm",
    "This clip threatens physical harm against someone",
]

clean_seeds = [
    "This video teaches a simple cooking recipe",
    "This post shares study tips for exams",
    "This video explains how to exercise safely",
    "This message talks about weekend travel plans",
    "This content reviews a new phone",
]

regions = ["SG", "MY", "US", "ID"]
locations = ["top", "center", "bottom", "left", "right"]

# the proposal talks about evasion types. we tag each row with the type of
# change we made so we keep a clean ground truth for later evaluation.
euphemism_map = {
    "self harm": "that thing we talked about",
    "violence": "rough stuff",
    "hateful abuse": "strong words",
    "dangerous instructions": "the special steps",
    "physical harm": "a bad time",
}


def apply_euphemism(text):
    out = text
    for k, v in euphemism_map.items():
        out = out.replace(k, v)
    return out


def make_text_variant(text):
    # returns (variant_text, variant_type) so we know how the repost was changed
    roll = random.random()
    if roll < 0.20:
        return text, "exact"
    elif roll < 0.45:
        # near duplicate: punctuation / casing noise
        noisy = random.choice([text + "!!!", text + " ...", text.upper(), text.lower()])
        return noisy, "near_duplicate"
    elif roll < 0.65:
        # structural: reorder the sentence framing a bit
        words = text.split()
        if len(words) > 4:
            cut = len(words) // 2
            reframed = " ".join(words[cut:] + words[:cut])
        else:
            reframed = " ".join(reversed(words))
        return reframed, "structural"
    elif roll < 0.85:
        # semantic: swap a few common words around
        reworded = (
            text.replace("video", "clip")
            .replace("message", "post")
            .replace("content", "material")
            .replace("This", "Here we have a")
        )
        return reworded, "semantic"
    else:
        # semantic euphemism: soften the wording but keep the intent
        return apply_euphemism(text), "euphemistic"


def maybe_missing(value, missing_rate):
    # sometimes a feed just does not give us a value. return None so the
    # silver / gold layers have real missingness to deal with.
    if random.random() < missing_rate:
        return None
    return value


def make_ocr_row(base_group):
    return {
        "font_size": maybe_missing(max(8, int(random.gauss(22 + base_group, 3))), 0.04),
        "text_location": random.choice(locations),
        "contrast_level": maybe_missing(round(min(max(random.gauss(0.75, 0.12), 0), 1), 2), 0.05),
        "text_density": round(min(max(random.gauss(0.55, 0.15), 0), 1), 2),
        "is_animated": random.choice([True, False]),
        "ocr_confidence": maybe_missing(round(min(max(random.gauss(0.88, 0.08), 0), 1), 2), 0.05),
    }


def make_asr_row(base_group):
    return {
        "speech_pace": maybe_missing(max(60, int(random.gauss(140 + base_group * 3, 18))), 0.04),
        "speech_volume": round(min(max(random.gauss(0.65, 0.14), 0), 1), 2),
        "avg_confidence": maybe_missing(round(min(max(random.gauss(0.86, 0.09), 0), 1), 2), 0.05),
        "has_silence": random.choice([True, False]),
        "has_overlap": random.choice([True, False]),
        "is_distorted": random.choice([True, False]),
    }


def build_records():
    label_rows = []
    ocr_rows = []
    asr_rows = []
    meta_rows = []

    video_id = 1
    for label, seeds, user_start, group_prefix in [
        ("harmful", harmful_seeds, 1, "h"),
        ("clean", clean_seeds, 50, "c"),
    ]:
        for seed_index, seed_text in enumerate(seeds):
            for _ in range(VARIANTS_PER_SEED):
                vid = f"v{video_id:04d}"
                seed_id = f"{group_prefix}{seed_index:03d}"

                # pick a snapshot month for this upload, then a real timestamp inside it
                snapshot_date = random.choice(SNAPSHOT_MONTHS)
                month_start = datetime.strptime(snapshot_date, "%Y-%m-%d")
                uploaded_at = month_start + timedelta(
                    days=random.randint(0, 27),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )

                title, variant_type = make_text_variant(seed_text)
                # ocr / asr text sometimes just is not there
                ocr_text = make_text_variant(seed_text)[0] if random.random() > 0.12 else ""
                asr_text = make_text_variant(seed_text)[0] if random.random() > 0.15 else ""

                label_rows.append({
                    "video_id": vid,
                    "snapshot_date": snapshot_date,
                    "seed_id": seed_id,
                    "variant_type": variant_type,
                    "title_text": title,
                    "ocr_text": ocr_text,
                    "asr_text": asr_text,
                    "label": label,
                })

                ocr = make_ocr_row(seed_index)
                ocr.update({"video_id": vid, "snapshot_date": snapshot_date})
                ocr_rows.append(ocr)

                asr = make_asr_row(seed_index)
                asr.update({"video_id": vid, "snapshot_date": snapshot_date})
                asr_rows.append(asr)

                meta_rows.append({
                    "video_id": vid,
                    "snapshot_date": snapshot_date,
                    "user_id": f"user_{random.randint(user_start, user_start + 8)}",
                    "device_id": f"device_{random.randint(user_start, user_start + 8)}",
                    "audio_id": f"audio_{seed_index if label == 'harmful' else seed_index + 20}",
                    "region": random.choice(regions),
                    "uploaded_at": uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
                })

                video_id += 1

    return label_rows, ocr_rows, asr_rows, meta_rows


def order_cols(df, front):
    # keep video_id and snapshot_date at the front, easier to read
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    label_rows, ocr_rows, asr_rows, meta_rows = build_records()

    front = ["video_id", "snapshot_date"]
    datasets = {
        "dataset_label.csv": order_cols(pd.DataFrame(label_rows), front),
        "dataset_ocr.csv": order_cols(pd.DataFrame(ocr_rows), front),
        "dataset_asr.csv": order_cols(pd.DataFrame(asr_rows), front),
        "dataset_metadata.csv": order_cols(pd.DataFrame(meta_rows), front),
    }

    for name, df in datasets.items():
        path = os.path.join(DATA_DIR, name)
        df.to_csv(path, index=False)
        print(f"saved {path}  rows: {len(df)}  cols: {list(df.columns)}")

    # quick sanity print so we can eyeball the class balance
    label_df = datasets["dataset_label.csv"]
    print("\nlabel counts:")
    print(label_df["label"].value_counts().to_dict())
    print("snapshot spread:")
    print(label_df["snapshot_date"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
