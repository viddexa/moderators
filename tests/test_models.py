# demo.py
from moderators.api import Moderator

def main():
    # Provide a local image file path here
    image_path = "/Users/sukrukirman/aky/moderators-project/tests/Ekran Resmi 2025-08-04 10.37.08.png"

    # suko/nsfw: config.json may be missing; metadata inference will select an integration if possible
    model = Moderator.from_pretrained("suko/nsfw")
    results = model.predict(image_path)

    for res in results:
        print(res.classifications)

if __name__ == "__main__":
    main()
