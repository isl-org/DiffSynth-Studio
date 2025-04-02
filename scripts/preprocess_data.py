import argparse
import os
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
)
from decord import VideoReader, cpu
from PIL import Image
import torch

import pandas as pd


def main(args):
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # model_name = "liuhaotian/llava-v1.5-7b"
    # llava_processor = AutoProcessor.from_pretrained(model_name)
    # llava_model = AutoModelForVision2Seq.from_pretrained(model_name)

    model_name = "microsoft/git-large-textcaps"
    video_processor = AutoProcessor.from_pretrained(model_name)
    video_model = AutoModelForVision2Seq.from_pretrained(model_name)

    data_dir = args.data_dir
    data = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            if file_path.endswith(".jpeg"):
                image = Image.open(file_path).convert("RGB")
                inputs = processor(image, return_tensors="pt")
                with torch.no_grad():
                    caption_ids = model.generate(**inputs)
                    caption = processor.batch_decode(
                        caption_ids, skip_special_tokens=True
                    )[0]
                    print("Blip Caption:", caption)
                # inputs = llava_processor(image, return_tensors="pt")
                # with torch.no_grad():
                #     output = llava_model.generate(**inputs)
                #     caption = llava_processor.decode(
                #         output[0], skip_special_tokens=True
                #     )
                #     print("LLava Caption", caption)
                data.append({"file_name": file_name, "text": caption})
            elif file_path.endswith(".mp4"):
                video_reader = VideoReader(file_path, ctx=cpu(0))
                frames = [
                    video_reader[i].asnumpy() for i in range(0, len(video_reader), 10)
                ]
                # print(help(video_processor))
                print("frames", len(frames))
                print("type", type(frames[0]))
                inputs = video_processor(images=frames, return_tensors="pt")
                with torch.no_grad():
                    output = video_model.generate(**inputs)
                caption = video_processor.batch_decode(
                    output, skip_special_tokens=True
                )[0]
                print("Video caption:", caption)
                data.append({"file_name": file_name, "text": caption})
    df = pd.DataFrame(data, columns=["file_name", "text"])
    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    df.to_csv(metadata_path, index=False, encoding="utf-8")
    print(f"Save {metadata_path} successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/export/share/projects/videogen/data/example_dataset/diffsynth_format/train",
        help="The directory that contains videos and images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/export/share/projects/videogen/data/example_dataset/diffsynth_format/",
        help="output directory of the output csv file.",
    )
    args = parser.parse_args()
    main(args)
