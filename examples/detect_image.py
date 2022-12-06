# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from Logger import Logger, Colors, FileHandler


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def main():
    global file
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',
                        help='File path of .tflite file', type=str, default="test_data/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite")
    parser.add_argument('-l', '--labels', help='File path of labels file', type=str, default="test_data/coco_labels.txt")
    parser.add_argument("-t", "--threshold", help="Score threshold for detected objects", type=float, default=0.4)
    parser.add_argument("-c", "--count", help="Number of times to run inference", type=int, default=1)
    parser.add_argument("-d", "--debug", help="Debug output", action="store_true")
    parser.add_argument("-u", "--unsafe", help="Crashes are not caught", action="store_true")
    parser.add_argument("-a", "--amount", help="Limit the amount of images computed", type=int, default=-1)
    parser.add_argument("-s", "--save", help="Save empty images", action="store_true")
    args = parser.parse_args()

    if args.debug:
        lg.level = Logger.Level.DEBUG

    labels = read_label_file(args.labels)
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    lg.info(f"Running model: {args.model.split('.')[0]}")

    # start inference
    while True:
        try:
            if not os.path.exists("./input"):
                os.mkdir("./input")
            new_files = [file for file in os.listdir("./input") if file.endswith(".jpg")]
            lg.info(f"Found {len(new_files)} new files")
            if len(new_files) > 0:
                for index, file in enumerate(new_files):
                    if index > args.amount > 0:
                        break
                    lg.info(f"[{index:2d}/{len(new_files)}] - {Colors.CYAN.value}{file:>}")
                    lg.info(f"{Colors.CYAN.value}{file}")
                    image = Image.open(f"./input/{file}")
                    _, scale = common.set_resized_input(
                        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

                    lg.info('----INFERENCE TIME----')
                    lg.debug('Note: The first inference is slow because it includes loading the model into Edge TPU '
                             'memory.')
                    for i in range(args.count):
                        start = time.perf_counter()
                        interpreter.invoke()
                        inference_time = str((time.perf_counter() - start) * 1000)
                        objs = detect.get_objects(interpreter, args.threshold, scale)

                        lg.info(f"{inference_time.split('.')[0]:>20}ms")

                    lg.info('-------RESULTS--------')

                    if not objs:
                        lg.warning('No objects detected')

                    for obj in objs:
                        lg.info(f"{Colors.BOLD.value}{Colors.GREEN.value}{labels.get(obj.id, obj.id)}")
                        lg.info(f'  id:    {Colors.BOLD.value}{Colors.GREEN.value}{obj.id:>22}')
                        lg.info(f'  score: {Colors.BOLD.value}{Colors.GREEN.value}{obj.score:>22}')
                        # lg.info(f'  bbox:  {Colors.BOLD.value}{obj.bbox}')

                    image = image.convert('RGB')
                    if not objs and args.save:
                        if not os.path.exists("./empty"):
                            os.mkdir("./empty")
                        image.save(f"./empty/{file}")
                    elif objs:
                        if not os.path.exists("./output"):
                            os.mkdir("./output")
                        draw_objects(ImageDraw.Draw(image), objs, labels)
                        image.save(f"./output/{file}")

                    lg.info(f"{Colors.CYAN.value}Deleted the source file for {file}")
                    os.remove(f"./input/{file}")
                    print("\n"*2)

            time.sleep(10)

        except Exception as e:
            lg.error(e)
            if "image file is truncated" in str(e):
                os.remove(f"./input/{file}")
            if args.unsafe:
                raise e


if __name__ == '__main__':
    colors = Logger.get_default_colors()
    colors["INFO"] = ""
    lg = Logger("ObjectDetection", formatter=Logger.minecraft_formatter, level_colors=colors, fh=FileHandler("logs/log.log"))
    main()
