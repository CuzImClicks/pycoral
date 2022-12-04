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

from Logger import Logger


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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',
                        help='File path of .tflite file', type=str, default="test_data/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite")
    parser.add_argument('-l', '--labels', help='File path of labels file', type=str, default="test_data/coco_labels.txt")
    parser.add_argument("-t", "--threshold", help="Score threshold for detected objects", type=float, default=0.4)
    parser.add_argument("-c", "--count", help="Number of times to run inference", type=int, default=10)
    parser.add_argument("-d", "--debug", help="Debug output", type=bool, default=False)
    args = parser.parse_args()

    if args.debug:
        lg.level = Logger.Level.DEBUG

    labels = read_label_file(args.labels)
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    # start inference
    while True:
        try:
            new_files = [file for file in os.listdir("./input") if file.endswith(".jpg")]
            lg.info(f"Found {len(new_files)} new files")
            if len(new_files) > 0:
                for index, file in enumerate(new_files):
                    image = Image.open(f"./input/{file}")
                    _, scale = common.set_resized_input(
                        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

                    lg.info('----INFERENCE TIME----')
                    lg.debug('Note: The first inference is slow because it includes loading the model into Edge TPU '
                             'memory.')
                    for _ in range(args.count):
                        start = time.perf_counter()
                        interpreter.invoke()
                        inference_time = time.perf_counter() - start
                        objs = detect.get_objects(interpreter, args.threshold, scale)
                        lg.info('%.2f ms' % (inference_time * 1000))

                    lg.info('-------RESULTS--------')
                    if not objs:
                        lg.warning('No objects detected')

                    for obj in objs:
                        lg.info(labels.get(obj.id, obj.id))
                        lg.info(f'  id:    {obj.id}')
                        lg.info(f'  score: {obj.score}')
                        lg.info(f'  bbox:  {obj.bbox}')

                    image = image.convert('RGB')
                    draw_objects(ImageDraw.Draw(image), objs, labels)
                    image.save(f"./output/{file}")

                    lg.info(f"Deleted the source file for {file}")
                    os.remove(f"./input/{file}")
                    lg.info("\n"*2)

            time.sleep(10)

        except Exception as e:
            lg.error(e)


if __name__ == '__main__':
    lg = Logger("ObjectDetection", formatter=Logger.minecraft_formatter)
    main()
