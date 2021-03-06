import colorsys
import imghdr
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO

from flask import Flask, Response

from yad2k import ObjectDetector

from camera import UsbCamera

class InferenceWorker:
    def __init__(self, model_path, anchors_path, classes_path, font_path,
            score_threshold=.3, iou_threshold=.5, use_camera=False):
        model_path = os.path.expanduser(model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        anchors_path = os.path.expanduser(anchors_path)
        classes_path = os.path.expanduser(classes_path)

        self.detector = ObjectDetector(model_path, anchors_path, classes_path, score_threshold, iou_threshold)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        self.colors = self._generate_colors(self.detector.class_names)

        self.font_path = font_path

        if use_camera:
            self._camera = UsbCamera()
            self._camera.start()
        else:
            self._camera = None

    def process_camera(self):
        if self._camera is None:
            raise Exception("The server is not configured for camera input")
        image_data = self._camera.read()
        if image_data is None:
            return None
        image = Image.fromarray(image_data)
        return self._process(image)

    def process_file(self, image_filepath):
        try:
            image_type = imghdr.what(image_filepath)
            if not image_type:
                print('Not an image')
                return
                # TODO: improve here
        except IsADirectoryError:
            print('IsADirectoryError exception')
            return
            # TODO: improve here

        image = Image.open(image_filepath)

        return self._process(image)

    def _process(self, image):
        objects = self.detector.detect(image)
        print('Found {} objects'.format(len(objects)))

        font = ImageFont.truetype(
            font=self.font_path,
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, object in reversed(list(enumerate(objects))):
            label = '{} {:.2f}'.format(object.class_name, object.score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            print(label, (object.left, object.top), (object.right, object.bottom))

            if object.top - label_size[1] >= 0:
                text_origin = np.array([object.left, object.top - label_size[1]])
            else:
                text_origin = np.array([object.left, object.top + 1])

            color = self.colors[object.class_name]

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle(
                    [object.left + j, object.top + j, object.right - j, object.bottom - j],
                    outline=color)

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color)
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        with BytesIO() as output:
            image.save(output, 'jpeg', quality=90)
            return output.getvalue()

    def close(self):
        self.detector.close()
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()

    def _generate_colors(self, class_names):
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        colors = {class_names[i]: v for i, v in enumerate(colors)}
        return colors


app = Flask(__name__)

@app.route('/')
def homepage():
    #message = '<html><head><script type="application/javascript">alert(1);console.write("000");document.addEventListener("DOMContentLoaded", function(event) { console.write("111"); setInterval(1000, function() { console.write("222"); document.getElementById("image").src = "/image?cache=" + Math.floor(Math.random() * 1000000) }) });</script></head><body><p>Hello friend!</p><p><img id="image" src="/image" /></p></body></html>'
    message = '''<html><head><script type="application/javascript">
    document.addEventListener("DOMContentLoaded",
        function(event) {
            setInterval(function() {
                    document.getElementById("image").src = "/image?cache=" + Math.floor(Math.random() * 1000000)
                }, 1000)
        });
    </script></head><body><p>Hello friend!</p><p><img id="image" src="/image" /></p></body></html>'''
    return message

@app.route('/image')
def image():
    bytes = worker.process_file(INPUT_PATH)
    return Response(bytes, mimetype='image/jpeg')

@app.route('/camera')
def camera():
    bytes = worker.process_camera()
    return Response(bytes, mimetype='image/jpeg')

if __name__ == "__main__":
    MODEL_PATH = '../model_data/udacity_object_model.h5'
    ANCHORS_PATH = '../model_data/yolo_anchors.txt'
    CLASSES_PATH = '../model_data/udacity_object_dataset_classes.txt'
    INPUT_PATH = '../temp/images/in.jpg'
    FONT_PATH = '../resources/font/FiraMono-Medium.otf'

    worker = InferenceWorker(MODEL_PATH, ANCHORS_PATH, CLASSES_PATH, FONT_PATH, use_camera=True)

    app.run(host='0.0.0.0', port=5000)
