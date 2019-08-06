from argparse import ArgumentParser
import tensorflow as tf

from mobilenetv3_factory import build_mobilenetv3
from dataload import _parse_image
import numpy as np
from PIL import Image
from collections import defaultdict
from keras import backend as K
import cv2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

def iter_occlusion(image, size=8):
    # taken from https://www.kaggle.com/blargl/simple-occlusion-and-saliency-maps

   occlusion = np.full((size * 5, size * 5, 1), [0.5], np.float32)
   occlusion_center = np.full((size, size, 1), [0.5], np.float32)
   occlusion_padding = size * 2

   print('padding...')
   image_padded = np.pad(image, ( \
   (occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0) \
   ), 'constant', constant_values = 0.0)

   for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):

       for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
           tmp = image_padded.copy()

           tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
             x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
             = occlusion
        

           tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center

           yield x - occlusion_padding, y - occlusion_padding, \
             tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]
def heatmap_generate(data, img_size, model):
    occlusion_size = 8
    correct_class = 2
    print('occluding...')
    heatmap = np.zeros((img_size, img_size), np.float32)
    class_pixels = np.zeros((img_size, img_size), np.int16)
    counters = defaultdict(int)

    for n, (x, y, img_float) in enumerate(iter_occlusion(data, size=occlusion_size)):
        X = img_float.reshape(1,224, 224, 3)
        out = model.predict(X)

        # print('#{}: {} @ {} (correct class: {})'.format(n, np.argmax(out), np.amax(out), out[0][correct_class]))
        # print('x {} - {} | y {} - {}'.format(x, x + occlusion_size, y, y + occlusion_size))

        heatmap[y:y + occlusion_size, x:x + occlusion_size] = out[0][correct_class]
        class_pixels[y:y + occlusion_size, x:x + occlusion_size] = np.argmax(out)
        counters[np.argmax(out)] += 1

    return heatmap, class_pixels

def visualize_feuturemap(model, img):
    for layer in model.get_layer('MobileNetV3_Small').layers:
        print(layer.name + ' featuremap saving...')
        # layer_middle = tf.keras.Model(inputs = model.input,\
        #     outputs = model.get_layer(layer.name).output)
        layer_middle = tf.keras.Model(inputs = model.input,\
            outputs = layer.output)
        layer_outputs = layer_middle.predict(img, batch_size=1, steps=1)
        if layer.name == 'LastStage':
            break
        if layer.name == 'FirstLayer':
            channels = 16
        if layer.name == 'Bneck':
            channels = 96
        # layer_output=layer_outputs[0,:,:,5]
        for i in range(0, channels):
            layer_output = layer_outputs[0,:,:,i]
            layer_max = np.max(layer_output)
            layer_output = layer_output.astype("float32")/layer_max*255
            layer_output = np.asarray(layer_output)
            im = Image.fromarray(layer_output)
            im = im.resize((224,224))
            if im.mode != 'RGB':
                im = im.convert('RGB')
                im.save("featuremaps/" + layer.name + str(i) + ".jpeg")

def visualize_heatmap(model, img):
    print("heatmap saving...")
    image = sess.run(img)
    model_heatmap = tf.keras.Model(inputs=model.input, outputs=model.output)
    heatmap, class_pixels = heatmap_generate(image, 224, model)
    heatmap_max = np.max(heatmap)
    heatmap = heatmap.astype("float32")/heatmap_max*255
    heatmap=heatmap.astype(np.uint8)
    heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    cv2.imwrite('heatMap.jpg', heatmap)
    print("Finish")

def save_image(img, name):
    img = np.asarray(img)
    im = Image.fromarray(img)
    if im.mode != 'RGB':
        im = im.convert('RGB')
        im.save(name + ".jpeg")

_available_optimizers = {
    "rmsprop": tf.train.RMSPropOptimizer,
    "adam": tf.train.AdamOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    }

def main(args):

    # load image
    img, _= _parse_image(args.image_path, 1)
    img_dims = tf.expand_dims(img, 0)

    
    # build model
    model = build_mobilenetv3(
        args.model_type,
        input_shape=(224, 224, 3),
        num_classes=3,
        width_multiplier=1,
        l2_reg=1e-5,
    )
    
    model.load_weights(args.model_path, by_name=True)
    model.summary()

    model.compile(
        optimizer=_available_optimizers.get('sgd')(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # visualize featuremap
    if args.featuremap:
        visualize_feuturemap(model, img_dims)

    # heat map
    if args.heatmap:
        visualize_heatmap(model, img)
    
    # visualize layers
    # for layer in model.get_layer('MobileNetV3_Small').layers:
    #     print(layer)
    
    result = model.predict(img_dims, batch_size=1, steps=1)
    # print(result)
    print('Result:')
    print('probability of shipping : ' + str(result[0][0]))
    print('probability of nutrition : ' + str(result[0][1]))
    print('probability of special handling : ' + str(result[0][2]))

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model_path", type=str, default="mobilenetv3_small_printer_30.h5")
    parser.add_argument("--model_type", type=str, default="small", choices=["small", "large"])
    # test image
    parser.add_argument("--image_path", type=str, default="test/1.jpg")
    # visualization
    parser.add_argument("--featuremap", type=bool, default=False)
    parser.add_argument("--heatmap", type=bool, default=False)

    args = parser.parse_args()
    main(args)
