from argparse import ArgumentParser
import tensorflow as tf

from mobilenetv3_factory import build_mobilenetv3
from dataload import _parse_image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

def main(args):
    model = build_mobilenetv3(
        args.model_type,
        input_shape=(224, 224, 3),
        num_classes=3,
        width_multiplier=1,
        l2_reg=1e-5,
    )
    model.load_weights(args.model_path, by_name=True)
    model.summary()
    img, _= _parse_image(args.image_path, 1)
    img = tf.expand_dims(img, 0)
    result = model.predict(img, batch_size=1, steps=1)
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

    args = parser.parse_args()
    main(args)