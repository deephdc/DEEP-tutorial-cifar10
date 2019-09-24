
# -*- coding: utf-8 -*-
"""
Model description
"""

import base64
import matplotlib
import numpy
import urllib.request
import argparse
import pkg_resources

# import project's config.py
import cifar10.config as cfg

from keras.preprocessing import image
from cifar10.models.cifar10_model  import train_nn, predict_nn

def get_metadata():
    """
    Function to read metadata
    """

    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': "Lara Lloret Iglesias",
        'Author-email': None,
        'License': None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par+":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def predict_file(*args):
    """
    Function to make prediction on a local file
    """


def predict_data(*args):
    """
    Function to make prediction on an uploaded file
    """
    outputpath=args[0]["outputpath"]
    thefile= args[0]['files'][0]
    thename= thefile.filename
    thepath= outputpath + "/" +thename
    thefile.save(thepath)
    img = image.load_img(thepath, target_size=(32,32))
    x= image.img_to_array(img)
    message=predict_nn(x,outputpath)
    return message



def predict_url(*args):
    """
    Function to make prediction on a URL
    """
    print("aqui estoy imprimiendo args : ", args)
    urllib.request.urlretrieve(args[0], 'image.jpg')
    message = 'Not implemented in the model (predict_url)'
    return message


###
# Uncomment the following two lines
# if you allow only authorized people to do training
###
#import flaat
#@flaat.login_required()
def train(train_args):
    """
    Train network
    train_args : dict
        Json dict with the user's configuration parameters.
        Can be loaded with json.loads() or with yaml.safe_load()    
    """
    train_nn(train_args['epochs'], train_args['lrate'],train_args['outputpath'])
   
    run_results = { "status": "SUCCESS",
                    "train_args": [],
                    "training": [],
                  }

    run_results["train_args"].append(train_args)

    print(run_results)
    return run_results


def get_train_args():
    """
    Returns a dict of dicts to feed the deepaas API parser
    """
    train_args = cfg.train_args

    # convert default values and possible 'choices' into strings
    for key, val in train_args.items():
       val['default'] = str(val['default']) #yaml.safe_dump(val['default']) #json.dumps(val['default'])
       if 'choices' in val:
           val['choices'] = [str(item) for item in val['choices']]

    return train_args

# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
def get_test_args():
    predict_args = cfg.predict_args
    # convert default values and possible 'choices' into strings
    for key, val in predict_args.items():
        val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]
        print(val['default'], type(val['default']))

    return predict_args

# during development it might be practical 
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """
    if args.method == 'get_metadata':
        get_metadata()
    elif args.method == 'train':
        train(args)
    else:
        get_metadata()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')

    # get arguments configured for get_train_args()
    train_args = get_train_args()
    for key, val in train_args.items():
        parser.add_argument('--%s' % key,
                            default=val['default'],
                            type=type(val['default']),
                            help=val['help'])

    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    args = parser.parse_args()

    main()
