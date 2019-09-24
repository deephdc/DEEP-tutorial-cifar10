# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))


# Training and predict(deepaas>=0.5.0) arguments as a dict of dicts 
# with the following structure to feed the deepaas API parser:
# (see also get_train_args() )
# { 'arg1' : {'default': 1,       # default value
#             'help': '',         # can be an empty string
#             'required': False   # bool
#             },
#   'arg2' : {'default': 'value1',
#             'choices': ['value1', 'value2', 'value3'],
#             'help': 'multi-choice argument',
#             'required': False
#             },
#   'arg3' : {...
#             },
# ...
# }
train_args = { 'epochs': {'default': 1,
                        'help': 'Number of epochs',
                        'required': True,
                        'type': int
                        },

               'lrate': {'default':0.001,
                         'help': 'Initial learning rate value',
                         'required': True,
                         'type': float
                        },

               'outputpath': {'default': "/tmp",
                         'help': 'Path for saving the model',
                         'required': True
                        },
}

predict_args = { 'outputpath': {'default': "/tmp",
                          'help': 'Path for loading the model',
                          'required': True
                         },
}
