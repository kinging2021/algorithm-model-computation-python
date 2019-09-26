import os
config_name = os.environ.get('CONFIG_NAME') or 'development'

if config_name == 'production':
    from .config_prod import *
elif config_name == 'test':
    from .config_test import *
else:  # development
    from .config_dev import *
