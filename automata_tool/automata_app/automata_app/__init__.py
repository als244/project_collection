from flask import Flask

app = Flask(__name__)

app.secret_key = b'fsd90fjel$&#Ndsff2/xvds'

import automata_app.views