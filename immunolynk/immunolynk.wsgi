#immunolynk.wsgi
import sys
sys.path.insert(0, '/var/www/html/immunolynk')

from server import server as application