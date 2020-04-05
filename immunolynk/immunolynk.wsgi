WSGIDaemonProcess immunolynk threads=5
WSGIScriptAlias / /var/www/html/immunolynk.wsgi

<Directory immunolynk>
    WSGIProcessGroup immunolynk
    WSGIApplicationGroup %{GLOBAL}
    Order deny,allow
    Allow from all
</Directory>