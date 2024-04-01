#!/bin/bash
#aicyberlabs-dir = /var/www/aicyberlabs-app
#dash-dir = /var/www/aicyberlabs-app/dash
#tmp-dir = /tmp

#test10
sudo pgrep -f "gunicorn" | sudo xargs kill -9

echo "Creating and changing ownership of /var/www/aicyberlabs-app/myenv"
sudo mkdir -v -p /var/www/aicyberlabs-app/myenv
sudo chown -R ec2-user:ec2-user /var/www/aicyberlabs-app/myenv

echo "Deleting old app"
sudo rm -rf /var/www/aicyberlabs-app

echo "Creating app folder"
sudo mkdir -p /var/www/aicyberlabs-app

echo "Moving files to app folder"
sudo mv * /var/www/aicyberlabs-app

cd /var/www/aicyberlabs-app/
sudo mv env .env

#sudo amazon-linux-extras install python3.9
cd ~
python3.9 -m venv myenv_new
source myenv_new/bin/activate

echo "Installing application dependencies from requirements.txt"
cd /var/www/aicyberlabs-app/
pip install gunicorn
pip install -v -r requirements.txt

echo "Restarting Nginx"
sudo systemctl restart nginx

#
#echo "Starting Gunicorn for webflow-fe-server"
#cd webflow_template/


echo "Starting Gunicorn for dash-ml-app"
cd dash/

#gunicorn -c gunicorn_config_port_5000.py webflow-fe-server:app --workers 4 &
#gunicorn -c gunicorn_config_port_7001.py dash-ml-app:server --workers 4 &
#sudo gunicorn -c conf_port_7002.py dash-ml-app-f1-score:server --workers 8 &

echo "Deployment completed successfully"

sleep 45


sudo cp /tmp/test_deploy2.sh /var/www/aicyberlabs-app/
sudo sh -xv /var/www/aicyberlabs-app/test_deploy2.sh


# Update and install Nginx if not already installed
#if ! command -v nginx > /dev/null; then
#    echo "Installing Nginx"
#    sudo apt-get update
#    sudo apt-get install -y nginx
#fi

# Configure Nginx to act as a reverse proxy if not already configured
#if [ ! -f /etc/nginx/sites-available/myapp ]; then
#    sudo rm -f /etc/nginx/sites-enabled/default
#    sudo bash -c 'cat > /etc/nginx/sites-available/myapp <<EOF
#server {
#    listen 80;
#    server_name _;
#
#    location / {
#        include proxy_params;
#        proxy_pass http://unix:/var/www/langchain-app/myapp.sock;
#    }
#}
#EOF'
#
#    sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled
#    sudo systemctl restart nginx
#else
#    echo "Nginx reverse proxy configuration already exists."
#fi
#
## Stop any existing Gunicorn process
#sudo pkill gunicorn
#sudo rm -rf myapp.sock
#
## # Start Gunicorn with the Flask application
## # Replace 'server:app' with 'yourfile:app' if your Flask instance is named differently.
## # gunicorn --workers 3 --bind 0.0.0.0:8000 server:app &
#echo "starting gunicorn"
#sudo gunicorn --workers 3 --bind unix:myapp.sock  server:app --user www-data --group www-data --daemon
#echo "started gunicorn "

#TODO: NEED TO CONFIG GUNICORN & AND PYTHON VENV