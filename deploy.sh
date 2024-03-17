#deploy shell
source ~/myenv/bin/activate
pip install -r requirements.txt

# Start your Dash apps
python dash-ml-app.py &
python webflow-fe-server.py &