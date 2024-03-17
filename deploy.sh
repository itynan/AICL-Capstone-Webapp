#deploy shell
source ~/myenv/bin/activate
pip install -r requirements.txt

# Start your Dash applications
python dash-ml-app.py &
python webflow-fe-server.py &