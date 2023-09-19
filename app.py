from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import shutil

app = Flask(__name__)

def run_scripts():
    try:
        subprocess.run(['python', 'scheduler.py'], check=True)
        subprocess.run(['python','pipeline.py'])
        print("Scripts executed successfully.")
        shutil.rmtree("dataset")
    except subprocess.CalledProcessError as e:
        print(f"Error executing scripts: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(run_scripts, 'interval', minutes=0.2)  # Adjust the interval as needed
scheduler.start()


if __name__ == '__main__':
    app.run(debug=False)
