"""
Streamlit Community Cloud entrypoint.
Points directly to the TripSense application (app.py).
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("app.py", run_name="__main__")
