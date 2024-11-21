from waitress import serve
from app import app


# open frontend in new tab and start api server
if __name__ == "__main__":
    print("Starting server, open the frontend's index.html manually")
    serve(app, host="0.0.0.0", port=8922, clear_untrusted_proxy_headers=False)