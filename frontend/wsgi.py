from app import app
import waitress

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=5006)
    args = ap.parse_args()

    print(f"Serving on 0.0.0.0:{args.port} (Press CTRL+C to quit)")
    waitress.serve(app, listen=f'0.0.0.0:{args.port}', url_scheme='https', threads=6)