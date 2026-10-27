"""Execute the actual upstream impls/main.py, with storage/offline adapters."""
from runtime import prepare

prepare()
import main as official_main
from absl import app

if __name__ == '__main__':
    app.run(official_main.main)
