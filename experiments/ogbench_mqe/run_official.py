"""Run pinned MQE impls/main.py without changing its agent, losses or sampler."""
from runtime import prepare
prepare()
import main as official
from absl import app

if __name__=='__main__':app.run(official.main)
