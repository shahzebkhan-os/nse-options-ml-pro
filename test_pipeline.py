from src.pipeline.predictor import VolatilityPredictor

print("Initializing Predictor...")
p = VolatilityPredictor()

print("Training (Simulated)...")
p.train("RELIANCE")

print("Predicting RELIANCE...")
res = p.predict("RELIANCE")
print(res)

print("Predicting TCS...")
res2 = p.predict("TCS")
print(res2)
