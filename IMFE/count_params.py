from src.imfe import IMFE, BatchedIMFE

model = IMFE()
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")