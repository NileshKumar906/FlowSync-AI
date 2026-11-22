import torch

MODEL_PATHS = {
    "DQN v1": "models/dqn_traffic_emergency.pth",
    "DQN v2": "models/dqn_traffic_emergency_v2.pth",
    "Double DQN": "models/double_dqn_traffic_emergency.pth",
}

models = {}

for name, path in MODEL_PATHS.items():
    state = torch.load(path, map_location="cpu", weights_only=True)
    models[name] = state

    print(f"\n{name}")
    print("Parameters:", len(state))

for name_a, name_b in [
    ("DQN v1", "DQN v2"),
    ("DQN v2", "Double DQN"),
    ("DQN v1", "Double DQN"),
]:
    a = models[name_a]
    b = models[name_b]

    identical = all(
        torch.equal(a[key], b[key])
        for key in a
    )

    total_difference = sum(
        (a[key].float() - b[key].float()).abs().sum().item()
        for key in a
    )

    print(f"\n{name_a} vs {name_b}")
    print("Identical weights:", identical)
    print("Total absolute weight difference:", total_difference)