import sys
import random

def generate_tensor(b, s, d):
    res = f"{b} {s} {d}\n"
    for _ in range(b * s):
        row = [str(round(random.uniform(-1, 1), 2)) for _ in range(d)]
        res += " ".join(row) + "\n"
    return res + "\n"

if __name__ == "__main__":
    B, Sq, Sk, Dk, Dv = 1, 2048, 2048, 2048, 2048

    filename = f"{B*Sq*Dv}.txt"

    with open(filename, "w") as f:
        f.write(generate_tensor(B, Sq, Dk))
        f.write(generate_tensor(B, Sk, Dk))
        f.write(generate_tensor(B, Sk, Dv))
    print(f"output tensor size: {B*Sq*Dv}")
