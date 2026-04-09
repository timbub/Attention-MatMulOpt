import sys
import random

def generate_tensor(b, s, d):
    res = f"{b} {s} {d}\n"
    for _ in range(b * s):
        row = [str(round(random.uniform(-1, 1), 2)) for _ in range(d)]
        res += " ".join(row) + "\n"
    return res + "\n"

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "input.txt"

    B, Sq, Sk, Dk, Dv = 100, 20, 30, 40, 50

    with open(filename, "w") as f:
        f.write(generate_tensor(B, Sq, Dk))
        f.write(generate_tensor(B, Sk, Dk))
        f.write(generate_tensor(B, Sk, Dv))
    print(f"output tensor size: {B*Sq*Dv}")
