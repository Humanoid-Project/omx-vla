import math

JOINT1_XY = (-0.01125, 0.0)
SHOULDER_Z = 0.034 + 0.0635
UPPER_ARM = (0.0415, 0.11315)
FOREARM = 0.162
TOOL = 0.0287 + 0.09193


def _rot(theta, v):
    return (v[0] * math.cos(theta) + v[1] * math.sin(theta), -v[0] * math.sin(theta) + v[1] * math.cos(theta))


def forward(q):
    q1, q2, q3, q4 = q[:4]
    a = _rot(q2, UPPER_ARM)
    b = _rot(q2 + q3, (FOREARM, 0.0))
    c = _rot(q2 + q3 + q4, (TOOL, 0.0))
    r = a[0] + b[0] + c[0]
    z = SHOULDER_Z + a[1] + b[1] + c[1]
    return (JOINT1_XY[0] + r * math.cos(q1), JOINT1_XY[1] + r * math.sin(q1), z)


def inverse_top_down(x, y, z, tool_pitch=math.pi / 2):
    dx, dy = x - JOINT1_XY[0], y - JOINT1_XY[1]
    q1 = math.atan2(dy, dx)
    r = math.hypot(dx, dy)
    tool = _rot(tool_pitch, (TOOL, 0.0))
    wr, wz = r - tool[0], z - tool[1] - SHOULDER_Z
    l1 = math.hypot(*UPPER_ARM)
    off = math.atan2(UPPER_ARM[1], UPPER_ARM[0])
    d2 = wr * wr + wz * wz
    c = (d2 - l1 * l1 - FOREARM * FOREARM) / (2 * l1 * FOREARM)
    if abs(c) > 1.0:
        raise ValueError(f"target out of reach: {(x, y, z)}")
    solutions = []
    for elbow in (1.0, -1.0):
        phi = elbow * math.acos(c)
        k1, k2 = l1 + FOREARM * math.cos(phi), FOREARM * math.sin(phi)
        alpha = math.atan2(wz, wr) + math.atan2(k2, k1)
        q2 = off - alpha
        q3 = phi - off
        solutions.append((alpha, (q1, q2, q3, tool_pitch - q2 - q3)))
    return max(solutions)[1]
