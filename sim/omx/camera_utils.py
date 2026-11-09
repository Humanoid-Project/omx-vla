import math


def look_at_quat_world(eye, target, up=(0.0, 0.0, 1.0)):
    fx, fy, fz = (target[i] - eye[i] for i in range(3))
    n = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / n, fy / n, fz / n
    lx, ly, lz = up[1] * fz - up[2] * fy, up[2] * fx - up[0] * fz, up[0] * fy - up[1] * fx
    n = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / n, ly / n, lz / n
    ux, uy, uz = fy * lz - fz * ly, fz * lx - fx * lz, fx * ly - fy * lx
    m = ((fx, lx, ux), (fy, ly, uy), (fz, lz, uz))
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    return (w, x, y, z)


def focal_length_for_hfov(hfov_deg, horizontal_aperture):
    return horizontal_aperture / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
