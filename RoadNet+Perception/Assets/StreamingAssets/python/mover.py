import random
import numpy as np
import UnityEngine as u

def oscillate(go, step=0.1):
    pos = go.transform.position
    x = pos.x

    direction = 1

    go.transform.position = u.Vector3(x + direction * step * u.Time.deltaTime,
                                      pos.y,
                                      pos.z)

    return random.randint(0, 100) + np.pi