import math

def main():
    n1 = int(input())
    A = [tuple(map(float, input().split())) for _ in range(n1)]
    n2 = int(input())
    B = [tuple(map(float, input().split())) for _ in range(n2)]
    px, py = map(float, input().split())
    
    A = [(x - px, y - py) for x, y in A]
    B = [(x - px, y - py) for x, y in B]
    
    eps = 1e-9
    angles = set()
    two_pi = 2 * math.pi
    
    for x, y in A + B:
        angles.add(math.atan2(y, x))
    
    for ax, ay in A:
        for bx, by in B:
            sx, sy = ax + bx, ay + by
            if abs(sx) < eps and abs(sy) < eps:
                continue
            ang = math.atan2(-sx, sy)
            angles.add(ang)
            angles.add(ang + math.pi)
    
    candidates = []
    for ang in angles:
        for delta in [0, -1e-9, 1e-9]:
            theta = ang + delta
            candidates.append(theta)
    
    best = -1.0
    
    for theta in candidates:
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        projA = [ax * nx + ay * ny for ax, ay in A]
        projB = [bx * nx + by * ny for bx, by in B]
        
        minA, maxA = min(projA), max(projA)
        minB, maxB = min(projB), max(projB)
        
        valid = False
        current_min_dist = float('inf')
        
        if maxA <= eps and minB >= -eps and maxA < minB:
            current_min_dist = min(-maxA, minB)
            valid = True
        elif maxB < -eps and minA > eps:
            current_min_dist = min(-maxB, minA)
            valid = True
        
        if valid and current_min_dist > best:
            best = current_min_dist
    
    if best < 0:
        print(-1)
    else:
        print(best)

if __name__ == '__main__':
    main()