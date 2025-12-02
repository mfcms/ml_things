import math

def main():
    n1 = int(input())
    A = [tuple(map(float, input().split())) for _ in range(n1)]
    n2 = int(input())
    B = [tuple(map(float, input().split())) for _ in range(n2)]
    px, py = map(float, input().split())
    
    A = [(x - px, y - py) for x, y in A]
    B = [(x - px, y - py) for x, y in B]
    
    def check(theta):
        nx, ny = math.cos(theta), math.sin(theta)
        minA = min(nx*x + ny*y for x, y in A)
        maxA = max(nx*x + ny*y for x, y in A)
        minB = min(nx*x + ny*y for x, y in B)
        maxB = max(nx*x + ny*y for x, y in B)
        if maxA < 0 and minB > 0:
            return min(-maxA, minB)
        if maxB < 0 and minA > 0:
            return min(-maxB, minA)
        return -1
    
    angles = []
    for x, y in A + B:
        angles.append(math.atan2(y, x))
        angles.append(math.atan2(-y, -x))
    for i in range(len(A)):
        for j in range(len(B)):
            x1, y1 = A[i]
            x2, y2 = B[j]
            a = x1 + x2
            b = y1 + y2
            if abs(a) > 1e-10 or abs(b) > 1e-10:
                ang = math.atan2(-a, b)
                angles.append(ang)
                angles.append(ang + math.pi)
    angles = list(set(angles))
    best = -1
    for ang in angles:
        for da in [0, 1e-7, -1e-7]:
            val = check(ang + da)
            if val > best:
                best = val
    if best < 0:
        print(-1)
    else:
        print(best)

if __name__ == '__main__':
    main()