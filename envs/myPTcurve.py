import numpy

def check_PT(P, T):
    status = 0
    # lower boundary
    TnPs = [[37, 17], [149, 17], [159, 17.3], [169, 17.6], [179, 20], [204, 31.6], [232, 44], [260, 58], [287.7, 71], [350., 100]]

    for idx, value in enumerate(TnPs):
        if T < value[0]:
            boundary = (TnPs[idx][1] - TnPs[idx-1][1])/(TnPs[idx][0] - TnPs[idx-1][0]) * (T - TnPs[idx][0]) + TnPs[idx][1]
            if P < boundary:
                status = -1
            break

    # upper boundary
    TnPs = [[37, 29.5], [65.5, 30.5], [93, 36.5], [104, 42], [110, 45.6], [115.5, 49], [121, 54.2], [148, 105], [176.5, 176], [186.5, 200]]

    for idx, value in enumerate(TnPs):
        if T < value[0]:
            boundary = (TnPs[idx][1] - TnPs[idx-1][1])/(TnPs[idx][0] - TnPs[idx-1][0]) * (T - TnPs[idx][0]) + TnPs[idx][1]
            if P > boundary:
                status = +1
            break

    return status

