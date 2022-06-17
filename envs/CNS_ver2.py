import socket
import time
from struct import *
from time import sleep
import numpy as np


#   통신 설정
CNS_IP = "10.74.54.121"
Remote_IP = '10.74.54.122'
UDP_PORT_LIST = [7000, 7002, 7004, 7006, 7008,
                 7010, 7012, 7014, 7016, 7018]


#   소켓 통신 설정
'''
sockets 는 listen 전용, 포트별(=CNS별) 하나씩
sock 은 send 전용
'''
sockets = []
for i in range(len(UDP_PORT_LIST)):
    sockets.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
    sockets[i].bind((Remote_IP, UDP_PORT_LIST[i]))
    sockets[i].settimeout(0.3)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class FastCNS:

    def __init__(self):
        self.parameter_list =  np.loadtxt('dat/list.dat', delimiter=',', dtype=str)
        self.msg = [0] * int(len(UDP_PORT_LIST))                                    # 사전할당, 값 0은 의미 없음
        self.counter = 0

    def one_sec(self, cns=1):
        global sockets
        while True:
            try:
                one_step(cns=cns)
                self.msg[cns-1], addr = sockets[cns-1].recvfrom(60008)
                break
            except socket.timeout:                                                      # 가끔 있는 반응 없는 경우 대비
                pass

    def read(self, want_to_know, cns=1):

        self.loc = []
        self.type = []

        for i in want_to_know:
            loc = np.where(self.parameter_list[:, 0] == i)[0]
            # 아래 줄에서 에러 발생하면 변수명을 확인하세요.
            try:
                self.loc.append(int(loc))
            except TypeError:
                print('Check a parameter name of ', i)
            self.type.append(int(self.parameter_list[loc, 1]))

        self.value = []

        for i, j in zip(self.loc, self.type):
             self.value.append(list(unpack('ififif', self.msg[cns-1][4 + 24 * (i): 4 + 24 * (i + 1)]))[4 + j])

        return np.array(self.value)


def initial_condition(num, cns=1):
    sleep(0.001)
    msg = pack('11if32s32s', 2, 0, int(num)+1, 0, 0, 0, 0, 1, 0, 0, 0, 0.0, b"Not needed", b"Not needed")
    sock.sendto(msg, (CNS_IP, UDP_PORT_LIST[cns - 1]))

    msg = pack('11if32s32s', 7, 0, 0, 0, 0, 0, 0, 0, int(0), 0, 0, 0.0, 'KCNTOMS'.encode('utf-8'), b"Not needed")
    sock.sendto(msg, (CNS_IP, UDP_PORT_LIST[cns - 1]))

    #print('CNS #', str(cns), ' : IC = #', str(num))

    sleep(3)


def snapshot(num, description, cns=1):
    sleep(0.01)

    msg = pack('11if32s32s', 3, 0, 0, num, 0, 0, 0, 0, 0, 0, 0, 0.0, b"Not needed", str(description).encode('utf-8'))
    sock.sendto(msg, (CNS_IP, UDP_PORT_LIST[cns - 1]))
    print('CNS #', str(cns), ' : Snapshot Save, IC = #', str(num), ' (', description, ')')


def malfunction(mal_fun, mal_option, mal_delay, cns=1):
    sleep(0.01)

    msg = pack('11if32s32s', 4, 0, 0, 0,  int(mal_fun), int(mal_option), int(mal_delay), 0, 0, 0, 0, 0.0, b"Not needed", b"Not needed")
    sock.sendto(msg, (CNS_IP, UDP_PORT_LIST[cns - 1]))
    print('CNS #', str(cns), ' : Malfunction, #', str(mal_fun), ' (',  str(mal_option), '). Time Delay =', str(mal_delay))

    sleep(0.01)


def time_scale(scale, cns=1):
    sleep(0.01)

    msg = pack('11if32s32s', 5, 0, 0, 0, 0, 0, 0, scale, 0, 0, 0, 0.0, b"Not needed", b"Not needed")
    sock.sendto(msg, (CNS_IP, UDP_PORT_LIST[cns - 1]))
    scale_value = ['0.1 REAL TIME', 'REAL TIME', '5 REAL TIME', '50 REAL TIME', '150 REAL TIME']
    print('CNS #', str(cns), ' : TIME SCALE =', scale_value[scale])

    sleep(0.01)


def integer_set(pid, value, cns=1):
    sleep(0.001)
    msg = pack('11if32s32s', 7, 0, 0, 0, 0, 0, 0, 0, int(value), 0, 0, 0.0, pid.encode('utf-8'), b"Not needed")
    sock.sendto(msg, (CNS_IP, UDP_PORT_LIST[cns - 1]))
    #print('CNS #', str(cns), ' : SET ', str(pid), ' as ', str(value))


def real_set(pid, value, cns=1):
    sleep(0.001)
    msg = pack('11if32s32s', 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(value), pid.encode('utf-8'), b"Not needed")
    sock.sendto(msg, (CNS_IP, UDP_PORT_LIST[cns - 1]))
    #print('CNS #', str(cns), ' : SET ', str(pid), ' as ', str(value))

def one_step(cns=1):
    sleep(0.001)
    msg = pack('11if32s32s', 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, b"N", b"N")
    sock.sendto(msg, (CNS_IP, UDP_PORT_LIST[cns - 1]))
