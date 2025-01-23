def cal_loss1(outputs, secret, length=100, k=8):
    t = 0
    for i in range(k):
        t += outputs[0][i * length]
    loss = (t - k * int(secret[0])) * (t - k * int(secret[0]))

    for i in range(1, length):
        t = 0
        for j in range(k):
            t += outputs[0][i + j * length]
        loss += (t - k * int(secret[i])) * (t - k * int(secret[i]))

    return loss

def cal_loss2(output1, output2):
    length = len(output1)
    loss = (output1[0][0] - output2[0][0]) * (output1[0][0] - output2[0][0])

    for i in range(1, length):
        loss += (output1[0][i] - output2[0][i]) * (output1[0][i] - output2[0][i])

    return loss

def show_false(outputs, secret, length, k=2):
    count = 0
    count0 = 0
    count1 = 0
    count_t = 0
    all = 0

    for i in range(length):
        t = 0
        
        for j in range(k):
            t += outputs[0][i + j * length]
        t = t.item()
        all = all + t

        if t>0.3:
            count_t = count_t + 1
        if secret[i] == '0' and t>0.8 :
            count = count + 1
            count0 = count0 + 1
        if secret[i] == '1' and t<0.8 :
            count = count + 1
            count1 = count1 + 1

    return count, count0, count1, count_t, all/length

