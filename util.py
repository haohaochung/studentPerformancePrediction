import matplotlib.pyplot as plt

def plotPredictScore(predict, real):
    plt.scatter(predict, real,s=2)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'k--', lw=2)
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.show()

def timeAffectMeasure(predict, real):
    good =0
    bad = 0
    soso =0
    threshold = 5
    for i in range(len(predict)):
        if abs(predict[i] - real[i]) < threshold:
            soso+=1
        elif abs(predict[i] - real[i]) >= threshold:
            if (predict[i]>real[i]):
                bad+=1
            else:
                good+=1
        elif predict[i] == real[i]:
            print("awosome")
    try:
        z = good/bad
    except ZeroDivisionError:
        z =0
    print("Good: {0}, Bad: {1}, Soso: {2}, Good/Bad: {3}, length: {4}" .format(good, bad, soso, z, len(predict)))


def flattenList(trainingCourse):
    if any(isinstance(i, list) for i in trainingCourse):  # is nested list
        flat_list = [item for sublist in trainingCourse for item in sublist]
    else:
        flat_list = trainingCourse

    return flat_list

def measure(x, y): # x:predcit, y:real
    a = 0
    b = 0
    c = 0
    d = 0

    for i in range(len(x)):
        if (x[i] >= 60) & (y[i] >= 60):
            d += 1
        elif (x[i] < 60) & (y[i] < 60):
            a += 1
        elif (x[i] >= 60) & (y[i] < 60):
            b += 1
        elif (x[i] < 60) & (y[i] >= 60):
            c += 1
    print(a, b, c, d)
    precision = a / (a + c)
    recall = a / (a + b)
    accuracy = (a + d) / (a + b + c + d)
    f_score = 2 * precision * recall / (precision + recall)

    print(accuracy, precision, recall, f_score)