import matplotlib.pyplot as plt

def print_scatter(y_score):
    x_values = [ i for i in range(len(y_score))]
    plt.scatter(y_score, x_values)
    plt.title('Simple Scatter Plot')
    plt.xlabel('score')
    plt.ylabel('image index')

    # 显示图表
    plt.show()

if __name__=="__main__":
    y_score=[0.8, 0.6, 0.5, 2, 0.4]
    print_scatter(y_score)