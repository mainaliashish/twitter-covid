from sklearn.datasets import load_iris
import os

if __name__ == '__main__':
    write_path = os.path.join("data", "iris.csv")
    iris = load_iris()
    print(iris.data)

    # data = pd.read_csv(write_path)
    # df = pd.DataFrame(data)
    # print(df.shape)
